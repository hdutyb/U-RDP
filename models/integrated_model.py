# -*- coding: utf-8 -*-
"""
PostOpGenAndCls：Stage-2 / Stage-3 共用的完整模型。

    stage_a        StageAFeatureExtractor  CT / 冰冻 WSI / 表格 -> condition_vector [B,1,m]（Stage-1 训练）
    anchor         AnchorPredictor         冰冻 WSI -> z_anchor [B,m]（train.py anchor 训练，冻结）
    diff           GaussianDiffusion1D     condition_vector -> 残差 r̂，x0_gen = z_anchor + r̂ [B,m]（Stage-2 训练）
    gen_classifier LinearHead              x0_gen -> logits（Stage-2 后半段训练）
    TitanAgg       TitanWithAggregator     真实术后 WSI 多切片特征 -> x0_real [B,m]（扩散目标，冻结）
    cmaa           CMAA                    [CT, 冰冻 WSI, x0_gen, 表格] 四模态融合分类（Stage-3 训练，只用 .fusion）

modality_mask 顺序：Stage-A 为 [CT, WSI_intra, Table]；CMAA 为 [CT, WSI_intra, x0_gen, Table]。
"""
import json
from typing import Optional, Dict

import torch
import torch.nn as nn

from config.configs import (
    CTEncoderConfig,
    WSIEncoderConfig,
    TableEncoderConfig,
    TriFusionConfig,
    DiffusionConfig,
    DenoiserPriorConfig,
)

from models.FourModalFusionNetwork import CMAA
from models.diffusion.denoiser_DALL import CondDenoiser_Prior
from models.featureExtraction import StageAFeatureExtractor
from models.diffusion.generator import GaussianDiffusion1D
from models.common.common import LinearHead, TitanWithAggregator
from utils.utils import FocalCrossEntropyLoss


criterion = FocalCrossEntropyLoss(
    gamma=2.0,
    weight=None,
    label_smoothing=0.05,
    reduction="mean",
)


def _to_cmaa_mask(mask3: torch.Tensor) -> torch.Tensor:
    """
    Stage-A mask [B,3] = [CT, WSI_intra, Table]
      -> CMAA mask [B,4] = [CT, WSI_intra, x0_gen, Table]（生成的伪术后 WSI 恒为可用）
    """
    mask3 = mask3.bool()
    return torch.stack(
        [mask3[:, 0], mask3[:, 1], torch.ones_like(mask3[:, 0]), mask3[:, 2]],
        dim=1,
    )


class PostOpGenAndCls(nn.Module):
    """
    完整流程：

    Stage A: Feature Extraction
        CT / Frozen Section feature / EHR
        -> CT / Table encoder
        -> tri-modal CMAA
        -> condition_vector [B,1,m]

    Stage B: Pathological Feature Generation
        frozen-section feature -> Anchor Predictor -> z_anchor
        condition_vector [B,1,m] -> CPD / residual diffusion -> r̂
        -> pseudo postoperative WSI feature x0_gen = z_anchor + r̂ [B,m]

    Stage C: Lung Adenocarcinoma Invasiveness Prediction
        CT_feature [B,m]
        + intraoperative WSI feature [B,m]
        + pseudo postoperative WSI feature x0_gen [B,m]
        + Table_feature [B,m]
        -> four-modal fusion
        -> classification
    """

    def __init__(
        self,
        cfg,
        ct_cfg,
        wsi_cfg,
        table_cfg,
        diff_cfg,
        tri_cfg,
    ):
        super().__init__()

        self.cfg = cfg

        # =====================================================
        # Stage A: 三模态条件编码器
        # =====================================================
        self.stage_a = StageAFeatureExtractor(
            cfg=cfg,
            ct_cfg=ct_cfg,
            wsi_cfg=wsi_cfg,
            table_cfg=table_cfg,
            tri_cfg=tri_cfg,
        )

        # =====================================================
        # Stage B: CPD / diffusion
        # =====================================================
        self.denoiser = CondDenoiser_Prior(DenoiserPriorConfig())
        self.diff = GaussianDiffusion1D(self.denoiser, diff_cfg)

        # 扩散在标准化空间中进行：x0_norm = (x0 - mean) / std，统计量按折在训练集上预先计算
        if diff_cfg.x0_stats_path is not None:
            print(f"[Diffusion] Loading x0 mean/std from {diff_cfg.x0_stats_path}")

            with open(diff_cfg.x0_stats_path, "r") as f:
                stats = json.load(f)

            mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1)
            std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1)

            device = next(self.diff.parameters()).device
            mean, std = mean.to(device), std.to(device)

            assert mean.shape[-1] == diff_cfg.m, \
                f"mean/std dim mismatch: {mean.shape[-1]} vs diffusion.m={diff_cfg.m}"

            self.diff.set_norm(mean, std)
            print(f"[Diffusion] mean={mean.mean():.4f}, std={std.mean():.4f}")

        # Stage B 生成侧分类头
        self.gen_classifier = LinearHead(
            in_dim=cfg.m,
            hidden_dim=cfg.hidden,
            out_dim=cfg.num_classes,
        )

        # =====================================================
        # Stage C: 四模态融合
        # 注意：CMAA 内部还会构建自己的 CTEncoder / TableEncoder / TitanAgg，
        # 但 Stage-3 只调用 self.cmaa.fusion(...)，这些子模块不参与前向（保存在 checkpoint 的 cmaa.* 下）。
        # CMAA 权重由 Stage-3 的 --fusion_ckpt（train.py fourmodal 的结果）初始化。
        # =====================================================
        self.cmaa = CMAA(
            ct_config=ct_cfg,
            table_config=table_cfg,
            fusion_cfg=tri_cfg,
            wsi_agg=wsi_cfg.agg,
        )

        # 真实术后 WSI 聚合器：多切片 [N,768] -> x0_real [1,768]，权重由 load_titanagg_from_ckpt 加载
        self.TitanAgg = TitanWithAggregator(agg=wsi_cfg.agg)

        # 残差生成（enable_anchor 后才存在）：扩散模型生成 r = x0 - z_anchor，输出 x0_gen = z_anchor + r̂
        self.anchor = None

    def enable_anchor(self, mode: str, dropout: float = 0.2, hidden: int = 768):
        """开启残差生成（models/diffusion/anchor.py）。之后 diff 的 mean/std 应为残差 r 的统计量。"""
        from models.diffusion.anchor import AnchorPredictor
        self.anchor = AnchorPredictor(self.cfg.m, mode, dropout, hidden).to(next(self.parameters()).device)

    def _cond(self, stage_a_out):
        """扩散模型条件 [B,1,m]。"""
        return stage_a_out["condition_vector"]

    def _anchor_base(self, stage_a_out):
        """残差基底 z_anchor [B,m]（float32，不求梯度）；未开启残差生成时为 None。"""
        if self.anchor is None:
            return None
        with torch.no_grad():
            return self.anchor(stage_a_out["condition_tokens"]).float()

    def sample_post(self, stage_a_out, steps, n_samples=1):
        """
        由 Stage-A 输出采样伪术后特征 x0_gen [B,m]（残差模式下加回 z_anchor）。
        n_samples=1 返回 [B,m]；n_samples>1 返回 List[[B,m]]（由调用者对预测取平均）。
        """
        base = self._anchor_base(stage_a_out)   # z_anchor，None 表示非残差模式
        cond = self._cond(stage_a_out)
        samples = [self.diff.sample(cond=cond, steps=steps) for _ in range(n_samples)]
        if base is not None:
            samples = [base + s.float() for s in samples]
        return samples[0] if n_samples == 1 else samples

    def _diff_train(self, x0_real, stage_a_out):
        """diff.train_step；残差模式下目标为 x0_real - z_anchor，返回的 x0_gen 加回 z_anchor（recon_loss 不变）。"""
        base = self._anchor_base(stage_a_out)
        cond = self._cond(stage_a_out)
        if base is None:
            return self.diff.train_step(x0=x0_real, cond=cond)
        out = self.diff.train_step(x0=x0_real.float() - base, cond=cond)
        return {**out, "x0_gen": base + out["x0_gen"].float(), "z_anchor": base}

    # ------------------------------------------------------------
    # Stage A
    # ------------------------------------------------------------
    def _forward_first(
        self,
        CT,
        WSI_intra_feat,
        Table_num,
        Table_cat,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        return self.stage_a(
            CT=CT,
            WSI_intra_feat=WSI_intra_feat,
            Table_num=Table_num,
            Table_cat=Table_cat,
            modality_mask=modality_mask,
        )

    # ------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------
    def forward(self, *args, stage: int, **kwargs) -> Dict[str, torch.Tensor]:
        if stage == 2:
            return self.forward_stage2(*args, **kwargs)
        raise ValueError("stage must be 2")

    # ------------------------------------------------------------
    # Stage-2: 训练 CPD 生成 postoperative WSI feature
    # ------------------------------------------------------------
    def forward_stage2(
        self,
        WSI_post,
        CT,
        WSI_intra_feat,
        Table_num,
        Table_cat,
        label,
        epoch=0,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        # 1) Stage A: 三模态 -> condition_vector [B,1,m]
        stage_a_out = self._forward_first(
            CT=CT,
            WSI_intra_feat=WSI_intra_feat,
            Table_num=Table_num,
            Table_cat=Table_cat,
            modality_mask=modality_mask,
        )

        cond_vec = stage_a_out["condition_vector"]  # [B,1,m]

        # 2) 真实 postoperative WSI -> x0_real [B,m]
        x0_real = self.TitanAgg(WSI_post)[1].squeeze(1)

        # 3) diffusion 训练
        diff_out = self._diff_train(x0_real, stage_a_out)

        diff_loss = diff_out["diff_loss"]
        recon_loss = diff_out["recon_loss"]
        x0_gen = diff_out["x0_gen"]  # [B,m]

        # 4) 生成特征分类
        # 训练脚本不使用这里的 logits / cls_loss，而是在外部用 x0_gen.detach() 单独更新 gen_classifier。
        # 这次前向仍保留：gen_classifier 含 dropout，删掉会改变随机数序列。
        logits_gen = self.gen_classifier(x0_gen)

        if epoch >= 30:
            cls_loss = criterion(logits_gen, label)
        else:
            cls_loss = torch.zeros((), device=x0_gen.device)

        total_loss = (
            self.cfg.w_diff_stage2 * diff_loss
            + self.cfg.w_rec_stage2 * 0.5 * recon_loss
        )

        return {
            "total_loss": total_loss,
            "diff_loss": diff_loss * self.cfg.w_diff_stage2,
            "recon_loss": recon_loss * self.cfg.w_rec_stage2,
            "cls_loss": cls_loss * self.cfg.w_main_stage2 if epoch >= 30 else torch.zeros_like(cls_loss),
            "logits": logits_gen,
            "x0_gen": x0_gen,
            "x0_real": x0_real,   # 扩散目标

            "CT_feature": stage_a_out["CT_feature"],
            "WSI_feature": stage_a_out["WSI_feature"],
            "Table_feature": stage_a_out["Table_feature"],
            "CT_feature_flat": stage_a_out["CT_feature_flat"],
            "WSI_feature_flat": stage_a_out["WSI_feature_flat"],
            "Table_feature_flat": stage_a_out["Table_feature_flat"],

            "condition_vector": cond_vec,
            "condition_flat": stage_a_out["condition_flat"],
            "modality_weights": stage_a_out["modality_weights"],
            "modality_mask": stage_a_out["modality_mask"],
        }

    # ------------------------------------------------------------
    # Stage-2 inference
    # ------------------------------------------------------------
    @torch.no_grad()
    def inference_stage2(
        self,
        CT,
        WSI_intra_feat,
        Table_num,
        Table_cat,
        steps=50,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        stage_a_out = self._forward_first(
            CT=CT,
            WSI_intra_feat=WSI_intra_feat,
            Table_num=Table_num,
            Table_cat=Table_cat,
            modality_mask=modality_mask,
        )

        cond_vec = stage_a_out["condition_vector"]  # [B,1,m]

        x0_gen = self.sample_post(stage_a_out, steps)

        logits = self.gen_classifier(x0_gen)
        probs = logits.softmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "x0_gen": x0_gen,

            "CT_feature": stage_a_out["CT_feature"],
            "WSI_feature": stage_a_out["WSI_feature"],
            "Table_feature": stage_a_out["Table_feature"],
            "CT_feature_flat": stage_a_out["CT_feature_flat"],
            "WSI_feature_flat": stage_a_out["WSI_feature_flat"],
            "Table_feature_flat": stage_a_out["Table_feature_flat"],

            "condition_vector": cond_vec,
            "condition_flat": stage_a_out["condition_flat"],
            "modality_weights": stage_a_out["modality_weights"],
            "modality_mask": stage_a_out["modality_mask"],
        }

    # ------------------------------------------------------------
    # Stage-3 inference
    # ------------------------------------------------------------
    @torch.no_grad()
    def inference_stage3(
            self,
            CT,
            WSI_intra_feat,
            Table_num,
            Table_cat,
            steps=100,
            modality_mask: Optional[torch.Tensor] = None,
    ):
        stage_a_out = self._forward_first(
            CT=CT,
            WSI_intra_feat=WSI_intra_feat,
            Table_num=Table_num,
            Table_cat=Table_cat,
            modality_mask=modality_mask,
        )

        CT_feat = stage_a_out["CT_feature_flat"]  # [B,m]
        WSI_feat = stage_a_out["WSI_feature_flat"]  # [B,m]
        Table_feat = stage_a_out["Table_feature_flat"]  # [B,m]
        cond_vec = stage_a_out["condition_vector"]  # [B,1,m]

        x0_gen = self.sample_post(stage_a_out, steps)

        # CMAA 四模态 mask [B,4] = [CT, WSI_intra, x0_gen, Table]
        mask4 = _to_cmaa_mask(stage_a_out["modality_mask"])

        fin = self.cmaa.fusion(
            CT_feat,
            WSI_feat,
            x0_gen,
            Table_feat,
            modality_mask=mask4,
        )

        logits = fin["logits"]
        probs = logits.softmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "x0_gen": x0_gen,

            "condition_vector": cond_vec,
            "condition_flat": stage_a_out["condition_flat"],
            "modality_weights": stage_a_out["modality_weights"],
            "modality_mask": stage_a_out["modality_mask"],

            # Stage-C 四模态 mask，顺序 [CT, WSI_intra, x0_gen, Table]
            "cmaa_modality_mask": mask4,

            # CMAA 权重，便于后续可视化/分析
            "modal_alpha": fin.get("modal_alpha", None),
            "modal_alpha_ct_intra_post_tab": fin.get("modal_alpha_ct_intra_post_tab", None),
        }

    @torch.no_grad()
    def inference(
        self,
        CT,
        WSI_intra_feat,
        Table_num,
        Table_cat,
        steps=100,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        return self.inference_stage3(
            CT=CT,
            WSI_intra_feat=WSI_intra_feat,
            Table_num=Table_num,
            Table_cat=Table_cat,
            steps=steps,
            modality_mask=modality_mask,
        )
