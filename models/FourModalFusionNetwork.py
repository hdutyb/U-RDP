# -*- coding: utf-8 -*-
"""
CMAA：四模态（CT / 冰冻 WSI / 术后 WSI / 表格）融合网络。

    fusion(ct_feat, intra_wsi, post_wsi, tab_feat, modality_mask)
        已编码好的四个 [B,D] 特征 -> 线性投影 -> L 层门控跨模态注意力 -> 逐模态打分加权 -> 分类
        Stage-3 只调用这一部分（post_wsi 换成扩散生成的 x0_gen）。
    forward(CT, intra_wsi, post_wsi_list, Table_num, Table_cat, modality_mask)
        端到端：内部用自带的 CTEncoder / TableEncoder / TitanAgg 编码后再 fusion
        （train.py fourmodal 使用真实术后 WSI，作为 Stage-2 的蒸馏教师）。
"""
from typing import Optional

import torch
from torch import nn

from models.common.common import Classifier, LinearHead, TitanWithAggregator
from models.common.cross_modal import CrossModalAttentionStack, ModalityScoreFusion, _linear_proj
from models.encoder.ctEncoder import CtEncoder
from models.encoder.tableEncoder import TableEncoder
from utils.utils import _load_pretrained_ct_encoder, _load_pretrained_agg, _load_pretrained_table_encoder


class ModalityScoreFusion4(ModalityScoreFusion):
    """四模态打分融合（token 顺序 [intra, post, ct, tab]）。"""

    def forward(self, intra_tok, post_tok, ct_tok, tab_tok, token_mask=None):
        return super().forward(torch.stack([intra_tok, post_tok, ct_tok, tab_tok], dim=1), token_mask)


class CMAA(nn.Module):
    """
    论文 Sec. III-C 的 CMAA（组件见 models/common/cross_modal.py）。

    forward 输入：
        CT:            [B,1,T,H,W]
        intra_wsi:     [B,768]            冰冻 WSI slide 特征
        post_wsi_list: List[List[Tensor]] 每个样本若干张术后切片特征，经 TitanAgg 聚合
        Table_num:     [B,num]
        Table_cat:     [B,cat]
    输出：dict（logits / 融合特征 / 模态权重 / 各单模态 logits）
    """
    def __init__(
        self,
        ct_config,
        table_config,
        fusion_cfg,
        # 术后 WSI 聚合器
        wsi_agg: str = "new_hybrid",

        # 预训练路径
        tab_pretrained_path: Optional[str] = None,
        ct_pretrained_path: Optional[str] = None,
        agg_pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.cfg = fusion_cfg
        d = self.cfg.input_dim

        # =============== 术后 WSI 聚合器（多切片） ===============
        self.TitanAgg = TitanWithAggregator(agg=wsi_agg)

        if agg_pretrained_path:
            _load_pretrained_agg(self.TitanAgg, agg_pretrained_path)

        # =============== CT、表格编码器 ===============
        self.CTEncoder = CtEncoder(config=ct_config)
        self.TableEncoder = TableEncoder(config=table_config)

        if ct_pretrained_path:
            _load_pretrained_ct_encoder(self.CTEncoder, ct_pretrained_path)

        if tab_pretrained_path:
            _load_pretrained_table_encoder(self.TableEncoder, tab_pretrained_path)

        L = getattr(self.cfg, "cmaa_layers", 2)
        H = getattr(self.cfg, "cmaa_heads", 8)
        drop = getattr(self.cfg, "cmaa_dropout", 0.1)
        print(f"[CMAA] L={L} H={H}")

        # =============== 四模态投影 ===============
        self.proj_ct = _linear_proj(d, d)
        self.proj_tab = _linear_proj(d, d)
        self.proj_post = _linear_proj(d, d)
        self.proj_intra = _linear_proj(d, d)

        # =============== 跨模态交互模块 ===============
        self.cmt = CrossModalAttentionStack(d_model=d, nhead=H, num_layers=L, dropout=drop)
        self.modal_fusion = ModalityScoreFusion4(d_model=d, hidden=self.cfg.cmaa_score_hidden)

        # =============== 分类 head（融合 + 单模态） ===============
        self.head_fuse = Classifier(
            tri_dim=d,
            num_classes=self.cfg.num_classes,
            dropout=self.cfg.dropout,
            hidden_ratio=self.cfg.classifier_hidden_ratio,
        )

        self.head_ct = LinearHead(in_dim=d, out_dim=self.cfg.num_classes)
        self.head_tab = LinearHead(in_dim=d, out_dim=self.cfg.num_classes)
        self.head_post = LinearHead(in_dim=768, out_dim=self.cfg.num_classes)
        self.head_intra = LinearHead(in_dim=768, out_dim=self.cfg.num_classes)


    # ---------------------------------------------------------
    # 聚合术后 WSI
    # ---------------------------------------------------------
    def _agg_post(self, post_wsi_list):
        out = []
        for feats in post_wsi_list:
            logit, z = self.TitanAgg(feats)    # z: [1,768]
            out.append(z)
        return torch.cat(out, dim=0)

    def fusion(self, ct_feat, intra_wsi, post_wsi, tab_feat, modality_mask=None):
        """
        ct_feat: [B,D]
        intra_wsi: [B,D]
        post_wsi: [B,D]
        tab_feat: [B,D]

        modality_mask:
            [B,4]，外部传入顺序为：
                [CT, intra, post, Table]

            进入 CMAA 内部后会转换为 token 顺序：
                [intra, post, CT, Table]
        """

        ct_tok = self.proj_ct(ct_feat)
        tab_tok = self.proj_tab(tab_feat)
        post_tok = self.proj_post(post_wsi)
        intra_tok = self.proj_intra(intra_wsi)

        token_mask = None

        if modality_mask is not None:
            modality_mask = modality_mask.to(ct_tok.device).bool()

            if modality_mask.dim() != 2 or modality_mask.size(1) != 4:
                raise ValueError(
                    f"modality_mask must be [B,4] in order [CT,intra,post,Table], "
                    f"but got {tuple(modality_mask.shape)}"
                )

            # 外部顺序：[CT, intra, post, Table]
            # CMAA 内部 token 顺序：[intra, post, CT, Table]
            token_mask = torch.stack(
                [
                    modality_mask[:, 1],  # intra
                    modality_mask[:, 2],  # post
                    modality_mask[:, 0],  # CT
                    modality_mask[:, 3],  # Table
                ],
                dim=1,
            )  # [B,4]

            # 投影后再次置零，避免 Linear bias 把缺失模态变成非零 token
            intra_tok = intra_tok * token_mask[:, 0:1].to(intra_tok.dtype)
            post_tok = post_tok * token_mask[:, 1:2].to(post_tok.dtype)
            ct_tok = ct_tok * token_mask[:, 2:3].to(ct_tok.dtype)
            tab_tok = tab_tok * token_mask[:, 3:4].to(tab_tok.dtype)

        tokens = torch.stack([intra_tok, post_tok, ct_tok, tab_tok], dim=1)  # [B,4,D]
        cmt_out = self.cmt(tokens, token_mask=token_mask)
        if token_mask is not None:
            cmt_out = cmt_out * token_mask.unsqueeze(-1).to(cmt_out.dtype)
        z_fuse, alpha = self.modal_fusion(
            cmt_out[:, 0], cmt_out[:, 1], cmt_out[:, 2], cmt_out[:, 3], token_mask=token_mask)

        # ========== 分类输出 ==========
        # 只有 logits 参与训练损失；单模态 logits 仅供分析。
        # 注意 head_intra / head_post 作用在投影前的原始特征上，未按 modality_mask 置零。
        logits = self.head_fuse(z_fuse)

        logits_intra = self.head_intra(intra_wsi)
        logits_post = self.head_post(post_wsi)
        logits_ct = self.head_ct(ct_tok)
        logits_tab = self.head_tab(tab_tok)

        # alpha 当前顺序是 [intra, post, CT, Table]
        # 额外给一个更直观的顺序 [CT, intra, post, Table]
        alpha_ct_intra_post_tab = torch.stack(
            [
                alpha[:, 2],
                alpha[:, 0],
                alpha[:, 1],
                alpha[:, 3],
            ],
            dim=1,
        )

        return {
            "logits": logits,
            "fusion_feature": z_fuse,

            # 原始 CMAA token 顺序：[intra, post, CT, Table]
            "modal_alpha": alpha,

            # 更直观顺序：[CT, intra, post, Table]
            "modal_alpha_ct_intra_post_tab": alpha_ct_intra_post_tab,

            "CT_feature": ct_tok,
            "Table_feature": tab_tok,
            "PostWSI_feature": post_tok,
            "IntraWSI_feature": intra_tok,

            "CT_logits": logits_ct,
            "TAB_logits": logits_tab,
            "POST_logits": logits_post,
            "INTRA_logits": logits_intra,

            "cmaa_token_mask": token_mask,
        }


    def forward(self, CT, intra_wsi, post_wsi_list, Table_num, Table_cat, modality_mask=None):
        device = next(self.parameters()).device
        # 术后切片特征通常是 CPU 上的 List[List[Tensor]]，逐个搬到模型所在设备
        self.TitanAgg.to(device)
        if hasattr(self.TitanAgg, "aggregator"):
            self.TitanAgg.aggregator.to(device)
        if isinstance(post_wsi_list, list):
            for i in range(len(post_wsi_list)):
                post_wsi_list[i] = [x.to(device) for x in post_wsi_list[i]]
        elif isinstance(post_wsi_list, torch.Tensor):
            post_wsi_list = post_wsi_list.to(device)

        # ========== 1. 术后 WSI 多片聚合 ==========
        post_wsi = self._agg_post(post_wsi_list)      # [B,768]

        # ========== 2. 冰冻（术中）WSI 单向量 ==========
        intra_wsi = intra_wsi.to(device)              # [B,768]

        # ========== 3. 编码 CT / 表格 ==========
        ct_feat = self.CTEncoder(CT)
        tab_feat = self.TableEncoder(Table_num, Table_cat)

        return self.fusion(
            ct_feat,
            intra_wsi,
            post_wsi,
            tab_feat,
            modality_mask=modality_mask,
        )
