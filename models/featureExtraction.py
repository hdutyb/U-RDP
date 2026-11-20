# -*- coding: utf-8 -*-
"""
Stage-A：三模态条件编码器（CT / 冰冻 WSI / 表格 -> CMAA -> condition_vector [B,1,m]）。

    CT -> CtEncoder、表格 -> TableEncoder、冰冻 WSI 直接使用 TITAN slide 特征，统一为 [B,1,m]
    线性投影 -> L 层门控跨模态注意力 -> 共享 MLP 逐模态打分 -> 加权求和 -> LayerNorm
modality_mask 顺序固定为 [CT, WSI_intra, Table]。
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

from models.encoder.ctEncoder import CtEncoder
from models.encoder.tableEncoder import TableEncoder
from models.common.cross_modal import CrossModalAttentionStack, ModalityScoreFusion, _linear_proj


class StageAFeatureExtractor(nn.Module):
    """
    输出 condition_vector [B,1,m]（Stage-2 扩散条件）、condition_tokens [B,3,m]（Anchor Predictor 输入）
    以及各模态特征 [B,m]（Stage-3 融合输入）。
    """

    def __init__(
        self,
        cfg,
        ct_cfg,
        wsi_cfg,
        table_cfg,
        tri_cfg=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.m = int(cfg.m)
        self.num_modalities = 3

        self.ct_expert = CtEncoder(config=ct_cfg)
        self.table_expert = TableEncoder(config=table_cfg)

        self.cmaa_proj = nn.ModuleList([_linear_proj(self.m, self.m) for _ in range(self.num_modalities)])
        self.cmaa_attn = CrossModalAttentionStack(d_model=self.m, nhead=getattr(cfg, "stage_a_cmaa_heads", 8),
                                                  num_layers=getattr(cfg, "stage_a_cmaa_layers", 2),
                                                  dropout=getattr(cfg, "stage_a_cmaa_dropout", 0.1))
        self.cmaa_score = ModalityScoreFusion(d_model=self.m, hidden=256)

        self.out_norm = nn.LayerNorm(self.m)

    def _ensure_b1m(self, feat: torch.Tensor, name: str) -> torch.Tensor:
        """
        将特征统一为 [B, 1, m]。

        允许输入：
            [B, m]
            [B, 1, m]

        不允许输入：
            [B, L, m] 且 L != 1
        """
        if not torch.is_tensor(feat):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(feat)}")

        if feat.dim() == 2:
            feat = feat.unsqueeze(1)  # [B,m] -> [B,1,m]

        elif feat.dim() == 3:
            if feat.size(1) != 1:
                raise ValueError(
                    f"{name} must be [B,m] or [B,1,m], "
                    f"but got {tuple(feat.shape)}"
                )
        else:
            raise ValueError(
                f"{name} must be [B,m] or [B,1,m], "
                f"but got {tuple(feat.shape)}"
            )

        if feat.size(-1) != self.m:
            raise ValueError(
                f"{name} feature dim mismatch: got {feat.size(-1)}, "
                f"expected cfg.m={self.m}"
            )

        return feat

    def _normalize_mask(
        self,
        modality_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        modality_mask:
            None: 默认三个模态都存在
            [3]: 单个 mask，自动 expand 到 batch
            [B,3]: 每个样本一个 mask

        顺序固定为：
            [CT, WSI_intra, Table]

        True / 1  表示存在；
        False / 0 表示缺失。
        """
        if modality_mask is None:
            return torch.ones(
                batch_size,
                self.num_modalities,
                dtype=torch.bool,
                device=device,
            )

        if not torch.is_tensor(modality_mask):
            modality_mask = torch.as_tensor(modality_mask, device=device)
        else:
            modality_mask = modality_mask.to(device)

        if modality_mask.dim() == 1:
            assert modality_mask.numel() == self.num_modalities, \
                f"modality_mask with dim=1 must have length {self.num_modalities}"
            modality_mask = modality_mask.unsqueeze(0).expand(batch_size, -1)

        assert modality_mask.shape == (batch_size, self.num_modalities), \
            f"modality_mask must be [B,3], got {tuple(modality_mask.shape)}"

        return modality_mask.bool()

    def forward(
        self,
        CT: torch.Tensor,
        WSI_intra_feat: torch.Tensor,
        Table_num: torch.Tensor,
        Table_cat: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # --------------------------------------------------
        # 1) 三个模态特征
        # --------------------------------------------------
        CT_feat = self.ct_expert(CT)
        Table_feat = self.table_expert(Table_num, Table_cat)

        # 统一为 [B,1,m]
        CT_feat = self._ensure_b1m(CT_feat, "CT_feat")
        WSI_feat = self._ensure_b1m(WSI_intra_feat, "WSI_intra_feat")
        Table_feat = self._ensure_b1m(Table_feat, "Table_feat")

        B = CT_feat.size(0)
        device = CT_feat.device

        mask = self._normalize_mask(modality_mask, B, device)  # [B,3]

        tokens = torch.cat(
            [CT_feat, WSI_feat, Table_feat],
            dim=1,
        )  # [B,3,m]


        # 缺失模态置零
        mask_f = mask.unsqueeze(-1).to(tokens.dtype)  # [B,3,1]
        masked_tokens = tokens * mask_f              # [B,3,m]

        # 投影 -> 门控跨模态注意力（缺失模态不作为 context）-> 逐模态打分加权
        y = torch.stack([self.cmaa_proj[i](masked_tokens[:, i]) for i in range(self.num_modalities)], dim=1)
        y = y * mask_f
        y = self.cmaa_attn(y, token_mask=mask)
        condition_flat, modality_weights = self.cmaa_score(y, token_mask=mask)
        return self._outputs(masked_tokens, mask, modality_weights, condition_flat)

    def _outputs(self, masked_tokens, mask, modality_weights, condition_flat) -> Dict[str, torch.Tensor]:
        """condition 过 out_norm；各模态特征是投影前的原始（置零后）特征。"""
        condition_flat = self.out_norm(condition_flat)

        # 第二阶段 CPD 输入：[B,1,m]
        condition_vector = condition_flat.unsqueeze(1)

        return {
            # 保持 [B,1,m]
            "CT_feature": masked_tokens[:, 0:1, :],
            "WSI_feature": masked_tokens[:, 1:2, :],
            "Table_feature": masked_tokens[:, 2:3, :],

            # 第三阶段四模态融合通常需要 [B,m]
            "CT_feature_flat": masked_tokens[:, 0, :],
            "WSI_feature_flat": masked_tokens[:, 1, :],
            "Table_feature_flat": masked_tokens[:, 2, :],

            # 第二阶段 CPD 条件输入
            "condition_vector": condition_vector,  # [B,1,m]

            # 辅助输出
            "condition_flat": condition_flat,      # [B,m]
            "condition_tokens": masked_tokens,     # [B,3,m]
            "modality_weights": modality_weights,  # [B,3]
            "modality_mask": mask,                 # [B,3]
        }
