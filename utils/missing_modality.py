# -*- coding: utf-8 -*-
"""
模态 mask 工具（Stage-1 / 2 / 3 共用）。

模态顺序固定：0: CT，1: WSI_intra（冰冻病理特征），2: Table
mask 语义：True / 1 = 该模态存在，False / 0 = 该模态缺失。
本流程所有样本均为全模态；mask 仍按样本下标保存为 fixed_missing_masks.pt，
训练时的随机模态 dropout 在其基础上进行（utils.pipeline.random_modality_drop）。
"""

from typing import Optional, Tuple, Sequence, Union

import torch
from torch.utils.data import Dataset


MODALITY_NAMES = ("CT", "WSI_intra", "Table")

# 与 enumerate_non_empty_masks(include_full=True) 的行顺序一一对应
MASK_PATTERN_NAMES = ("CT", "WSI", "Table", "CT+WSI", "CT+Table", "WSI+Table", "CT+WSI+Table")


def enumerate_non_empty_masks(
    include_full: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    返回三模态所有非空组合。

    include_full=True 时返回 7 种：
        [1,0,0]
        [0,1,0]
        [0,0,1]
        [1,1,0]
        [1,0,1]
        [0,1,1]
        [1,1,1]

    include_full=False 时返回除 [1,1,1] 外的 6 种：
        [1,0,0]
        [0,1,0]
        [0,0,1]
        [1,1,0]
        [1,0,1]
        [0,1,1]
    """
    masks = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]

    if include_full:
        masks.append([1, 1, 1])

    return torch.tensor(masks, dtype=dtype, device=device)


class FixedMissingMaskManager:
    """每个样本的固定模态 mask [N,3]（全模态）。按 full_ds 中的原始样本编号取 mask。"""

    def __init__(self, num_samples: int):
        self.num_samples = int(num_samples)
        self.masks = torch.ones(self.num_samples, 3, dtype=torch.bool)

    def get(self, indices: Union[torch.Tensor, Sequence[int]], device: Optional[torch.device] = None) -> torch.Tensor:
        idx = indices.detach().cpu().long().view(-1) if torch.is_tensor(indices) else \
            torch.as_tensor(indices, dtype=torch.long).view(-1)
        if idx.numel() == 0:
            raise ValueError("indices is empty.")
        if int(idx.min()) < 0 or int(idx.max()) >= self.num_samples:
            raise IndexError(f"indices out of range: min={int(idx.min())}, max={int(idx.max())}, "
                             f"num_samples={self.num_samples}")
        out = self.masks.index_select(0, idx)
        return (out.to(device) if device is not None else out).bool()

    def summary(self) -> dict:
        """各模态组合的样本数（写日志用）。"""
        patterns = enumerate_non_empty_masks(include_full=True, dtype=torch.bool)
        return {
            name: int((self.masks == pat.view(1, 3)).all(dim=1).sum().item())
            for name, pat in zip(MASK_PATTERN_NAMES, patterns)
        }


class IndexedSubset(Dataset):
    """
    Subset 的增强版：返回样本时额外带上原始 full_ds 的 index。

    这样即使 DataLoader shuffle=True，也能根据原始样本编号取固定缺失 mask。
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(map(int, indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        original_idx = self.indices[item]
        sample = self.dataset[original_idx]

        if not isinstance(sample, dict):
            raise TypeError(
                "IndexedSubset expects the base dataset to return a dict, "
                f"but got {type(sample)}"
            )

        sample = dict(sample)
        sample["__sample_index__"] = torch.tensor(original_idx, dtype=torch.long)
        return sample


def ensure_wsi_b1m(wsi_feat: torch.Tensor) -> torch.Tensor:
    """WSI_intra 特征 [B,m] 或 [B,1,m] -> [B,1,m]。"""
    if wsi_feat.dim() == 2:
        return wsi_feat.unsqueeze(1)

    if wsi_feat.dim() == 3 and wsi_feat.size(1) == 1:
        return wsi_feat

    raise ValueError(f"WSI feature must be [B,m] or [B,1,m], got {tuple(wsi_feat.shape)}")


def apply_missing_to_inputs(
    CT: torch.Tensor,
    WSI_intra_feat: torch.Tensor,
    Table_num: torch.Tensor,
    Table_cat: Optional[torch.Tensor],
    modality_mask: torch.Tensor,
    zero_ct_input: bool = False,
    zero_table_input: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    根据 modality_mask 构造缺失模态输入。

    - WSI_intra 是预提取特征，缺失时在这里置零。
    - CT / Table 默认保留原始输入：StageAFeatureExtractor 会在特征层按 mask 置零；
      需要输入层也置零时设 zero_ct_input / zero_table_input=True。
    返回 (CT, WSI_intra [B,1,m], Table_num, Table_cat)。

    modality_mask:
        [B,3] bool，顺序 [CT, WSI_intra, Table]。
    """
    if modality_mask.dim() != 2 or modality_mask.size(1) != 3:
        raise ValueError(f"modality_mask must be [B,3], got {tuple(modality_mask.shape)}")

    B = modality_mask.size(0)
    if CT.size(0) != B or WSI_intra_feat.size(0) != B or Table_num.size(0) != B:
        raise ValueError(
            "Batch size mismatch among CT, WSI_intra_feat, Table_num and modality_mask."
        )

    mask = modality_mask.bool()

    CT_out = CT
    WSI_out = ensure_wsi_b1m(WSI_intra_feat)
    Table_num_out = Table_num
    Table_cat_out = Table_cat

    # WSI feature 缺失时置零。
    wsi_keep = mask[:, 1].view(B, 1, 1).to(WSI_out.dtype)
    WSI_out = WSI_out * wsi_keep

    # 可选：CT 输入层置零。
    if zero_ct_input:
        ct_keep_shape = [B] + [1] * (CT.dim() - 1)
        ct_keep = mask[:, 0].view(*ct_keep_shape).to(CT.dtype)
        CT_out = CT * ct_keep

    # 可选：Table 输入层置零。
    if zero_table_input:
        table_keep = mask[:, 2].view(B, 1).to(Table_num.dtype)
        Table_num_out = Table_num * table_keep

        # 类别特征缺失时置 0。注意这要求 0 是合法类别 id。
        if Table_cat_out is not None:
            Table_cat_out = Table_cat_out.clone()
            Table_cat_out[~mask[:, 2]] = 0

    return CT_out, WSI_out, Table_num_out, Table_cat_out
