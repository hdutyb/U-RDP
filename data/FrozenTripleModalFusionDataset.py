# -*- coding: utf-8 -*-
"""
Stage-1 三模态数据集：CT + 冰冻病理 TITAN slide 特征 [768] + 表格。
"""
import os
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data._common import encode_table, labels_to_onehot, load_ct_volume


class TriModalFromXLSX(Dataset):
    """
    样本筛选：CT .npy 与冰冻 WSI .h5 都存在，且（filter_missing=True 时）标签合法；保持 Excel 原有行顺序。
    注意：Stage-1 按样本下标保存 fixed_missing_masks.pt，Stage-2/3 用 FourModalFromXLSX 复用，
    两者的筛选结果与行顺序必须一致。

    __getitem__ 返回 dict：
        CT           FloatTensor [1, T, H, W]（默认 [1,49,57,57]）
        wsi_feat     FloatTensor [768]
        x_num        FloatTensor [n_num]
        x_cat        LongTensor  [n_cat]（无离散列时为 None）
        missing_mask BoolTensor  [n_num]
        y            FloatTensor one-hot [C]（label_col 不为 None 时）

    暴露 df / label_col / label2idx 供 split_dataset_stratified(_kfold) 使用。
    """

    def __init__(self,
                 xlsx_path: str = "",
                 ct_col: str = "__npy_path__",
                 wsi_id_col: str = "冰冻切片号",
                 wsi_dir: str = "",
                 label_col: Optional[str] = "术后病理诊断类型",
                 label_names: Sequence[str] = ("AAH/AIS", "MIA", "AC"),
                 # 表格列（编码规则见 data._common.encode_table）
                 num_cols: List[str] = (),
                 binary_cols: List[str] = (),
                 pleura_col: Optional[str] = None,
                 gender_col: Optional[str] = None,
                 density_col: Optional[str] = None,
                 # CT 预处理
                 use_zscore: bool = True,
                 window: Optional[Tuple[float, float]] = None,  # 例 (-1000, 400)
                 to_112: bool = False,
                 filter_missing: bool = True,
                 verbose: bool = True):
        super().__init__()
        self.xlsx_path = xlsx_path
        self.ct_col = ct_col
        self.wsi_id_col = wsi_id_col
        self.wsi_dir = wsi_dir
        self.label_col = label_col
        self.label_names = [str(s) for s in label_names]
        self.label2idx = {name: i for i, name in enumerate(self.label_names)}
        self.idx2label = {i: name for name, i in self.label2idx.items()}

        df = pd.read_excel(self.xlsx_path)

        require_cols = [ct_col, wsi_id_col] + ([label_col] if label_col is not None else [])
        for c in require_cols:
            if c not in df.columns:
                raise ValueError(f"[TriModalFromXLSX] Excel 缺少必需列：{c}")

        df[wsi_id_col] = df[wsi_id_col].astype(str).str.strip()
        if label_col is not None:
            df[label_col] = df[label_col].astype(str).str.strip()

        df["__ct_path__"] = df[ct_col].astype(str).str.strip()
        df["__wsi_h5__"] = df[wsi_id_col].apply(lambda s: os.path.join(wsi_dir, f"{s}.h5"))

        # 筛选：文件存在 & 标签合法
        exists_ct = df["__ct_path__"].apply(os.path.isfile)
        exists_wsi = df["__wsi_h5__"].apply(os.path.isfile)
        label_ok = df[label_col].isin(self.label_names) if label_col is not None else True
        keep_mask = exists_ct & exists_wsi & label_ok if filter_missing else (exists_ct & exists_wsi)
        kept = df[keep_mask].copy().reset_index(drop=True)

        if verbose:
            print(f"[TriModalFromXLSX] 总行数: {len(df)} / 保留: {len(kept)} "
                  f"/ 缺CT: {(~exists_ct).sum()} / 缺WSI: {(~exists_wsi).sum()}"
                  + (f" / 非法标签: {(~df[label_col].isin(self.label_names)).sum()}" if label_col else ""))

        self.x_num, self.missing_mask, self.x_cat, _, _ = encode_table(
            kept, num_cols, binary_cols, density_col=density_col, pleura_col=pleura_col, gender_col=gender_col)

        self.y = None
        if label_col is not None:
            # 兼容数字标签 0/1/2（filter_missing=True 时标签已合法，这一步不改变任何值）
            kept[label_col] = (
                kept[label_col].astype(str).str.strip().str.upper()
                .replace({"0": "AAH/AIS", "1": "AC", "2": "MIA"})
            )
            self.y = labels_to_onehot(kept[label_col], self.label2idx, len(self.label_names))

        self.df = kept
        self.paths_ct = kept["__ct_path__"].tolist()
        self.paths_wsi = kept["__wsi_h5__"].tolist()

        self.use_zscore = use_zscore
        self.window = window
        self.to_112 = to_112

    def __len__(self):
        return len(self.df)

    def _load_ct_npy_as_tensor(self, path: str) -> torch.Tensor:
        """.npy (57,57,49) -> [1,49,57,57]；use_zscore=False 时 min-max 到 [0,1]"""
        return load_ct_volume(path, use_zscore=self.use_zscore, window=self.window, to_112=self.to_112)

    @staticmethod
    def _load_wsi_feat_from_h5(h5_path: str) -> torch.Tensor:
        """读取冰冻 WSI slide 特征，要求 squeeze 后为 [768]。"""
        with h5py.File(h5_path, "r") as f:
            if "features" not in f:
                raise KeyError(f"[TriModalFromXLSX] H5中找不到 'features' 键：{h5_path}")
            arr = f["features"][()]
        arr = np.asarray(arr).squeeze()
        if arr.ndim != 1:
            raise ValueError(f"[TriModalFromXLSX] 读取到的 features 形状异常：{arr.shape}")
        if arr.shape[0] != 768:
            raise ValueError(f"[TriModalFromXLSX] features 期望 768 维，得到 {arr.shape[0]}：{h5_path}")
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx: int):
        out = {
            "CT": self._load_ct_npy_as_tensor(self.paths_ct[idx]),          # [1, T, H, W]
            "wsi_feat": self._load_wsi_feat_from_h5(self.paths_wsi[idx]),   # [768]
            "x_num": self.x_num[idx],                                        # [n_num]
            "x_cat": self.x_cat[idx] if self.x_cat is not None else None,   # [n_cat] or None
            "missing_mask": self.missing_mask[idx],                          # [n_num]
        }
        if self.y is not None:
            out["y"] = self.y[idx]                                           # one-hot [C]
        return out
