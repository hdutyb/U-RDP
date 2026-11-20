# -*- coding: utf-8 -*-
"""
CT 单模态数据集（train_encoder.py --modality ct 用）。
"""
import os
from typing import Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data._common import load_ct_volume


class CTNPYFromXLSX(Dataset):
    """
    由 Excel 的“编号”列定位 npy_dir/<编号大写>.npy；只保留文件存在且标签合法的行（保持原行顺序）。

    __getitem__ 返回 tuple：
        x FloatTensor [1, 49, 57, 57]（C=1, T=49, H=57, W=57）
        y FloatTensor one-hot [num_classes]（按 label_names 顺序）
    """

    def __init__(
        self,
        npy_dir: str = "",
        xlsx_path: str = "",
        id_col: str = "编号",
        label_col: str = "术后病理诊断类型",
        label_names: Sequence[str] = ("AAH/AIS", "MIA", "AC"),
        use_zscore: bool = True,
        window: Optional[Tuple[float, float]] = None,       # 设置后先 clip 再归一化，例如 (-1000, 400)
        to_112: bool = False,                               # H,W 插值到 112
        global_mean: Optional[float] = None,                # 提供全局 mean/std 时用全局 z-score，否则逐 volume
        global_std: Optional[float] = None,
        minmax_range: Tuple[float, float] = (0.0, 1.0),     # use_zscore=False 时 min-max 到该区间
        verbose: bool = True,
    ):
        super().__init__()
        self.npy_dir = npy_dir
        self.xlsx_path = xlsx_path
        self.id_col = id_col
        self.label_col = label_col

        self.label_names = [str(s) for s in label_names]
        self.label2idx = {name: i for i, name in enumerate(self.label_names)}
        self.idx2label = {i: name for name, i in self.label2idx.items()}

        df = pd.read_excel(self.xlsx_path)
        if id_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Excel 缺少必要列：{id_col} / {label_col}")

        df[id_col] = df[id_col].astype(str).str.strip().str.upper()
        df[label_col] = df[label_col].astype(str).str.strip()
        df["__npy_path__"] = df[id_col].apply(lambda s: os.path.join(self.npy_dir, f"{s}.npy"))

        # 筛选：文件存在 & 标签合法
        df["__exists__"] = df["__npy_path__"].apply(os.path.isfile)
        df["__label_ok__"] = df[label_col].isin(self.label_names)
        kept = df[df["__exists__"] & df["__label_ok__"]].copy()

        if verbose:
            print(f"[CTNPYFromXLSX] 总行数: {len(df)}；保留: {len(kept)}；"
                  f"缺文件: {(~df['__exists__']).sum()}；非法标签: {(~df['__label_ok__']).sum()}")

        self.df = kept.reset_index(drop=True)   # 供 split_dataset_stratified(_kfold) 使用
        self.paths = self.df["__npy_path__"].tolist()

        self.use_zscore = use_zscore
        self.window = window
        self.to_112 = to_112
        self.global_mean = global_mean
        self.global_std = global_std
        self.minmax_range = minmax_range

    def __len__(self):
        return len(self.df)

    def _load_npy_as_tensor(self, path: str) -> torch.Tensor:
        return load_ct_volume(path, use_zscore=self.use_zscore, window=self.window, to_112=self.to_112,
                              global_mean=self.global_mean, global_std=self.global_std,
                              minmax_range=self.minmax_range)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._load_npy_as_tensor(row["__npy_path__"])
        y_idx = self.label2idx[str(row[self.label_col]).strip()]
        y_onehot = F.one_hot(torch.tensor(y_idx, dtype=torch.long), num_classes=len(self.label_names)).to(torch.float32)
        return x, y_onehot
