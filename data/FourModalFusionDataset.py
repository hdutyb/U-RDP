# -*- coding: utf-8 -*-
"""
Stage-2 / Stage-3 四模态数据集：CT + 表格 + 冰冻病理 slide 特征 [768] + 术后病理（多张 slide 特征）。
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data._common import encode_table, labels_to_onehot, list_case_h5, load_ct_volume, read_h5_features


class FourModalFromXLSX(Dataset):
    """
    样本筛选：CT .npy 与冰冻 WSI .h5 都存在，且（filter_missing=True 时）标签合法；保持 Excel 原有行顺序。
    筛选规则与 TriModalFromXLSX 相同，因此样本下标与 Stage-1 的 fixed_missing_masks.pt 一一对应。
    术后病理目录不参与筛选；__getitem__ 时若缺失/为空则报错（post_raise_empty=True）。

    __getitem__ 返回 dict：
        CT            FloatTensor [1, T, H, W]
        intra_wsi     FloatTensor [768]               冰冻病理（术中）slide 特征
        post_wsi_list List[FloatTensor [768]]         post_wsi_root/<术后病理前缀>/*.h5，按文件名排序
        x_num         FloatTensor [n_num]
        x_cat         LongTensor  [n_cat] 或 None
        missing_mask  BoolTensor  [n_num]
        y             FloatTensor one-hot [C]（label_col 不为 None 时）
    """

    def __init__(self,
                 xlsx_path: str,
                 ct_col: str = "__npy_path__",
                 # 术中（冰冻）病理
                 intra_wsi_id_col: str = "冰冻切片号",
                 intra_wsi_dir: str = "",
                 # 术后病理
                 post_wsi_prefix_col: str = "术后病理前缀",
                 post_wsi_root: str = "",
                 post_glob: str = "*.h5",
                 post_sort: bool = True,
                 post_raise_empty: bool = True,
                 # 标签
                 label_col: Optional[str] = "术后病理诊断类型",
                 label_names: Sequence[str] = ("AAH/AIS", "MIA", "AC"),
                 # 表格列（编码规则见 data._common.encode_table）
                 num_cols: List[str] = (),
                 binary_cols: List[str] = (),
                 pleura_col: Optional[str] = None,
                 gender_col: Optional[str] = None,
                 density_col: Optional[str] = None,
                 # CT 预处理（注意：use_zscore=False 时不做任何归一化）
                 use_zscore: bool = True,
                 window: Optional[Tuple[float, float]] = None,
                 to_112: bool = False,
                 filter_missing: bool = True,
                 verbose: bool = True,
                 ):
        super().__init__()
        self.xlsx_path = xlsx_path
        raw = pd.read_excel(self.xlsx_path)

        require_cols = [ct_col, intra_wsi_id_col, post_wsi_prefix_col] + ([label_col] if label_col else [])
        for c in require_cols:
            if c not in raw.columns:
                raise ValueError(f"Excel 缺少必需列：{c}")

        self.ct_col = ct_col
        self.intra_wsi_id_col = intra_wsi_id_col
        self.intra_wsi_dir = intra_wsi_dir
        self.post_wsi_prefix_col = post_wsi_prefix_col
        self.post_root = Path(post_wsi_root)
        self.post_glob = post_glob
        self.post_sort = post_sort
        self.post_raise_empty = post_raise_empty

        self.label_col = label_col
        self.label_names = [str(s) for s in label_names]
        self.label2idx = {n: i for i, n in enumerate(self.label_names)}

        df = raw.copy()
        df[intra_wsi_id_col] = df[intra_wsi_id_col].astype(str).str.strip()
        df[post_wsi_prefix_col] = df[post_wsi_prefix_col].astype(str).str.strip()
        if label_col:
            df[label_col] = df[label_col].astype(str).str.strip()

        df["__ct_path__"] = df[ct_col].astype(str).str.strip()
        df["__intra_h5__"] = df[intra_wsi_id_col].apply(lambda s: os.path.join(intra_wsi_dir, f"{s}.h5"))

        # 筛选：文件存在 & 标签合法（与 TriModalFromXLSX 一致）
        exists_ct = df["__ct_path__"].apply(os.path.isfile)
        exists_intra = df["__intra_h5__"].apply(os.path.isfile)
        label_ok = df[label_col].isin(self.label_names) if label_col else True
        keep = exists_ct & exists_intra & label_ok if filter_missing else (exists_ct & exists_intra)
        df = df[keep].copy().reset_index(drop=True)

        if verbose:
            print(f"[FourModalFromXLSX] 总 {len(raw)} 条 → 保留 {len(df)} 条")
            print(f"   缺CT: {(~exists_ct).sum()}  缺术中WSI: {(~exists_intra).sum()}"
                  + (f" 非法标签: {(~label_ok).sum()}" if label_col else ""))

        self.df = df
        self.paths_ct = df["__ct_path__"].tolist()
        self.paths_intra = df["__intra_h5__"].tolist()

        self.x_num, self.missing_mask, self.x_cat, _, _ = encode_table(
            df, num_cols, binary_cols, density_col=density_col, pleura_col=pleura_col, gender_col=gender_col)

        self.y = labels_to_onehot(df[label_col], self.label2idx, len(self.label_names)) if label_col else None

        self.use_zscore = use_zscore
        self.window = window
        self.to_112 = to_112

    def _load_ct(self, path: str) -> torch.Tensor:
        return load_ct_volume(path, use_zscore=self.use_zscore, window=self.window, to_112=self.to_112,
                              minmax_if_no_zscore=False)

    @staticmethod
    def _load_intra(h5_path: str) -> torch.Tensor:
        """冰冻病理 slide 特征 -> [768]"""
        with h5py.File(h5_path, "r") as f:
            arr = f["features"][()]
        return torch.tensor(np.asarray(arr).squeeze(), dtype=torch.float32)

    def _load_post_list(self, prefix: str) -> List[torch.Tensor]:
        """术后病理：同一病例前缀目录下的所有 slide 特征"""
        h5_list = list_case_h5(self.post_root / prefix, self.post_glob, self.post_sort,
                               self.post_raise_empty, case_name=prefix)
        return [read_h5_features(p) for p in h5_list]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = {
            "CT": self._load_ct(self.paths_ct[idx]),
            "intra_wsi": self._load_intra(self.paths_intra[idx]),                          # [768]
            "post_wsi_list": self._load_post_list(self.df.iloc[idx][self.post_wsi_prefix_col]),  # List[[768]]
            "x_num": self.x_num[idx],
            "x_cat": self.x_cat[idx] if self.x_cat is not None else None,
            "missing_mask": self.missing_mask[idx],
        }
        if self.y is not None:
            out["y"] = self.y[idx]
        return out

    def __len__(self):
        return len(self.df)
