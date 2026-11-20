# -*- coding: utf-8 -*-
"""
术后病理 WSI 数据集（多局部、多 h5，按“术后病理前缀”聚合为一个病例）。
每个 h5 为一张 slide 的 TITAN slide 级特征 [768]。
表格中同一前缀只保留第一行；前缀目录不存在时报错，目录为空时按 raise_if_empty 报错或跳过该病例。
"""
from pathlib import Path
from typing import Dict, List, Any
import glob

import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd

CLASS_NAMES = ["AAH/AIS", "MIA", "AC"]
N_CLASSES = 3

from config.paths import EXCEL as TABLE_PATH, POST_WSI_ROOT as ROOT_DIR_SLIDE

# 表格至少包含两列：术后病理前缀、术后病理诊断类型
COL_GROUP  = "术后病理前缀"          # 病例ID/子目录名
COL_LABEL  = "术后病理诊断类型"       # 标签列

# h5 根目录：<ROOT_DIR>/<术后病理前缀>/*.h5
H5_GLOB_PATTERN  = "*.h5"
SORT_H5          = True
RAISE_IF_EMPTY   = True   # 若某个前缀目录下找不到 h5，是否直接报错

class PostOpWSIPrefixDataset(Dataset):
    """
    术后病理（多局部、多 h5）Dataset（按“术后病理前缀”在 ROOT_DIR 子目录中扫描 h5）：
      - 表格每行是一个病例（一个前缀），并给出该病例的标签
      - 实际的 h5 路径通过扫描 ROOT_DIR/<prefix>/*.h5 获得
    同时暴露：
      - self.df         （每病例一行，含 label 列）
      - self.label_col  （= COL_LABEL）
      - self.label2idx  （标签到索引映射）
    以适配 utils.pipeline 的数据划分
    """
    LABELS = CLASS_NAMES

    def __init__(self,
                 table_path: str = TABLE_PATH,
                 root_dir: str = None,
                 group_col: str = COL_GROUP,
                 label_col: str = COL_LABEL,
                 h5_glob: str = H5_GLOB_PATTERN,
                 sort_h5: bool = SORT_H5,
                 raise_if_empty: bool = RAISE_IF_EMPTY):
        super().__init__()
        if root_dir is None:
            root_dir = ROOT_DIR_SLIDE
        self.root_dir = Path(root_dir)
        self.group_col = group_col
        self.label_col = label_col
        self.h5_glob = h5_glob
        self.sort_h5 = sort_h5
        self.raise_if_empty = raise_if_empty

        # 读表（支持 xlsx/csv）
        ext = Path(table_path).suffix.lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(table_path)
        elif ext in [".csv", ".txt"]:
            df = pd.read_csv(table_path)
        else:
            raise ValueError(f"不支持的表格格式: {table_path}")

        # 基础清洗 & 列检查
        for col in [group_col, label_col]:
            if col not in df.columns:
                raise ValueError(f"缺少列: {col}")
            df[col] = df[col].astype(str).str.strip()

        # 若表里同一个前缀重复出现，只保留一条（如果你有更复杂场景可去掉这行）
        df = df.drop_duplicates(subset=[group_col]).reset_index(drop=True)

        # 扫描每个 prefix 的 h5 清单
        groups: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            prefix = row[group_col]
            label  = row[label_col]

            case_dir = self.root_dir / prefix
            if not case_dir.exists():
                raise FileNotFoundError(f"未找到病例子目录: {case_dir}")

            h5_list = [Path(p) for p in glob.glob(str(case_dir / self.h5_glob))]
            if self.sort_h5:
                h5_list = sorted(h5_list, key=lambda p: p.name)

            if len(h5_list) == 0:
                msg = f"病例 {prefix} 在 {case_dir} 下未匹配到 {self.h5_glob}"
                if self.raise_if_empty:
                    raise FileNotFoundError(msg)
                else:
                    # 跳过该病例
                    continue

            groups.append({
                "group_id": prefix,
                "paths": [str(p) for p in h5_list],
                "label": label
            })

        if len(groups) == 0:
            raise RuntimeError("未收集到任何病例（groups 为空）。请检查表格或根目录。")

        self.groups = groups

        # 为数据划分提供 df/label_col/label2idx
        self.df = pd.DataFrame({
            group_col: [g["group_id"] for g in groups],
            label_col: [g["label"]     for g in groups],
        })
        self.label2idx = {lb: i for i, lb in enumerate(self.LABELS)}

    def __len__(self):
        return len(self.groups)

    def _read_one_h5(self, h5_path: str):
        """features [768]"""
        with h5py.File(h5_path, "r") as f:
            return torch.from_numpy(f["features"][:]).float()

    def _one_hot(self, label: str) -> torch.Tensor:
        if label not in self.label2idx:
            raise ValueError(f"未知标签: {label}（应为 {self.LABELS}）")
        y = torch.zeros(len(self.LABELS), dtype=torch.float32)
        y[self.label2idx[label]] = 1.0
        return y

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item   = self.groups[idx]
        prefix = item["group_id"]
        paths  = item["paths"]
        label  = item["label"]

        out = {
            "case_prefix": prefix,                 # 供外部日志/调试使用
            "label_onehot": self._one_hot(label),  # [C]
        }
        out["features_list"] = [self._read_one_h5(p) for p in paths]  # List[Tensor(768)]
        return out

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {}
        out["case_prefix"] = [b["case_prefix"] for b in batch]                  # List[str]（B项）
        out["features_list"] = [b["features_list"] for b in batch]              # List[List[Tensor]]
        out["label_onehot"] = torch.stack([b["label_onehot"] for b in batch], dim=0)  # (B,C)
        return out
