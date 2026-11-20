# -*- coding: utf-8 -*-
"""
表格单模态数据集（train_encoder.py --modality table 用）。
"""
from typing import List, Optional

import pandas as pd
from torch.utils.data import Dataset

from data._common import encode_table, labels_to_onehot


class TabularDataset(Dataset):
    """
    读取已补缺失（缺失记为 -1）的 Excel，不做任何样本筛选（全表）。
    编码规则见 data._common.encode_table；density / pleura / gender 列必须存在。

    __getitem__ 返回 dict：
        x_num        FloatTensor [n_num]（缺失处为 0.0）
        x_cat        LongTensor  [n_cat]
        missing_mask BoolTensor  [n_num]
        y            FloatTensor one-hot [3]（label_col 不为 None 时；非法标签为全 0）

    注意：标签顺序为 AAH/AIS, AC, MIA（与其它数据集的 AAH/AIS, MIA, AC 不同），见 label2idx。
    """
    LABEL2IDX = {"AAH/AIS": 0, "AC": 1, "MIA": 2}

    def __init__(self,
                 xlsx_path: str,
                 num_cols: List[str],
                 binary_cols: List[str],
                 pleura_col: str,
                 gender_col: str,
                 density_col: str,
                 label_col: Optional[str] = None):
        super().__init__()
        df = pd.read_excel(xlsx_path)

        self.x_num, self.missing_mask, self.x_cat, cat_cols, cat_cards = encode_table(
            df, num_cols, binary_cols, density_col=density_col, pleura_col=pleura_col, gender_col=gender_col)

        self.y = None
        label2idx = dict(self.LABEL2IDX)
        if label_col is not None:
            if label_col not in df.columns:
                raise ValueError(f"找不到标签列：{label_col}")
            # 标签统一大写；数字标签 0/1/2 映射回字符串
            df[label_col] = (
                df[label_col].astype(str).str.strip().str.upper()
                .replace({"0": "AAH/AIS", "1": "MIA", "2": "AC"})
            )
            self.y = labels_to_onehot(df[label_col], label2idx, num_classes=3)

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.cat_cardinalities = cat_cards      # 供 TableEncoderConfig.cat_cardinalities

        # 供 split_dataset_stratified(_kfold) 使用
        self.df = df.copy()
        self.label_col = label_col
        self.label2idx = label2idx

    def __len__(self):
        return self.x_num.size(0)

    def __getitem__(self, idx):
        item = {
            "x_num": self.x_num[idx],
            "x_cat": self.x_cat[idx],
            "missing_mask": self.missing_mask[idx],
        }
        if self.y is not None:
            item["y"] = self.y[idx]
        return item
