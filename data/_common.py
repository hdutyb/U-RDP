# -*- coding: utf-8 -*-
"""
各数据集共用的预处理：
    - encode_table      : 表格列编码（数值 + 离散），所有含表格的数据集编码规则一致
    - load_ct_volume    : 读取 CT .npy 并归一化为 [1, T, H, W]
    - labels_to_onehot  : 标签列 -> one-hot（非法标签为全 0）
    - read_h5_features  : 读取 h5 中的 'features'
"""
import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DENSITY_MAP = {"磨玻璃": 0, "部分实性": 1, "实性": 2}
MALE_TOKENS = ["男", "male", "m", "1"]
FEMALE_TOKENS = ["女", "female", "f", "0"]


def _require_cols(df: pd.DataFrame, cols, kind: str):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"找不到{kind}列：{c}")


def _map_density(v) -> int:
    """磨玻璃/部分实性/实性 -> 0/1/2；也接受数字 0/1/2；其它 -> -1（缺失）"""
    if pd.isna(v):
        return -1
    s = str(v).strip()
    if s in DENSITY_MAP:
        return DENSITY_MAP[s]
    try:
        iv = int(float(s))
        return iv if iv in (0, 1, 2) else -1
    except (ValueError, TypeError):
        return -1


def _map_gender(v) -> int:
    """女=0，男=1（也接受 female/f/0、male/m/1）；其它 -> -1（缺失）"""
    if pd.isna(v):
        return -1
    s = str(v).strip().lower()
    if s in MALE_TOKENS:
        return 1
    if s in FEMALE_TOKENS:
        return 0
    try:
        iv = int(float(s))
        return iv if iv in (0, 1) else -1
    except (ValueError, TypeError):
        return -1


def encode_table(
    df: pd.DataFrame,
    num_cols: Sequence[str] = (),
    binary_cols: Sequence[str] = (),
    density_col: Optional[str] = None,
    pleura_col: Optional[str] = None,
    gender_col: Optional[str] = None,
):
    """
    把 Excel 中已补缺失（缺失记为 -1）的表格列编码为张量（整表一次性编码）。

    数值列：缺失 (== -1) 处置 0.0，同时 missing_mask=True
    离散列（顺序固定：二分类 -> 密度 -> 胸膜 -> 性别），缺失统一映射为“缺失类”：
        二分类 0/1，-1 -> 2                    card=3
        密度   磨玻璃/部分实性/实性 0/1/2，-1 -> 3  card=4
        胸膜   0..4（越界截断），-1 -> 5          card=6
        性别   女 0 / 男 1，-1 -> 2              card=3

    Returns:
        x_num        FloatTensor [N, n_num]
        missing_mask BoolTensor  [N, n_num]
        x_cat        LongTensor  [N, n_cat]，没有离散列时为 None
        cat_cols     离散列名（与 x_cat 列顺序一致）
        cat_cards    各离散列类别数（供 TableEncoderConfig.cat_cardinalities）
    """
    n = len(df)

    if num_cols:
        _require_cols(df, num_cols, "连续")
        x_num_df = df[list(num_cols)].copy()
        missing_df = (x_num_df == -1)
        x_num_df = x_num_df.mask(missing_df, other=0.0)
        x_num = torch.tensor(x_num_df.values, dtype=torch.float32)
        missing_mask = torch.tensor(missing_df.values, dtype=torch.bool)
    else:
        x_num = torch.empty((n, 0), dtype=torch.float32)
        missing_mask = torch.empty((n, 0), dtype=torch.bool)

    cat_arrays: List[torch.Tensor] = []
    cat_cols: List[str] = []
    cat_cards: List[int] = []

    def _add(col: str, values: pd.Series, card: int):
        cat_cols.append(col)
        cat_arrays.append(torch.tensor(values.values, dtype=torch.long))
        cat_cards.append(card)

    _require_cols(df, binary_cols, "二分类")
    for c in binary_cols:
        s = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int).replace({-1: 2})
        _add(c, s, 3)

    if density_col is not None:
        _require_cols(df, [density_col], "")
        _add(density_col, df[density_col].apply(_map_density).astype(int).replace({-1: 3}), 4)

    if pleura_col is not None:
        _require_cols(df, [pleura_col], "")
        s = pd.to_numeric(df[pleura_col], errors="coerce").fillna(-1).astype(int)
        _add(pleura_col, s.clip(lower=-1, upper=4).replace({-1: 5}), 6)

    if gender_col is not None:
        _require_cols(df, [gender_col], "")
        _add(gender_col, df[gender_col].apply(_map_gender).astype(int).replace({-1: 2}), 3)

    x_cat = torch.stack(cat_arrays, dim=1) if cat_arrays else None
    return x_num, missing_mask, x_cat, cat_cols, cat_cards


def load_ct_volume(
    path: str,
    use_zscore: bool = True,
    window: Optional[Tuple[float, float]] = None,
    to_112: bool = False,
    minmax_if_no_zscore: bool = True,
    global_mean: Optional[float] = None,
    global_std: Optional[float] = None,
    minmax_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    读取 CT .npy (H, W, D) -> [1, T=D, H, W]。

    处理顺序：window clip -> 归一化 -> 深度轴移到最前 -> （可选）H,W 插值到 112。
    归一化：
        use_zscore=True  : z-score；有 global_mean/std 时用全局统计量，否则逐 volume 计算
        use_zscore=False : minmax_if_no_zscore=True 时 min-max 到 [0,1]（或 minmax_range），否则不归一化
    """
    vol = np.load(path).astype("float32")

    if window is not None:
        lo, hi = window
        vol = np.clip(vol, lo, hi)

    if use_zscore:
        if global_mean is not None and global_std is not None:
            m, s = float(global_mean), float(global_std)
        else:
            m, s = float(vol.mean()), float(vol.std())
        vol = (vol - m) / (s + 1e-6)
    elif minmax_if_no_zscore:
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
            if minmax_range is not None:
                a, b = minmax_range
                vol = vol * (b - a) + a

    vol = np.moveaxis(vol, -1, 0)    # (T, H, W)，深度 D 作为 T
    x = torch.from_numpy(vol)[None]  # [1, T, H, W]

    if to_112:
        T = x.shape[1]
        x = F.interpolate(x, size=(T, 112, 112), mode="trilinear", align_corners=False)

    return x.contiguous()


def labels_to_onehot(labels: pd.Series, label2idx: dict, num_classes: int) -> torch.Tensor:
    """标签字符串 -> one-hot [N, C]；不在 label2idx 中的标签为全 0 行。"""
    idx = torch.tensor(labels.map(label2idx).fillna(-1).astype(int).values, dtype=torch.long)
    valid = idx >= 0
    y = F.one_hot(idx.clamp(min=0), num_classes=num_classes).to(torch.float32)
    y[~valid] = 0.0
    return y


def read_h5_features(h5_path: str) -> torch.Tensor:
    """读取 h5['features'] 为 float32 张量（形状保持原样）。"""
    with h5py.File(h5_path, "r") as f:
        return torch.from_numpy(f["features"][:]).float()


def list_case_h5(case_dir: Path, pattern: str = "*.h5", sort: bool = True,
                 raise_if_empty: bool = True, case_name: str = "") -> List[str]:
    """
    列出术后病理病例目录 case_dir 下的 h5 文件（按文件名排序）。
    目录不存在或为空时：raise_if_empty=True 报错，否则返回 []。
    """
    if not case_dir.exists():
        if raise_if_empty:
            raise FileNotFoundError(f"未找到术后病理目录: {case_dir}")
        return []
    h5_list = glob.glob(str(case_dir / pattern))
    if sort:
        h5_list = sorted(h5_list)
    if not h5_list and raise_if_empty:
        raise FileNotFoundError(f"{case_name or case_dir} 下没有找到任何 h5")
    return h5_list
