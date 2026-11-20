# -*- coding: utf-8 -*-
"""
主流程（Stage-1 / 2 / 3）及多模态融合训练共用的工具：
    - 数据列定义、数据集构造、TableEncoderConfig
    - collate（含 CT H 维中心裁剪）
    - 加载 Stage-1 保存的固定缺失 mask
    - Stage-A 分类模型与各类 checkpoint 加载（含按折解析 checkpoint 路径）
    - k 折循环 / DataLoader / 最优指标跟踪 / 汇总、学习率、日志打印
"""
import glob
import json
import math
import os
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.configs import TableEncoderConfig
from config.paths import (  # noqa: F401  （部分常量供训练脚本从此处导入）
    EXCEL, INTRA_WSI_DIR, POST_WSI_ROOT, CT_PRETRAINED, TAB_PRETRAINED, AGG_PRETRAINED, X0_STATS_ROOT,
)
from models.common.common import LinearHead
from models.featureExtraction import StageAFeatureExtractor
from utils.missing_modality import IndexedSubset, enumerate_non_empty_masks
from utils.utils import (
    _get_autocast_dtype,
    _load_pretrained_agg,
    compute_epoch_metrics_macro,
    ensure_finite,
    format_confusion_matrix,
    sanitize_logits_for_metrics,
    save_epoch_metrics,
    update_epoch_buffers,
)

# =======================================================================
# 类别 / 表格列
# =======================================================================
CLASS_NAMES = ["AAH/AIS", "MIA", "AC"]   # 标签顺序：0=AAH/AIS, 1=MIA, 2=AC
N_CLASSES = 3
SUMMARY_METRICS = ("acc", "precision", "recall", "f1", "auroc")

NUM_COLS = ["年龄", "长径cm", "短径cm", "实性成分长径"]
BINARY_COLS = [
    "圆形、类圆形", "不规则", "分叶", "结节征", "空泡", "空洞", "钙化", "支气管征",
    "清楚", "光滑锐利", "毛刺", "尖角、桃尖", "索条", "血管集束征", "胸膜凹陷",
]
PLEURA_COL = "与胸膜的关系"
GENDER_COL = "性别"
DENSITY_COL = "密度"

TABLE_COLUMNS = dict(
    num_cols=NUM_COLS,
    binary_cols=BINARY_COLS,
    pleura_col=PLEURA_COL,
    gender_col=GENDER_COL,
    density_col=DENSITY_COL,
)


def make_table_cfg() -> TableEncoderConfig:
    return TableEncoderConfig(
        num_features=len(NUM_COLS),
        cat_cardinalities=[3] * len(BINARY_COLS) + [4, 6, 3],
        emb_dim=32,
        hidden_dims=(256, 256),
        activation="relu",
        dropout=0.1,
        use_attention=True,
        n_heads=4,
        n_attn_layers=4,
        feature_extraction=True,
    )


# =======================================================================
# 数据集构造（Stage-1 与 Stage-2/3 的样本顺序必须一致：固定缺失 mask 按样本下标保存/复用）
# =======================================================================
def build_trimodal_dataset(excel: str = EXCEL, wsi_dir: str = INTRA_WSI_DIR):
    """Stage-1：CT + 冰冻 WSI 特征 + 表格。返回 dict(CT, wsi_feat, x_num, x_cat, y)。"""
    from data.FrozenTripleModalFusionDataset import TriModalFromXLSX
    return TriModalFromXLSX(
        xlsx_path=excel,
        ct_col="__npy_path__",
        wsi_id_col="冰冻切片号",
        wsi_dir=wsi_dir,
        label_col="术后病理诊断类型",
        label_names=tuple(CLASS_NAMES),
        **TABLE_COLUMNS,
        use_zscore=True,
        window=None,
        to_112=False,
        filter_missing=True,
        verbose=True,
    )


def build_fourmodal_dataset(excel: str = EXCEL):
    """Stage-2/3 与四模态融合：CT + 冰冻 WSI + 术后 WSI 列表 + 表格。返回 dict(CT, intra_wsi, post_wsi_list, x_num, x_cat, y)。"""
    from data.FourModalFusionDataset import FourModalFromXLSX
    return FourModalFromXLSX(
        xlsx_path=excel,
        ct_col="__npy_path__",
        intra_wsi_id_col="冰冻切片号",
        intra_wsi_dir=INTRA_WSI_DIR,
        post_wsi_prefix_col="术后病理前缀",
        post_wsi_root=POST_WSI_ROOT,
        label_col="术后病理诊断类型",
        label_names=CLASS_NAMES,
        **TABLE_COLUMNS,
        filter_missing=True,
        verbose=True,
    )


# =======================================================================
# CT shape 修正：少数 H=65 的 CT 中心裁剪为 H=33
# =======================================================================
def fix_ct_shape(x, target_h=33):
    """
    x: [C, D, H, W]
    只处理 H 维：H=65 -> 中心裁剪到 33。
    """
    h = x.shape[-2]

    if h == target_h:
        return x

    if h > target_h:
        start = (h - target_h) // 2
        end = start + target_h
        return x[:, :, start:end, :]

    raise ValueError(f"CT H={h} smaller than target_h={target_h}, cannot crop.")


# =======================================================================
# Collate
# =======================================================================
LIST_KEYS = ("post_wsi_list", "wsi_list")   # 每个样本是 List[Tensor]，不能 stack


def make_collate_fn(ct_crop_h: Optional[int] = 33):
    """
    通用 collate：
        - post_wsi_list / wsi_list 保持 List[List[Tensor]]
        - x_cat 可为 None
        - CT 在 stack 前按 ct_crop_h 做 H 维中心裁剪（ct_crop_h=None/0 表示不裁剪）
        - 其余张量 stack，无法 stack 的保持 list
    """
    def collate(batch: List[dict]):
        out = {}
        for k in batch[0].keys():
            vals = [b[k] for b in batch]

            if k in LIST_KEYS:
                out[k] = vals
            elif k == "x_cat":
                out[k] = None if vals[0] is None else torch.stack(vals, dim=0)
            elif k == "CT":
                if ct_crop_h:
                    vals = [fix_ct_shape(v, target_h=ct_crop_h) for v in vals]
                out[k] = torch.stack(vals, dim=0)
            else:
                try:
                    out[k] = torch.stack(vals, dim=0)
                except Exception:
                    out[k] = vals
        return out

    return collate


# =======================================================================
# 固定缺失 mask 管理器：直接加载 Stage-1 保存的 fixed_missing_masks.pt
# =======================================================================
class LoadedFixedMaskManager:
    """
    从 Stage-1 保存的 fixed_missing_masks.pt 中读取 masks。

    要求：
        ckpt["masks"]: [N,3] bool / 0-1 tensor

    使用：
        modality_mask = mask_manager.get(sample_indices, device=device)
    """

    def __init__(self, mask_file: str, expected_len: Optional[int] = None):
        if not os.path.isfile(mask_file):
            raise FileNotFoundError(
                f"Stage-1 fixed missing mask file not found: {mask_file}\n"
                f"请确认 Stage-1 是否保存了 fixed_missing_masks.pt。"
            )

        ckpt = torch.load(mask_file, map_location="cpu")

        if isinstance(ckpt, dict) and "masks" in ckpt:
            masks = ckpt["masks"]
            self.meta = {k: v for k, v in ckpt.items() if k != "masks"}
        else:
            masks = ckpt
            self.meta = {}

        if not torch.is_tensor(masks):
            raise TypeError(f"masks must be a torch.Tensor, got {type(masks)}")

        if masks.dim() != 2 or masks.size(1) != 3:
            raise ValueError(f"masks must be [N,3], got {tuple(masks.shape)}")

        self.masks = masks.cpu().bool()

        if expected_len is not None and self.masks.size(0) != int(expected_len):
            raise ValueError(
                f"Mask length mismatch: masks has {self.masks.size(0)} samples, "
                f"but dataset has {expected_len} samples. "
                f"这通常说明 Stage-1 和当前阶段的 full_ds 过滤规则或样本顺序不一致。"
            )

    def get(self, indices, device=None):
        if torch.is_tensor(indices):
            idx = indices.detach().cpu().long().view(-1)
        else:
            idx = torch.as_tensor(indices, dtype=torch.long).view(-1)

        if idx.numel() == 0:
            raise ValueError("indices is empty.")

        if int(idx.min()) < 0 or int(idx.max()) >= self.masks.size(0):
            raise IndexError(
                f"indices out of range: min={int(idx.min())}, max={int(idx.max())}, "
                f"num_masks={self.masks.size(0)}"
            )

        out = self.masks.index_select(0, idx)
        if device is not None:
            out = out.to(device)
        return out.bool()

    def summary(self):
        patterns = enumerate_non_empty_masks(include_full=True, dtype=torch.bool)
        names = ["CT", "WSI", "Table", "CT+WSI", "CT+Table", "WSI+Table", "CT+WSI+Table"]
        return {
            name: int((self.masks == pat.view(1, 3)).all(dim=1).sum().item())
            for name, pat in zip(names, patterns)
        }


# =======================================================================
# Stage-A + 分类头（Stage-1 训练模型）
# =======================================================================
class StageAClsModel(nn.Module):
    """
    第一阶段训练模型：
        StageAFeatureExtractor
            -> condition_flat [B,m]
            -> classifier
            -> logits [B,C]
    """

    def __init__(self, cfg, ct_cfg, wsi_cfg, table_cfg, tri_cfg):
        super().__init__()

        self.stage_a = StageAFeatureExtractor(
            cfg=cfg,
            ct_cfg=ct_cfg,
            wsi_cfg=wsi_cfg,
            table_cfg=table_cfg,
            tri_cfg=tri_cfg,
        )

        self.classifier = LinearHead(
            in_dim=cfg.m,
            hidden_dim=cfg.hidden,
            out_dim=cfg.num_classes,
        )

    def forward(self, CT, WSI_intra_feat, Table_num, Table_cat=None, modality_mask=None):
        out = self.stage_a(
            CT=CT,
            WSI_intra_feat=WSI_intra_feat,
            Table_num=Table_num,
            Table_cat=Table_cat,
            modality_mask=modality_mask,
        )

        out["logits"] = self.classifier(out["condition_flat"])
        return out


# =======================================================================
# Checkpoint 加载
# =======================================================================
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "net", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_best_effort(module: nn.Module, ckpt_path: Optional[str], prefix_candidates=()):
    """
    尽量从旧 checkpoint 中给某个子模块加载权重：
    依次尝试原始 key、去掉 prefix 的 key，取与 module 匹配 key 最多的一组。
    """
    if ckpt_path is None:
        return

    if not os.path.isfile(ckpt_path):
        print(f"[Pretrain] Skip, file not found: {ckpt_path}")
        return

    print(f"[Pretrain] Loading from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    if not isinstance(state, dict):
        print(f"[Pretrain] Skip, unsupported checkpoint format: {type(state)}")
        return

    target_keys = set(module.state_dict().keys())
    candidates = [state]

    for prefix in prefix_candidates:
        stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if stripped:
            candidates.append(stripped)

    for prefix in prefix_candidates:
        name = prefix[:-1] if prefix.endswith(".") else prefix
        marker = name + "."
        stripped = {k.split(marker, 1)[1]: v for k, v in state.items() if marker in k}
        if stripped:
            candidates.append(stripped)

    best_state = None
    best_match = -1

    for cand in candidates:
        match = len(set(cand.keys()) & target_keys)
        if match > best_match:
            best_match = match
            best_state = cand

    if best_state is None or best_match <= 0:
        print("[Pretrain] No matched keys, skip.")
        return

    missing, unexpected = module.load_state_dict(best_state, strict=False)
    print(
        f"[Pretrain] matched_keys={best_match}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def _load_submodule_from_ckpt(submodule: nn.Module, ckpt_path: str, prefix: str, tag: str,
                              skip_prefixes: Sequence[str] = (), device="cuda"):
    """
    从完整模型 checkpoint 中按 prefix（如 "stage_a."）加载某个子模块；
    也兼容 checkpoint 直接保存该子模块 state_dict 的情况。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{tag} checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)}")

    target_state = submodule.state_dict()
    load_state = {}
    stats = {"matched": 0, "skipped_prefix": 0, "skipped_shape": 0, "skipped_other": 0}

    for k, v in state_dict.items():
        if any(k.startswith(p) or k.startswith("module." + p) for p in skip_prefixes):
            stats["skipped_prefix"] += 1
            continue

        if k.startswith(prefix):
            new_k = k[len(prefix):]
        elif k.startswith("module." + prefix):
            new_k = k[len("module." + prefix):]
        else:
            new_k = k

        if new_k in target_state:
            if target_state[new_k].shape == v.shape:
                load_state[new_k] = v
                stats["matched"] += 1
            else:
                stats["skipped_shape"] += 1
        else:
            stats["skipped_other"] += 1

    missing, unexpected = submodule.load_state_dict(load_state, strict=False)
    print(f"[{tag} Load] " + ", ".join(f"{k}={v}" for k, v in stats.items())
          + f", missing={len(missing)}, unexpected={len(unexpected)}")

    if stats["matched"] == 0:
        raise RuntimeError(f"No {tag} weights were loaded. 请检查 checkpoint 中是否包含 {prefix}*。")


def resolve_fold_ckpt(path: Optional[str], fold_id: int) -> Optional[str]:
    """
    按折解析 checkpoint 路径：
        - 含 {fold_id} 的模板        -> 代入折号
        - 目录（某次 k 折运行的根目录）-> <dir>/fold_k/checkpoints 下 val acc 最高的权重
                                       （优先读 best_top3.json，否则按文件名中的 acc 选）
        - 具体文件                   -> 原样返回（所有折共用同一个文件）
    """
    if path is None:
        return None
    path = path.format(fold_id=fold_id)
    if not os.path.isdir(path):
        return path

    ckpt_dir = os.path.join(path, f"fold_{fold_id}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = os.path.join(path, "checkpoints") if os.path.isdir(os.path.join(path, "checkpoints")) else path

    index = os.path.join(ckpt_dir, "best_top3.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            for item in json.load(f):          # 已按 acc 降序
                cand = os.path.join(ckpt_dir, item["path"])
                if os.path.isfile(cand):
                    return cand

    files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoint found for fold {fold_id} under {path}")

    def acc_of(p):
        m = re.search(r"acc([0-9.]+)\.pth$", os.path.basename(p))
        return float(m.group(1)) if m else -1.0

    return max(files, key=acc_of)


def load_stage_a_from_stage1_ckpt(model, ckpt_path: str, device="cuda"):
    """从 Stage-1 checkpoint 中加载 model.stage_a，自动跳过 classifier.*。"""
    _load_submodule_from_ckpt(model.stage_a, ckpt_path, prefix="stage_a.", tag="Stage-A",
                              skip_prefixes=("classifier.",), device=device)


def load_diff_from_stage2_ckpt(model, ckpt_path: str, device="cuda"):
    """
    从 Stage-2 checkpoint 中加载 model.diff（不加载 gen_classifier）。
    checkpoint 为残差生成（payload 有 residual，含 anchor.*）时同时开启并严格加载 Anchor，
    此时 diff 的 mean/std buffer 是残差 r 的统计量。
    """
    _load_submodule_from_ckpt(model.diff, ckpt_path, prefix="diff.", tag="Stage-2 Diff", device=device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mode = ckpt.get("residual") if isinstance(ckpt, dict) else None
    if mode:
        model.enable_anchor(mode, ckpt.get("anchor_dropout", 0.2), ckpt.get("anchor_hidden", 768))
        sd = {k[len("anchor."):]: v for k, v in ckpt["model"].items() if k.startswith("anchor.")}
        model.anchor.load_state_dict(sd, strict=True)
        print(f"[Stage-2 Anchor Load] residual={mode}, {len(sd)} tensors (strict)")


def load_titanagg_from_ckpt(model, ckpt_path: Optional[str], device="cuda"):
    """加载 model.TitanAgg 的聚合器权重（见 utils.utils._load_pretrained_agg）。"""
    if ckpt_path is None:
        return
    _load_pretrained_agg(model.TitanAgg, ckpt_path)


# =======================================================================
# 学习率 / 汇总
# =======================================================================
def build_param_groups(module_lrs: Sequence[Tuple[str, nn.Module, float]], weight_decay: float) -> List[dict]:
    """
    分模块学习率：module_lrs = [(name, module, lr), ...]。
    lr > 0 的模块解冻并加入优化器；lr <= 0 的模块冻结（requires_grad=False）。
    """
    groups = []
    for name, module, lr in module_lrs:
        params = list(module.parameters())
        trainable = lr > 0
        for p in params:
            p.requires_grad_(trainable)
        n = sum(p.numel() for p in params)
        print(f"[Optim] {name:<14s} lr={lr:.1e}  params={n / 1e6:.2f}M  {'train' if trainable else 'frozen'}")
        if trainable and params:
            groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": name})
    return groups


def trainable_params(optimizer) -> List[torch.nn.Parameter]:
    return [p for g in optimizer.param_groups for p in g["params"]]


def build_warmup_cosine(optimizer, num_epochs: int, warmup_epochs: int = 5, min_ratio: float = 0.1):
    """线性 warmup 后 cosine 衰减到 min_ratio * base_lr（按 epoch step）。"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def best_stats(val_stats: dict) -> dict:
    return {m: float(val_stats[m]) for m in SUMMARY_METRICS}


class BestTracker:
    """记录评估折 acc 最高的 epoch 指标（4:1 划分下评估折即 test 折）。"""

    def __init__(self):
        self.acc = 0.0
        self.stats = None

    def update(self, val_stats: dict) -> None:
        if self.stats is None or val_stats["acc"] > self.acc:
            self.acc = val_stats["acc"]
            self.stats = best_stats(val_stats)
        print(f"Best test_acc = {self.acc:.4f}")


# =======================================================================
# 数据划分：train/test = 4:1 轮换（5 折）。无验证集：每个 epoch 在 test 折上评估，取 acc 最高的 epoch
# =======================================================================
def _sample_keys(dataset) -> List[str]:
    """样本主键：病例表的 ID 列；术后 WSI 数据集没有 ID，用 术后病理前缀（与 ID 一一对应）。"""
    df = dataset.df
    if "ID" in df.columns:
        return df["ID"].astype(str).tolist()
    return df["术后病理前缀"].astype(str).tolist()


def _canonical_chunks(n_chunks: int, seed: int) -> Dict[str, int]:
    """
    k 折划分的统一分块：按 EXCEL 的 术后病理诊断类型 分层，每类打乱后均分成 n_chunks 块。
    返回 {ID: chunk}，同时提供 {术后病理前缀: chunk}，使 CT / 表格 / 冰冻 / 术后 WSI 各数据集的划分完全一致
    （不依赖各数据集内部的样本顺序与 label2idx）。
    """
    import pandas as pd
    df = pd.read_excel(EXCEL)
    rng = np.random.default_rng(seed)
    chunk = np.empty(len(df), dtype=int)
    for _, idx in sorted(df.groupby(df["术后病理诊断类型"].astype(str).str.strip()).indices.items()):
        idx = np.array(idx)[rng.permutation(len(idx))]
        for c, part in enumerate(np.array_split(idx, n_chunks)):
            chunk[part] = c
    out = {str(k): int(c) for k, c in zip(df["ID"], chunk)}
    out.update({str(k): int(c) for k, c in zip(df["术后病理前缀"], chunk)})
    return out


def split_41(dataset, fold_id: int, n_folds: int = 5, seed: int = 42) -> Tuple[list, list]:
    """第 fold_id 折（1-based）：test = 块 fold_id-1，其余 n_folds-1 块为 train。返回数据集下标 (train, test)。"""
    chunks = _canonical_chunks(n_folds, seed)
    keys = _sample_keys(dataset)
    missing = [k for k in keys if k not in chunks]
    if missing:
        raise KeyError(f"{len(missing)} 个样本不在 EXCEL 中，无法划分：{missing[:5]}")
    train, test = [], []
    for i, k in enumerate(keys):
        (test if chunks[k] == fold_id - 1 else train).append(i)
    return train, test


def iter_folds(dataset, n_folds: int, seed: int, log_dir: str, only: Optional[Sequence[int]] = None,
               **_) -> Iterator[Tuple[int, list, list, str, str]]:
    """
    4:1 k 折迭代：yield (fold_id, train_idx, test_idx, fold_dir, ckpt_dir)，并创建 fold_dir/checkpoints。
    only: 只跑指定折（1-based）。
    """
    os.makedirs(log_dir, exist_ok=True)
    for fold_id in range(1, n_folds + 1):
        if only and fold_id not in only:
            continue
        train_idx, test_idx = split_41(dataset, fold_id, n_folds, seed)
        print(f"\n===================== FOLD {fold_id} / {n_folds} =====================")
        fold_dir = os.path.join(log_dir, f"fold_{fold_id}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[Split 41] train={len(train_idx)} test={len(test_idx)}")
        with open(os.path.join(fold_dir, "split.json"), "w", encoding="utf-8") as f:
            keys = _sample_keys(dataset)
            json.dump({name: [keys[i] for i in idx] for name, idx in (("train", train_idx), ("test", test_idx))},
                      f, ensure_ascii=False)
        yield fold_id, train_idx, test_idx, fold_dir, ckpt_dir


def make_loaders(dataset, train_idx, val_idx, collate_fn, num_workers: int, indexed: bool = True,
                 batch_size: int = 1):
    """
    train / val DataLoader（默认 batch_size=1，与历史实验一致）。
    indexed=True 时用 IndexedSubset，在 batch 中附带原始样本下标 "__sample_index__"（固定缺失 mask 需要）。
    batch_size > 1 时最后一个 batch 可能不满（drop_last=False）。
    """
    from torch.utils.data import Subset
    wrap = IndexedSubset if indexed else Subset
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return (DataLoader(wrap(dataset, train_idx), shuffle=True, **kw),
            DataLoader(wrap(dataset, val_idx), shuffle=False, **kw))


def stage_a_arch(stage1_ckpt: Optional[str]) -> dict:
    """Stage-1 checkpoint 中记录的 Stage-A CMAA 结构，供 Stage-2/3 构建同结构的 Stage-A。"""
    out = {}
    if stage1_ckpt and os.path.isfile(stage1_ckpt):
        ck = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
        if isinstance(ck, dict):
            for k in ("stage_a_cmaa_layers", "stage_a_cmaa_heads", "stage_a_cmaa_dropout"):
                if k in ck:
                    out[k] = ck[k]
    return out


def add_fusion_args(p):
    """四模态融合模块（CMAA）结构。"""
    p.add_argument("--cmaa_layers", type=int, default=2, help="CMAA 层数 L")
    p.add_argument("--cmaa_heads", type=int, default=8, help="CMAA 头数 H")
    p.add_argument("--cmaa_dropout", type=float, default=0.1, help="CMAA 注意力 dropout 概率")


def make_fusion_cfg(args):
    from config.configs import TriFusionConfig
    return TriFusionConfig(cmaa_layers=getattr(args, "cmaa_layers", 2), cmaa_heads=getattr(args, "cmaa_heads", 8),
                           cmaa_dropout=getattr(args, "cmaa_dropout", 0.1))


def random_ct_flip(CT: torch.Tensor) -> torch.Tensor:
    """CT 训练时数据增强：各以 0.5 概率左右（W）/ 上下（H）翻转，[B,1,D,H,W]；测试不做。"""
    if torch.rand(1).item() < 0.5:
        CT = CT.flip(-1)
    if torch.rand(1).item() < 0.5:
        CT = CT.flip(-2)
    return CT


def random_modality_drop(mask: torch.Tensor, p: float) -> torch.Tensor:
    """
    训练时的模态 dropout（n 个模态通用）：每个样本以概率 p 换成一个随机的"非空、非全"模态子集（与原 mask 取交），
    至少保留 1 个模态；验证 / 测试不使用。防止融合只依赖某一个模态，起正则作用。mask [B,n] bool。
    """
    if p <= 0:
        return mask
    B, n = mask.shape
    mask = mask.bool()
    drop = torch.rand(B, device=mask.device) < p
    if not drop.any():
        return mask
    bits = (torch.arange(1, 2 ** n - 1, device=mask.device).unsqueeze(1) >> torch.arange(n, device=mask.device)) & 1
    pick = bits.bool()[torch.randint(len(bits), (B,), device=mask.device)]          # [B,n] 非空非全
    out = torch.where(drop.unsqueeze(1), pick & mask, mask)
    empty = out.sum(1) == 0                                                          # 与原 mask 取交后可能为空
    return torch.where(empty.unsqueeze(1), mask, out)


def best_ckpt_in(ckpt_dir: str) -> str:
    """本折 checkpoints 目录中 acc 最高的权重（读 best_top3.json，已按 acc 降序）。"""
    with open(os.path.join(ckpt_dir, "best_top3.json"), encoding="utf-8") as f:
        for item in json.load(f):
            cand = os.path.join(ckpt_dir, item["path"])
            if os.path.isfile(cand):
                return cand
    raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")


def _mean_std(results: List[dict]) -> Tuple[dict, dict]:
    results = [f for f in results if f is not None]
    if not results:
        return {}, {}
    return ({m: float(np.mean([f[m] for f in results])) for m in SUMMARY_METRICS},
            {m: float(np.std([f[m] for f in results])) for m in SUMMARY_METRICS})


def save_kfold_summary(log_dir: str, fold_best_results: List[dict], **extra) -> dict:
    """写 {log_dir}/kfold_summary.json：各折 test 最优 epoch 的指标及其 mean / std。"""
    mean, std = _mean_std(fold_best_results)
    summary = {"split": "41", "selection": "best_epoch_on_test (no validation set)",
               "test_results": fold_best_results, "test_mean": mean, "test_std": std}
    summary.update(extra)

    with open(os.path.join(log_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== KFold Final Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def autocast_ctx(device: str):
    """bf16/fp16 autocast（CPU 上关闭）。"""
    ac_dtype = _get_autocast_dtype(device)
    if ac_dtype is not None and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=ac_dtype)
    return torch.cuda.amp.autocast(enabled=False)


def run_supervised_epoch(model, loader, forward_fn, criterion, device, epoch, log_dir,
                         optimizer=None, num_classes: int = N_CLASSES, split: Optional[str] = None):
    """
    端到端多模态融合模型的一个 epoch（optimizer=None 为验证）。
        forward_fn(model, batch, device) -> logits
    非有限 logits / loss 的 batch 会被跳过；loss 为 epoch 内求和。
    """
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    buf = {"y_true": [], "y_pred": [], "proba": []}
    ac_ctx = autocast_ctx(device)
    suffix = "" if train else "(val)"

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader):
            y_1h = batch["y"].to(device)
            y_idx = y_1h.argmax(dim=-1)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with ac_ctx:
                logits = forward_fn(model, batch, device)

            if not ensure_finite("logits" + suffix, logits):
                continue

            loss = criterion(logits, y_idx)
            if not ensure_finite("loss" + suffix, loss):
                continue

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.item())
            update_epoch_buffers(buf, sanitize_logits_for_metrics(logits), y_1h)

    metrics = compute_epoch_metrics_macro(buf, num_classes=num_classes)
    metrics["loss"] = total_loss

    if log_dir:
        save_epoch_metrics(log_dir, split or ("train" if train else "val"), epoch, metrics)

    return metrics


_PRINT_KEYS = ("loss", "diff_loss", "recon_loss", "cls_loss",
               "acc", "precision", "recall", "f1", "auroc")


def print_epoch(epoch: int, train_stats: dict, val_stats: dict, extra: str = "") -> None:
    """打印一个 epoch 的 train / val 指标（只打印存在的 key）与混淆矩阵。"""
    def fmt(stats):
        return " ".join(f"{k}={stats[k]:.4f}" for k in _PRINT_KEYS if k in stats)

    print(f"Epoch {epoch:02d} | train: {fmt(train_stats)} | test: {fmt(val_stats)}" + (f" | {extra}" if extra else ""))
    print("Train CM:\n", format_confusion_matrix(train_stats["cm"], labels=CLASS_NAMES))
    print("Test  CM:\n", format_confusion_matrix(val_stats["cm"], labels=CLASS_NAMES))
