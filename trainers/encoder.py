# -*- coding: utf-8 -*-
"""
Stage-0：单模态编码器预训练（k 折），产出 Stage-1 用于初始化的 CT / 表格编码器权重。

    python train.py ct --log_dir <dir>       # CT 3D ResNet18 编码器
    python train.py table --log_dir <dir>    # 表格编码器
"""
import os
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.paths import EXCEL, EXCEL_CT, CT_NPY_DIR
from utils.pipeline import (
    CLASS_NAMES, N_CLASSES, BestTracker, build_warmup_cosine, iter_folds,
    make_loaders, print_epoch, save_kfold_summary,
)
from utils.utils import (
    FocalCrossEntropyLoss, _maybe_save_topk, compute_epoch_metrics_macro, save_epoch_metrics, set_seed,
    update_epoch_buffers,
)

DEFAULTS = {
    "ct": dict(xlsx=EXCEL_CT, npy_dir=CT_NPY_DIR),
    "table": dict(xlsx=EXCEL, npy_dir=None),
}


# ------------------------------
# 各模态：dataset / collate / model / optimizer
#   collate 统一返回 (inputs: dict, y)，模型调用为 model(**inputs)
# ------------------------------
def collate_ct(batch):
    x = torch.stack([b[0] for b in batch])  # [B, 1, D, H, W]
    y = torch.stack([b[1] for b in batch])  # one-hot
    return {"x": x}, y


def collate_table(batch):
    x_num = torch.stack([b["x_num"] for b in batch]) if batch[0]["x_num"] is not None else None
    x_cat = torch.stack([b["x_cat"] for b in batch]) if batch[0]["x_cat"] is not None else None
    msk = torch.stack([b["missing_mask"] for b in batch]) if batch[0]["missing_mask"] is not None else None
    y = torch.stack([b["y"] for b in batch])  # one-hot
    return {"x_num": x_num, "x_cat": x_cat, "missing_mask": msk}, y


def build_ct(args):
    from config.configs import CTEncoderConfig
    from data.CTDataset import CTNPYFromXLSX
    from models.encoder.ctEncoder import CtEncoder

    cfg = CTEncoderConfig(
        out_dim=768, n_classes=N_CLASSES, gn_groups=16, feature_extraction=False,
        lr=1e-4, weight_decay=1e-5, batch_size=1, epochs=args.epochs, use_amp=True,
        output_dir=args.log_dir, use_pretrained=False,
    )
    ds = CTNPYFromXLSX(
        npy_dir=args.npy_dir, xlsx_path=args.xlsx, id_col="编号", label_col="术后病理诊断类型",
        label_names=CLASS_NAMES, use_zscore=False, window=None, to_112=False, verbose=True,
    )

    def make_model(device):
        model = CtEncoder(cfg).to(device)
        return model, model.make_optimizer()

    return cfg, ds, collate_ct, make_model


def build_table(args):
    from config.configs import TableEncoderConfig
    from data.TableDataset import TabularDataset
    from models.encoder.tableEncoder import TableEncoder, make_optimizer

    cfg = TableEncoderConfig(
        num_features=4, cat_cardinalities=[3] * 15 + [4, 6, 3],  # 会被数据集真实值覆盖
        emb_dim=32, hidden_dims=(256, 256), out_dim=768, n_classes=N_CLASSES, activation="relu",
        dropout=0.1, use_attention=True, n_heads=4, n_attn_layers=4, feature_extraction=False,
        lr=1e-4, weight_decay=1e-5, batch_size=1, epochs=args.epochs, use_amp=True, output_dir="",
    )
    ds = TabularDataset(
        xlsx_path=args.xlsx,
        num_cols=["年龄", "长径cm", "短径cm", "实性成分长径"],
        binary_cols=["圆形、类圆形", "不规则", "分叶", "结节征", "空泡", "空洞", "钙化",
                     "支气管征", "清楚", "光滑锐利", "毛刺", "尖角、桃尖", "索条",
                     "血管集束征", "胸膜凹陷"],
        pleura_col="与胸膜的关系", gender_col="性别", density_col="密度", label_col="术后病理诊断类型",
    )
    cfg.num_features = len(ds.num_cols)
    cfg.cat_cardinalities = ds.cat_cardinalities

    def make_model(device):
        model = TableEncoder(cfg).to(device)
        return model, make_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return cfg, ds, collate_table, make_model


BUILDERS = {"ct": build_ct, "table": build_table}


# ------------------------------
# Train / Val loops
# ------------------------------
def _to_device(inputs: dict, device: str) -> dict:
    return {k: (v.to(device, non_blocking=True) if v is not None else None) for k, v in inputs.items()}


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str,
              optimizer: optim.Optimizer = None, scaler: GradScaler = None,
              log_dir: str = None, epoch: int = 1, split: str = "train") -> Dict[str, float]:
    """optimizer 为 None 时为验证（no_grad）"""
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    buf = {"y_true": [], "y_pred": [], "proba": []}

    with torch.set_grad_enabled(train):
        for inputs, y in tqdm(loader, desc=f"[{'Train' if train else 'Val  '}] Epoch {epoch}"):
            inputs = _to_device(inputs, device)
            y = y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(**inputs)
                loss = criterion(logits, y)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.item())
            update_epoch_buffers(buf, logits, y)

    metrics = compute_epoch_metrics_macro(buf, num_classes=N_CLASSES)
    metrics["loss"] = total_loss
    if log_dir is not None:
        save_epoch_metrics(log_dir, split, epoch, metrics)
    return metrics


def add_args(p, modality: str):
    d = DEFAULTS[modality]
    p.set_defaults(modality=modality)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--xlsx", default=d["xlsx"])
    if modality == "ct":
        p.add_argument("--npy_dir", default=d["npy_dir"], help="CT 体数据 .npy 目录")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--folds", type=int, nargs="*", default=None, help="只跑指定折（1-based），默认全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)


def run(args):
    set_seed(args.seed if getattr(args, "train_seed", None) is None else args.train_seed)   # --train_seed 只影响训练随机性
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg, ds, collate_fn, make_model = BUILDERS[args.modality](args)
    fold_best = []

    for fold_id, train_idx, val_idx, fold_log, ckpt_dir in iter_folds(
            ds, args.n_splits, args.seed, args.log_dir, args.folds,
            min_per_class_each_fold=1, shuffle=True, verbose=True):
        train_loader, val_loader = make_loaders(ds, train_idx, val_idx, collate_fn, args.num_workers, indexed=False)

        model, optimizer = make_model(device)
        # 线性 warmup 4 个 epoch 后 cosine 衰减到 0
        scheduler = build_warmup_cosine(optimizer, cfg.epochs, warmup_epochs=4, min_ratio=0.0)
        scaler = GradScaler(enabled=cfg.use_amp and (device == "cuda"))
        criterion = FocalCrossEntropyLoss(gamma=2.0, weight=None, label_smoothing=0.05, reduction="mean")

        print(f"[Init] device={device} | train={len(train_idx)} | val={len(val_idx)} | cfg={asdict(cfg)}")

        best = BestTracker()
        topk_ckpts = []
        for epoch in range(1, cfg.epochs + 1):
            tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, fold_log, epoch, "train")
            va = run_epoch(model, val_loader, criterion, device, log_dir=fold_log, epoch=epoch, split="val")
            scheduler.step()
            print_epoch(epoch, tr, va)
            best.update(va)

            payload = {"epoch": epoch, "val_acc": float(va["acc"]), "model": model.state_dict(), "cfg": asdict(cfg)}
            _maybe_save_topk(ckpt_dir, args.topk, topk_ckpts, epoch, float(va["acc"]), payload)

        print(f"[Fold {fold_id}] Best Val Acc = {best.acc:.4f}")
        fold_best.append(best.stats)

    save_kfold_summary(args.log_dir, fold_best)
