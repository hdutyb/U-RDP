# -*- coding: utf-8 -*-
"""
Stage-0：术后病理 WSI 单模态训练（一个病例 = 按“术后病理前缀”聚合的多张 slide）。
预计算 TITAN slide 特征 [768]/张 -> 聚合器（PAM）-> LinearHead；只保存 model.aggregator.state_dict()，
该聚合器在 Stage-2 中把真实术后 WSI 聚合为扩散目标 x0_real。

    python train.py wsi_post --agg new_hybrid --log_dir <dir>
"""
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.TitanPostSlideDataset import PostOpWSIPrefixDataset
from models.common.agg import AGGREGATORS
from models.common.common import TitanWithAggregator
from utils.pipeline import (
    N_CLASSES, BestTracker, iter_folds, print_epoch, save_kfold_summary,
)
from utils.utils import (
    FocalCrossEntropyLoss, _maybe_save_topk, compute_epoch_metrics_macro, save_epoch_metrics, set_seed,
    update_epoch_buffers,
)


def add_args(p):
    p.add_argument("--agg", choices=list(AGGREGATORS), default="new_hybrid")
    p.add_argument("--kfold", type=int, default=5, help="折数")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--head_dropout", type=float, default=0.2, help="LinearHead dropout")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--topk", type=int, default=5, help="保留 val acc 最高的 K 个 checkpoint")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--folds", type=int, nargs="*", default=None, help="只跑指定折（1-based），默认全部")
    p.add_argument("--seed", type=int, default=42)


def build_model(args) -> nn.Module:
    model = TitanWithAggregator(agg=args.agg, out_dim=N_CLASSES)
    model.head.dropout.p = args.head_dropout
    return model


def forward_case(model: nn.Module, batch, device: str):
    """batch_size=1：返回 (logits [1,C], z [1,768])"""
    return model([t.to(device) for t in batch["features_list"][0]])


# =========================
# 训练 / 验证（batch=1）
# =========================
def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device: str,
              log_dir: str, epoch: int, split: str) -> Dict[str, float]:
    """optimizer 为 None 时做验证（no_grad + eval）"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    buf = {"y_true": [], "y_pred": [], "proba": []}

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader):
            y = batch["label_onehot"].to(device)  # [1,C]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits, _ = forward_case(model, batch, device)
                loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            update_epoch_buffers(buf, logits, y)

    metrics = compute_epoch_metrics_macro(buf, num_classes=N_CLASSES)
    metrics["loss"] = total_loss
    if log_dir:
        save_epoch_metrics(log_dir, split, epoch, metrics)
    return metrics


def train_one_split(args, train_set, val_set, log_dir: str, device: str):
    """训练一个 train/test 划分，返回最佳 epoch 的 test 指标"""
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    loader_kw = dict(batch_size=1, num_workers=args.num_workers, pin_memory=True,
                     collate_fn=PostOpWSIPrefixDataset.collate_fn)
    train_loader = DataLoader(train_set, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    model = build_model(args).to(device)
    criterion = FocalCrossEntropyLoss(gamma=2.0, weight=None, label_smoothing=0.05, reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    topk_ckpts = []
    best = BestTracker()
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, optimizer, device, log_dir, epoch, split="train")
        val_stats = run_epoch(model, val_loader, criterion, None, device, log_dir, epoch, split="val")
        scheduler.step()

        print_epoch(epoch, train_stats, val_stats, extra=f"lr={optimizer.param_groups[0]['lr']:.2e}")
        best.update(val_stats)

        # 只存聚合器权重（下游 --agg_ckpt 格式）
        ckpt_payload = {"epoch": epoch, "val_acc": float(val_stats["acc"]), "model": model.aggregator.state_dict()}
        _maybe_save_topk(ckpt_dir, args.topk, topk_ckpts, epoch, float(val_stats["acc"]), ckpt_payload)

    return best.stats


def run(args):
    set_seed(args.seed if getattr(args, "train_seed", None) is None else args.train_seed)   # --train_seed 只影响训练随机性
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PostOpWSIPrefixDataset()

    fold_best_results = []
    for fold_id, train_idx, val_idx, fold_dir, _ in iter_folds(dataset, args.kfold, args.seed, args.log_dir, args.folds):
        best = train_one_split(args, Subset(dataset, train_idx), Subset(dataset, val_idx), fold_dir, device)
        print(f"[Fold {fold_id}] Best Test Acc = {best['acc']:.4f}")
        fold_best_results.append(best)
    save_kfold_summary(args.log_dir, fold_best_results)
