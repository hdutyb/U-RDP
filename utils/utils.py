# -*- coding: utf-8 -*-
"""
通用训练工具：
    - 随机种子
    - epoch 指标缓冲 / 宏平均指标 / 写 metrics_{split}.txt / 混淆矩阵打印
    - top-k checkpoint 保存
    - Focal CE 损失
    - NaN 防护、autocast dtype
    - 预训练权重加载（CT / 表格编码器、术后 WSI 聚合器）与冻结工具
"""
import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


# =======================================================================
# 随机种子
# =======================================================================
def set_seed(seed: int = 42):
    """固定 python / numpy / torch 随机种子（不改 cudnn 设置）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =======================================================================
# 指标
# =======================================================================
def _to_index_labels(y: torch.Tensor) -> torch.Tensor:
    """one-hot [B,C] -> [B]；已是索引则原样返回。"""
    return y.argmax(dim=-1) if y.ndim == 2 else y


def _softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


def update_epoch_buffers(buf: dict, logits: torch.Tensor, y: torch.Tensor):
    """
    把一个 batch 的真值 / 预测 / 概率追加到 buf。
    buf = {"y_true": [], "y_pred": [], "proba": []}；y 可为 one-hot 或类别索引。
    """
    y_idx = _to_index_labels(y).detach().cpu().numpy()         # [B]
    proba = _softmax_logits(logits).detach().cpu().numpy()     # [B,C]
    y_pred = proba.argmax(axis=-1)                              # [B]
    buf["y_true"].append(y_idx)
    buf["y_pred"].append(y_pred)
    buf["proba"].append(proba)


def compute_epoch_metrics_macro(buf: dict, num_classes: int = 3) -> dict:
    """
    由 update_epoch_buffers 累积的一个 epoch 数据计算指标。

    返回：
        acc          准确率
        precision    宏平均 Precision
        recall       宏平均 Recall
        f1           宏平均 F1
        auroc        宏平均 AUROC（OvR；验证集缺类等无法计算时为 nan）
        bal_acc      Balanced Accuracy（各类 Recall 的平均）
        auprc_macro  宏平均 AUPRC（OvR；只统计在真值中出现且非全部的类别）
        cm           混淆矩阵 [C, C]（行=True，列=Pred）
    buf 为空时各指标为 0、cm 为全 0。
    """
    # 空缓冲保护
    if not buf or len(buf.get("y_true", [])) == 0:
        return {
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
            "bal_acc": 0.0,
            "auprc_macro": 0.0,
            "cm": np.zeros((num_classes, num_classes), dtype=int),
        }

    # 拼接所有 batch
    y_true = np.concatenate(buf["y_true"], axis=0)  # [N], 取值 0..C-1
    y_pred = np.concatenate(buf["y_pred"], axis=0)  # [N]
    proba = np.concatenate(buf["proba"], axis=0)  # [N, C]

    # 基础准确率
    acc = float((y_pred == y_true).mean())

    # 宏平均 P/R/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0
    )

    # 宏平均 AUROC（OvR）
    try:
        auroc = roc_auc_score(
            y_true, proba,
            multi_class="ovr",
            average="macro",
            labels=np.arange(num_classes)
        )
        auroc = float(auroc)
    except Exception:
        auroc = float("nan")

    # Balanced Accuracy（各类 Recall 的平均）
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # 宏平均 AUPRC（OvR：对每一类当正类）
    ap_list = []
    for c in range(num_classes):
        pos = (y_true == c).astype(int)
        # 需要该类在真值中至少出现一次且有负类样本，否则 AP 不定义
        if (pos.sum() > 0) and (pos.sum() < len(pos)):
            ap = average_precision_score(pos, proba[:, c])
            ap_list.append(ap)
    auprc_macro = float(np.mean(ap_list)) if len(ap_list) > 0 else float("nan")

    # 混淆矩阵（行=True，列=Pred）
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "bal_acc": float(balanced_acc),
        "auprc_macro": float(auprc_macro),
        "cm": cm,
    }


def save_epoch_metrics(log_dir: str, split: str, epoch: int, metrics: dict):
    """
    以固定列宽向 {log_dir}/metrics_{split}.txt 追加一行（文件不存在时先写表头）。
    列：epoch, loss, diff_loss, recon_loss, cda_loss, angle_loss, cls_loss, acc, balanced_acc,
        precision, recall, f1, auroc, auprc_macro, cm_r_c...；metrics 中没有的列写 nan。
    """
    os.makedirs(log_dir, exist_ok=True)
    if split == "val":   # 4:1 划分无验证集，每个 epoch 的评估即 test 折
        split = "test"
    path = os.path.join(log_dir, f"metrics_{split}.txt")

    # ----------- 读取指标值（缺省为 nan）-----------
    loss = float(metrics.get("loss", float("nan")))
    diff_loss = float(metrics.get("diff_loss", float("nan")))
    recon_loss = float(metrics.get("recon_loss", float("nan")))
    cls_loss = float(metrics.get("cls_loss", float("nan")))
    cda_loss = float(metrics.get("cda_loss", float("nan")))
    angle_loss = float(metrics.get("angle_loss", float("nan")))
    acc = float(metrics.get("acc", float("nan")))
    # compute_epoch_metrics_macro 返回的 key 为 bal_acc
    balanced_acc = float(metrics.get("balanced_acc", metrics.get("bal_acc", float("nan"))))
    precision = float(metrics.get("precision", float("nan")))
    recall = float(metrics.get("recall", float("nan")))
    f1 = float(metrics.get("f1", float("nan")))
    auroc = metrics.get("auroc", float("nan"))
    auprc_macro = float(metrics.get("auprc_macro", float("nan")))

    cm = np.asarray(metrics["cm"], dtype=int).reshape(-1)
    num_classes = int(len(cm) ** 0.5)

    # ----------- 列定义（顺序必须与 row_items 一致）-----------
    col_defs = [
        ("epoch", 6, "d"),
        ("loss", 12, ".6f"),
        ("diff_loss", 12, ".6f"),
        ("recon_loss", 12, ".6f"),
        ("cda_loss", 12, ".6f"),
        ("angle_loss", 12, ".6f"),
        ("cls_loss", 12, ".6f"),
        ("acc", 10, ".6f"),
        ("balanced_acc", 14, ".6f"),
        ("precision", 12, ".6f"),
        ("recall", 10, ".6f"),
        ("f1", 10, ".6f"),
        ("auroc", 10, ".6f"),
        ("auprc_macro", 14, ".6f"),
    ]

    cm_width = 6
    cm_cols = [(f"cm_{r}_{c}", cm_width, "d")
               for r in range(num_classes)
               for c in range(num_classes)]

    all_cols = col_defs + cm_cols

    # -------- 写表头 --------
    if not os.path.exists(path):
        header = []
        for name, width, _ in all_cols:
            header.append(f"{name:>{width}}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(" ".join(header) + "\n")

    # -------- 写数据行 --------
    def fmt(val, width, spec):
        if isinstance(val, str):
            s = val
        else:
            try:
                s = f"{int(val)}" if spec == "d" else format(float(val), spec)
            except Exception:
                s = str(val)
        return f"{s:>{width}}"

    row_vals = [
        epoch,
        loss,
        diff_loss,
        recon_loss,
        cda_loss,
        angle_loss,
        cls_loss,
        acc,
        balanced_acc,
        precision,
        recall,
        f1,
        auroc,
        auprc_macro,
    ]

    row_items = [fmt(val, width, spec) for (_, width, spec), val in zip(col_defs, row_vals)]
    row_items += [fmt(cm[i], width, spec) for i, (_, width, spec) in enumerate(cm_cols)]

    with open(path, "a", encoding="utf-8") as f:
        f.write(" ".join(row_items) + "\n")


def format_confusion_matrix(cm, labels=("AAH/AIS", "MIA", "AC"), col_width=6):
    """混淆矩阵 -> 可直接 print 的字符串（行=True，列=Pred）。"""
    cm = np.asarray(cm)
    header = " " * (col_width + 2) + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    rows = []
    for i, lbl in enumerate(labels):
        row_vals = "".join(f"{int(cm[i, j]):>{col_width}}" for j in range(len(labels)))
        rows.append(f"{lbl:>{col_width}} |{row_vals}")
    return header + "\n" + "\n".join(rows)


# =======================================================================
# Checkpoint
# =======================================================================
def _maybe_save_topk(ckpt_dir: str, k: int, topk_list: list, cur_epoch: int, cur_acc: float, payload: dict):
    """
    只保留 val acc 最高的 k 个 checkpoint：{ckpt_dir}/ckpt_epoch{EEE}_acc{A.AAAA}.pth。
    topk_list 由调用方持有（每折一个空列表），元素 {"epoch", "acc", "path"}，按 acc 降序。
    acc 严格高于当前第 k 名才会保存；列表同步写到 {ckpt_dir}/best_top3.json（文件名固定，与 k 无关）。
    """
    # 生成当次待保存文件名
    safe_acc = f"{cur_acc:.4f}"
    save_path = os.path.join(ckpt_dir, f"ckpt_epoch{cur_epoch:03d}_acc{safe_acc}.pth")

    # 如果当前 epoch 已存在记录且更好/相同则先移除旧的（极少见，但保持幂等）
    for i, it in enumerate(topk_list):
        if it["epoch"] == cur_epoch:
            try:
                if os.path.exists(it["path"]) and it["path"] != save_path:
                    os.remove(it["path"])
            except Exception:
                pass
            topk_list.pop(i)
            break

    # 如果还没达到 k 个，或者当前 acc 比最差的更高，就纳入
    need_add = (len(topk_list) < k) or (cur_acc > min(topk_list, key=lambda x: x["acc"])["acc"])
    rank = lambda x: (-x["acc"], x["epoch"])

    if need_add:
        # 先保存当前 checkpoint
        torch.save(payload, save_path)

        # 放入列表并排序（按 acc 降序），best_top3.json 第一项即下游加载的权重
        topk_list.append({"epoch": cur_epoch, "acc": float(cur_acc), "path": save_path})
        topk_list.sort(key=rank)

        # 如果超过 k 个，删除最差的权重文件
        while len(topk_list) > k:
            worst = topk_list.pop(-1)
            try:
                if os.path.exists(worst["path"]):
                    os.remove(worst["path"])
            except Exception:
                pass

        # 同步写一个 JSON 方便后续查看/复现实验
        meta_path = os.path.join(ckpt_dir, "best_top3.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"epoch": it["epoch"], "acc": round(it["acc"], 6), "path": os.path.basename(it["path"])} for it in topk_list],
                f, ensure_ascii=False, indent=2
            )


# =======================================================================
# 损失
# =======================================================================
class FocalCrossEntropyLoss(nn.Module):
    """
    Focal loss：(1 - p_t)^gamma * CE（CE 含可选 label smoothing / 类别权重）。
    targets 可为类别索引 [B] 或 one-hot [B,C]。
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =======================================================================
# NaN 防护 / autocast
# =======================================================================
def _finite_min_max(x: torch.Tensor):
    """有限值的 (min, max)；没有有限值时 (None, None)。"""
    if not torch.is_tensor(x):
        return None, None
    mask = torch.isfinite(x)
    if not mask.any():
        return None, None
    x = x[mask]
    return float(x.min().item()), float(x.max().item())


def ensure_finite(name: str, t: torch.Tensor) -> bool:
    """t 含 NaN/Inf 时打印统计并返回 False（调用方据此跳过该 batch）；非张量视为合法。"""
    if not torch.is_tensor(t):
        return True
    bad = ~torch.isfinite(t)
    if bad.any():
        n = int(bad.sum().item())
        tmin, tmax = _finite_min_max(t)
        print(f"[NaNGuard] {name}: {n} non-finite (min={tmin}, max={tmax})")
        return False
    return True


def sanitize_logits_for_metrics(logits: torch.Tensor, clamp_val=30.0):
    """仅用于统计指标：非有限值置 0 并截断到 [-clamp_val, clamp_val]（不参与反传）。"""
    x = logits.float().detach()
    mask = torch.isfinite(x)
    x = torch.where(mask, x, torch.zeros_like(x))
    return x.clamp(-clamp_val, clamp_val)


def _get_autocast_dtype(device: str):
    """CUDA 上优先 bf16，不支持时 fp16；CPU 返回 None（不开 autocast）。"""
    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


# =======================================================================
# 预训练权重加载
# =======================================================================
def _load_encoder_state(model: nn.Module, ckpt_path: str, prefix: str, tag: str, name: str):
    """
    把 train_encoder.py 保存的编码器 checkpoint 加载到 model（strict=False）。
    state_dict 取 ckpt["state_dict"] / ckpt["model"] / ckpt 本身；若有 key 带 prefix，只保留这些 key 并去掉前缀。
    文件不存在时只打印警告。
    """
    if not os.path.isfile(ckpt_path):
        print(f"WARNING: 找不到{tag}预训练权重: {ckpt_path}")
        return

    print(f"加载{tag}预训练权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    sd = ckpt
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]

    if any(k.startswith(prefix) for k in sd.keys()):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"{name} missing keys: {len(missing)}")
    if unexpected:
        print(f"{name} unexpected keys: {len(unexpected)}")


def _load_pretrained_ct_encoder(model, ckpt_path: str):
    _load_encoder_state(model, ckpt_path, prefix="CTEncoder.", tag=" CT ", name="CTEncoder")


def _load_pretrained_table_encoder(model, ckpt_path: str):
    _load_encoder_state(model, ckpt_path, prefix="TableEncoder.", tag="表格", name="TableEncoder")


def _load_pretrained_agg(model: nn.Module, path: str) -> int:
    """
    加载术后 WSI 聚合器权重（train_wsi_post.py --feature slide 保存的 checkpoint）。

    model: TitanWithAggregator（加载到其 .aggregator）或聚合器本身。
    兼容 checkpoint 中三种 key 形式：
        attention.fc.weight                    —— 直接保存 aggregator.state_dict()（实际使用的格式）
        aggregator.attention.fc.weight         —— 保存 TitanWithAggregator.state_dict()
        TitanAgg.aggregator.attention.fc.weight —— 保存整网
    返回加载成功的参数个数；一个都没对上时报错（以前会静默跳过，导致聚合器一直是随机初始化）。
    """
    if not os.path.isfile(path):
        print(f"[WARN] Aggregator pretrained not found: {path}")
        return 0

    print(f"[INFO] Loading Aggregator pretrained: {path}")
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt

    target = getattr(model, "aggregator", model)
    target_keys = set(target.state_dict().keys())

    agg_sd = {}
    for k, v in sd.items():
        for prefix in ("module.", "TitanAgg.", "aggregator."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if k in target_keys:
            agg_sd[k] = v

    if not agg_sd:
        raise RuntimeError(f"No aggregator weights matched in {path} (keys: {list(sd)[:5]} ...)")

    missing, unexpected = target.load_state_dict(agg_sd, strict=False)
    print(f"[INFO] Aggregator weights loaded: matched={len(agg_sd)}, missing={missing}, unexpected={unexpected}")
    return len(agg_sd)


# =======================================================================
# 冻结 / 解冻
# =======================================================================
def freeze_module(m: nn.Module):
    """requires_grad=False（不改 train/eval 模式，调用方需自行 eval）。"""
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_module(m: nn.Module):
    """requires_grad=True 并切到 train 模式。"""
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()


def unfreeze_classifier(model):
    """解冻 model.gen_classifier（Stage-2 在 cls_start_epoch 调用）。"""
    for p in model.gen_classifier.parameters():
        p.requires_grad_(True)
    model.gen_classifier.train()
