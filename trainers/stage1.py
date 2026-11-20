# -*- coding: utf-8 -*-
"""
Stage-1：CT + 冰冻 WSI 特征 + 表格 -> 三模态条件编码器（CMAA）-> 分类头，k 折。

训练的是三模态条件编码器：其输出 condition_vector 作为 Stage-2 扩散模型的条件，
CT / 冰冻 WSI / 表格特征作为 Stage-3 融合的输入。CT / 表格编码器由 stage0（train.py ct / table）预训练初始化。
每个样本的模态 mask 保存为 {log_dir}/fixed_missing_masks.pt 供 Stage-2/3 复用（全模态）。

    python train.py stage1 --ct_ckpt <ct_run> --tab_ckpt <table_run> --log_dir <dir>
"""
import json
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from config.configs import CTEncoderConfig, ViTWSIConfig, TriFusionConfig, OverallModelConfig
from config.paths import EXCEL, INTRA_WSI_DIR, CT_PRETRAINED, TAB_PRETRAINED
from utils.missing_modality import FixedMissingMaskManager, apply_missing_to_inputs, ensure_wsi_b1m
from utils.pipeline import (
    N_CLASSES, BestTracker, StageAClsModel, autocast_ctx, build_param_groups, build_trimodal_dataset,
    build_warmup_cosine, iter_folds, load_best_effort, make_collate_fn, make_loaders,
    make_table_cfg, print_epoch, random_modality_drop, resolve_fold_ckpt, save_kfold_summary,
)
from utils.utils import (
    FocalCrossEntropyLoss, _maybe_save_topk, compute_epoch_metrics_macro, ensure_finite,
    sanitize_logits_for_metrics, save_epoch_metrics, set_seed, update_epoch_buffers,
)

criterion = FocalCrossEntropyLoss(gamma=2.0, weight=None, label_smoothing=0.05, reduction="mean")


# ------------------------------
# 训练与验证
# ------------------------------
def _unpack(batch, device):
    """batch -> (CT, WSI [B,1,m], x_num, x_cat, y_onehot)"""
    CT = batch["CT"].to(device, non_blocking=True)
    WSI = ensure_wsi_b1m(batch["wsi_feat"].to(device, non_blocking=True))
    xnum = batch["x_num"].to(device, non_blocking=True)
    xcat = batch["x_cat"].to(device, non_blocking=True) if batch["x_cat"] is not None else None
    if batch.get("y") is None:
        raise ValueError("数据集必须提供 y(one-hot)。")
    return CT, WSI, xnum, xcat, batch["y"].to(device, non_blocking=True)


def _forward_masked(model, CT, WSI, xnum, xcat, modality_mask):
    """按 modality_mask 置零缺失模态输入后前向。mask 顺序 [CT, WSI_intra, Table]。"""
    CT_in, WSI_in, xnum_in, xcat_in = apply_missing_to_inputs(
        CT=CT, WSI_intra_feat=WSI, Table_num=xnum, Table_cat=xcat,
        modality_mask=modality_mask, zero_ct_input=False, zero_table_input=False,
    )
    return model(CT=CT_in, WSI_intra_feat=WSI_in, Table_num=xnum_in, Table_cat=xcat_in, modality_mask=modality_mask)


def train_one_epoch(model, loader, optimizer, mask_manager, device, log_dir, epoch, modality_dropout=0.0):
    """modality_dropout > 0 时训练中随机去掉部分模态。"""
    model.train()
    total_loss = 0.0
    buf = {"y_true": [], "y_pred": [], "proba": []}
    ac_ctx = autocast_ctx(device)

    for batch in tqdm(loader):
        CT, WSI, xnum, xcat, y_1h = _unpack(batch, device)
        y_idx = y_1h.argmax(dim=-1)
        modality_mask = random_modality_drop(mask_manager.get(batch["__sample_index__"], device=CT.device),
                                             modality_dropout)  # [B,3]

        optimizer.zero_grad(set_to_none=True)
        with ac_ctx:
            logits = _forward_masked(model, CT, WSI, xnum, xcat, modality_mask)["logits"]

        if not ensure_finite("logits", logits):
            print("[NaNGuard] skip batch due to non-finite logits")
            continue
        loss = criterion(logits, y_idx)
        if not ensure_finite("loss", loss):
            print("[NaNGuard] skip batch due to non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item())
        update_epoch_buffers(buf, sanitize_logits_for_metrics(logits), y_1h)

    metrics = compute_epoch_metrics_macro(buf, num_classes=N_CLASSES)
    metrics["loss"] = total_loss
    save_epoch_metrics(log_dir, "train", epoch, metrics)
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, mask_manager, device, log_dir, epoch, split="val"):
    """验证（split="test" 时为测试）：每个样本使用其固定 mask。"""
    model.eval()
    total_loss = 0.0
    buf = {"y_true": [], "y_pred": [], "proba": []}
    ac_ctx = autocast_ctx(device)

    for batch in tqdm(loader):
        CT, WSI, xnum, xcat, y_1h = _unpack(batch, device)
        y_idx = y_1h.argmax(dim=-1)
        modality_mask = mask_manager.get(batch["__sample_index__"], device=CT.device)

        with ac_ctx:
            logits = _forward_masked(model, CT, WSI, xnum, xcat, modality_mask)["logits"]
        if not ensure_finite("logits(val)", logits):
            print("[NaNGuard] skip batch(val) due to non-finite logits")
            continue
        loss = criterion(logits, y_idx)
        if not ensure_finite("loss(val)", loss):
            print("[NaNGuard] skip batch(val) due to non-finite loss")
            continue

        total_loss += float(loss.item())
        update_epoch_buffers(buf, sanitize_logits_for_metrics(logits), y_1h)

    metrics = compute_epoch_metrics_macro(buf, num_classes=N_CLASSES)
    metrics["loss"] = total_loss
    save_epoch_metrics(log_dir, split, epoch, metrics)
    return metrics


def add_args(p):
    p.add_argument("--excel", default=EXCEL)
    p.add_argument("--wsi_dir", default=INTRA_WSI_DIR)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--ct_crop_h", type=int, default=33, help="CT H 维中心裁剪到该值；0 表示不裁剪")
    p.add_argument("--ct_ckpt", default=CT_PRETRAINED,
                   help="stage0 CT 编码器权重：k 折运行根目录（按折取最优）/ 含 {fold_id} 的模板 / 具体文件")
    p.add_argument("--tab_ckpt", default=TAB_PRETRAINED, help="stage0 表格编码器权重（写法同上）")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5, help="CMAA / 输出 norm / 分类头的 lr")
    p.add_argument("--lr_pretrained", type=float, default=0.0,
                   help="stage0 预训练的 CT / 表格编码器的 lr；0 = 冻结")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--modality_dropout", type=float, default=0.3,
                   help="训练时每个样本以该概率随机去掉 1~2 个模态（正则，验证 / 测试仍全模态）")
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--cmaa_layers", type=int, default=2, help="三模态 CMAA 层数")
    p.add_argument("--cmaa_heads", type=int, default=8, help="三模态 CMAA 头数")
    p.add_argument("--cmaa_dropout", type=float, default=0.1, help="CMAA 注意力 dropout")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--folds", type=int, nargs="*", default=None, help="只跑指定折（1-based），默认全部")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)


def run(args):
    global criterion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed if getattr(args, "train_seed", None) is None else args.train_seed)   # --train_seed 只影响训练随机性
    criterion = FocalCrossEntropyLoss(gamma=2.0, weight=None, label_smoothing=args.label_smoothing,
                                      reduction="mean")

    full_ds = build_trimodal_dataset(args.excel, args.wsi_dir)

    # 每个样本的模态 mask（全模态），按样本下标保存，所有折、所有阶段共用
    mask_manager = FixedMissingMaskManager(num_samples=len(full_ds))
    fixed_missing_meta = {"mode": "full", "modality_order": ["CT", "WSI_intra", "Table"],
                          "summary": mask_manager.summary()}
    os.makedirs(args.log_dir, exist_ok=True)
    mask_file = os.path.join(args.log_dir, "fixed_missing_masks.pt")
    torch.save({"masks": mask_manager.masks, **fixed_missing_meta}, mask_file)

    collate_fn = make_collate_fn(ct_crop_h=args.ct_crop_h)
    fold_best_results = []

    for fold_id, train_idx, val_idx, fold_dir, ckpt_dir in iter_folds(full_ds, args.n_folds, args.seed,
                                                                        args.log_dir, args.folds):
        train_loader, val_loader = make_loaders(full_ds, train_idx, val_idx, collate_fn, args.num_workers,
                                                batch_size=args.batch_size)

        model = StageAClsModel(
            cfg=OverallModelConfig(stage_a_cmaa_layers=args.cmaa_layers, stage_a_cmaa_heads=args.cmaa_heads,
                                   stage_a_cmaa_dropout=args.cmaa_dropout),
            ct_cfg=CTEncoderConfig(feature_extraction=True),
            wsi_cfg=ViTWSIConfig(),
            table_cfg=make_table_cfg(),
            tri_cfg=TriFusionConfig(),
        ).to(device)

        # CT / 表格专家用 stage0 单模态预训练权重初始化
        load_best_effort(model.stage_a.ct_expert, resolve_fold_ckpt(args.ct_ckpt, fold_id),
                         prefix_candidates=("ct_expert.", "CTEncoder.", "ct_encoder.", "encoder.", "stage_a.ct_expert."))
        load_best_effort(model.stage_a.table_expert, resolve_fold_ckpt(args.tab_ckpt, fold_id),
                         prefix_candidates=("table_expert.", "TableEncoder.", "table_encoder.", "encoder.",
                                            "stage_a.table_expert."))

        # 分模块学习率：stage0 预训练的 CT / 表格编码器（默认冻结），新初始化的 CMAA / 分类头用 --lr
        sa = model.stage_a
        optimizer = optim.AdamW(build_param_groups([
            ("ct_expert", sa.ct_expert, args.lr_pretrained),
            ("table_expert", sa.table_expert, args.lr_pretrained),
            ("cmaa_proj", sa.cmaa_proj, args.lr), ("cmaa_attn", sa.cmaa_attn, args.lr),
            ("cmaa_score", sa.cmaa_score, args.lr),
            ("out_norm", sa.out_norm, args.lr),
            ("classifier", model.classifier, args.lr),
        ], args.weight_decay))
        scheduler = build_warmup_cosine(optimizer, args.epochs, args.warmup_epochs)
        best = BestTracker()
        topk_ckpts = []

        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(model, train_loader, optimizer, mask_manager, device, fold_dir, epoch,
                                          modality_dropout=args.modality_dropout)
            val_stats = validate_one_epoch(model, val_loader, mask_manager, device, fold_dir, epoch)
            scheduler.step()

            print_epoch(epoch, train_stats, val_stats)
            best.update(val_stats)

            ckpt_payload = {"epoch": epoch, "val_acc": float(val_stats["acc"]), "model": model.state_dict(),
                            "fixed_missing": fixed_missing_meta,
                            "stage_a_cmaa_layers": args.cmaa_layers, "stage_a_cmaa_heads": args.cmaa_heads,
                            "stage_a_cmaa_dropout": args.cmaa_dropout}
            _maybe_save_topk(ckpt_dir, args.topk, topk_ckpts, epoch, float(val_stats["acc"]), ckpt_payload)

        print(f"[Fold {fold_id}] Best Val Acc = {best.acc:.4f}")
        fold_best_results.append(best.stats)

    save_kfold_summary(args.log_dir, fold_best_results,
                       fixed_missing={**fixed_missing_meta, "mask_file": mask_file})
