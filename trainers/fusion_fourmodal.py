# -*- coding: utf-8 -*-
"""
四模态融合（CT + 冰冻 WSI + 真实术后 WSI + 表格），k 折，端到端训练 CMAA（models.FourModalFusionNetwork）。
使用真实术后病理；其 checkpoint 作为 Stage-2 的蒸馏教师（--teacher_ckpt）与 Stage-3 的 CMAA 初始化（--fusion_ckpt）。

    python train.py fourmodal --ct_ckpt <ct_run> --tab_ckpt <table_run> --agg_ckpt <agg_run> --log_dir <dir>
"""
import torch
import torch.optim as optim

from config.configs import CTEncoderConfig, TriFusionConfig
from config.paths import EXCEL, CT_PRETRAINED, TAB_PRETRAINED, AGG_PRETRAINED
from models.FourModalFusionNetwork import CMAA
from utils.pipeline import (
    BestTracker, build_fourmodal_dataset, build_warmup_cosine, iter_folds,
    add_fusion_args, make_collate_fn, make_fusion_cfg, make_loaders, make_table_cfg, print_epoch, random_modality_drop,
    resolve_fold_ckpt,
    run_supervised_epoch,
    save_kfold_summary,
)
from utils.utils import FocalCrossEntropyLoss, _maybe_save_topk, set_seed

criterion = FocalCrossEntropyLoss(gamma=2.0, weight=None, label_smoothing=0.05, reduction="mean")
MODALITY_DROPOUT = 0.0   # 由 --modality_dropout 设置


def forward_fn(model, batch, device):
    """CMAA 端到端前向：术后 WSI 为 List[List[Tensor]]，由模型内部 TitanAgg 聚合。"""
    CT = batch["CT"].to(device)
    intra = batch["intra_wsi"].to(device)
    post_list = batch["post_wsi_list"]  # list[list[tensor]]
    xnum = batch["x_num"].to(device)
    xcat = batch["x_cat"].to(device) if batch["x_cat"] is not None else None
    mask = None
    if model.training and MODALITY_DROPOUT > 0:   # 训练时随机去掉部分模态（顺序 [CT, intra, post, Table]）
        mask = random_modality_drop(torch.ones(CT.size(0), 4, dtype=torch.bool, device=CT.device), MODALITY_DROPOUT)
    return model(CT, intra, post_list, xnum, xcat, modality_mask=mask)["logits"]


def build_param_groups(model, base_lr: float = 1e-4, weight_decay: float = 1e-5, enc_lr_scale: float = 1.0,
                       agg_lr_scale: float = 2.0):
    """
    分组学习率：编码器 1e-4，术后 WSI 聚合器 / 融合部分 2e-4，投影 / 单模态头 1.5e-4（相对 base_lr=1e-4）。
    注意 CMAA 的投影为 proj_ct / proj_tab / proj_post / proj_intra，只有前两者归入 proj 组（与原脚本一致）。
    """
    groups = {k: [] for k in ("ct", "tab", "wsi_agg", "wsi_head", "proj", "cmt",
                              "modal_fusion", "fuse_head", "mono_heads", "other")}
    rules = [
        ("ct", ("CTEncoder.",)),
        ("tab", ("TableEncoder.",)),
        ("wsi_agg", ("TitanAgg.aggregator.",)),
        ("wsi_head", ("TitanAgg.head.",)),
        ("proj", ("proj_ct", "proj_tab", "proj_wsi")),
        ("cmt", ("cmt.",)),
        ("modal_fusion", ("modal_fusion.",)),
        ("fuse_head", ("head_fuse.",)),
        ("mono_heads", ("head_wsi.", "head_ct.", "head_tab.")),
    ]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = next((g for g, prefixes in rules if name.startswith(prefixes)), "other")
        groups[group].append(param)

    scale = {"ct": enc_lr_scale, "tab": enc_lr_scale, "wsi_agg": agg_lr_scale, "wsi_head": 2.0, "proj": 1.5, "cmt": 2.0,
             "modal_fusion": 2.0, "fuse_head": 2.0, "mono_heads": 1.5, "other": 1.0}
    return [{"params": groups[g], "lr": base_lr * scale[g], "weight_decay": weight_decay} for g in groups if groups[g]]


def add_args(p):
    p.add_argument("--excel", default=EXCEL)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--ct_ckpt", default=CT_PRETRAINED,
                   help="stage0 CT 编码器：k 折运行根目录（按折取最优）/ 含 {fold_id} 的模板 / 具体文件")
    p.add_argument("--tab_ckpt", default=TAB_PRETRAINED, help="stage0 表格编码器（写法同上）")
    p.add_argument("--agg_ckpt", default=AGG_PRETRAINED, help="stage0 术后 WSI 聚合器（写法同上）")
    p.add_argument("--wsi_agg", default="new_hybrid")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5, help="基础学习率，各参数组按比例缩放")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--modality_dropout", type=float, default=0.3,
                   help="训练时每个样本以该概率随机去掉 1~3 个模态（正则，测试仍全模态）")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    add_fusion_args(p)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--folds", type=int, nargs="*", default=None, help="只跑指定折（1-based），默认全部")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)


def run(args):
    global criterion, MODALITY_DROPOUT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed if getattr(args, "train_seed", None) is None else args.train_seed)   # --train_seed 只影响训练随机性
    criterion = FocalCrossEntropyLoss(gamma=2.0, weight=None, label_smoothing=args.label_smoothing, reduction="mean")
    MODALITY_DROPOUT = args.modality_dropout

    full_ds = build_fourmodal_dataset(args.excel)
    collate_fn = make_collate_fn(ct_crop_h=None)
    fold_best_results = []

    for fold_id, train_idx, val_idx, fold_dir, ckpt_dir in iter_folds(
            full_ds, args.n_folds, args.seed, args.log_dir, args.folds,
            min_per_class_each_fold=1, shuffle=True, verbose=True):
        train_loader, val_loader = make_loaders(full_ds, train_idx, val_idx, collate_fn, args.num_workers,
                                                indexed=False)

        model = CMAA(
            ct_config=CTEncoderConfig(feature_extraction=True),
            table_config=make_table_cfg(),
            fusion_cfg=make_fusion_cfg(args),
            wsi_agg=args.wsi_agg,
            tab_pretrained_path=resolve_fold_ckpt(args.tab_ckpt, fold_id),
            ct_pretrained_path=resolve_fold_ckpt(args.ct_ckpt, fold_id),
            agg_pretrained_path=resolve_fold_ckpt(args.agg_ckpt, fold_id),
        ).to(device)

        optimizer = optim.AdamW(build_param_groups(model, args.lr, args.weight_decay), betas=(0.9, 0.999))
        scheduler = build_warmup_cosine(optimizer, args.epochs, args.warmup_epochs)
        best = BestTracker()
        topk_ckpts = []

        for epoch in range(1, args.epochs + 1):
            print(f"\n========== Epoch {epoch}/{args.epochs} ==========")
            train_stats = run_supervised_epoch(model, train_loader, forward_fn, criterion, device,
                                               epoch, fold_dir, optimizer=optimizer)
            val_stats = run_supervised_epoch(model, val_loader, forward_fn, criterion, device, epoch, fold_dir)
            scheduler.step()
            print_epoch(epoch, train_stats, val_stats)
            best.update(val_stats)

            ckpt_payload = {"epoch": epoch, "val_acc": float(val_stats["acc"]), "model": model.state_dict(),
                            "cmaa_layers": args.cmaa_layers, "cmaa_heads": args.cmaa_heads}
            _maybe_save_topk(ckpt_dir, args.topk, topk_ckpts, epoch, float(val_stats["acc"]), ckpt_payload)

        print(f"[Fold {fold_id}] Best Val Acc = {best.acc:.4f}")
        fold_best_results.append(best.stats)

    save_kfold_summary(args.log_dir, fold_best_results)
