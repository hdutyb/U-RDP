# -*- coding: utf-8 -*-
"""
Stage-3：最终四模态融合分类（CMAA）。

    Stage-A（Stage-1 权重，冻结）       -> CT 特征 / 冰冻 WSI 特征 / 表格特征 [B,m]，以及 condition_vector
    Anchor + 扩散模型（Stage-2，冻结）  -> 伪术后病理特征 x0_gen = z_anchor + r̂ [B,m]
    CMAA.fusion([CT, 冰冻 WSI, x0_gen, 表格]) -> logits（CMAA 由 --fusion_ckpt 初始化）

只训练 CMAA；推理（验证 / 测试）时不需要真实术后病理，术后特征由扩散模型从术前 / 术中信息采样生成。

    python train.py stage3 --stage2_ckpt <stage2_run> --stage1_ckpt <stage1_run> --fusion_ckpt <fourmodal_run> ...
"""
import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config.configs import DiffusionConfig
from config.paths import EXCEL, AGG_PRETRAINED, X0_STATS_ROOT, STAGE1_MASK_FILE, STAGE2_RUN
from trainers.stage2 import build_model, masked_inputs
from utils.pipeline import (
    N_CLASSES, BestTracker, LoadedFixedMaskManager, autocast_ctx, best_stats, build_fourmodal_dataset,
    add_fusion_args, build_param_groups, stage_a_arch,
    build_warmup_cosine, iter_folds, load_diff_from_stage2_ckpt, load_stage_a_from_stage1_ckpt,
    load_titanagg_from_ckpt, make_collate_fn, make_loaders, print_epoch, random_modality_drop, resolve_fold_ckpt,
    save_kfold_summary, trainable_params,
)
from utils.utils import (
    FocalCrossEntropyLoss, _maybe_save_topk, compute_epoch_metrics_macro, ensure_finite, freeze_module,
    sanitize_logits_for_metrics, save_epoch_metrics, set_seed, update_epoch_buffers,
)

criterion = FocalCrossEntropyLoss()


def stage2_ddim_steps(stage2_ckpt: Optional[str]) -> int:
    """Stage-2 checkpoint 中记录的 train_ddim_steps（采样步数须与 Stage-2 训练一致）。"""
    if not stage2_ckpt or not os.path.isfile(stage2_ckpt):
        return DiffusionConfig.train_ddim_steps
    ckpt = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    return int(ckpt.get("train_ddim_steps", DiffusionConfig.train_ddim_steps))


def set_stage3_train_mode(model: nn.Module):
    """只训练 CMAA，其余模块保持 eval。"""
    model.train()
    model.stage_a.eval()
    model.diff.eval()
    model.TitanAgg.eval()
    model.gen_classifier.eval()
    model.cmaa.train()
    if model.anchor is not None:
        model.anchor.eval()


def load_fusion_from_ckpt(model, ckpt_path: Optional[str], device="cuda"):
    """加载已训练的 CMAA 权重（train.py fourmodal 的结果或完整模型的 cmaa.*）。"""
    if ckpt_path is None:
        return
    if not os.path.isfile(ckpt_path):
        print(f"[Fusion] checkpoint not found, skip: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    target_state = model.cmaa.state_dict()
    load_state, skipped = {}, 0

    for k, v in state_dict.items():
        for prefix in ("cmaa.", "module.cmaa."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        if k in target_state and target_state[k].shape == v.shape:
            load_state[k] = v
        else:
            skipped += 1

    missing, unexpected = model.cmaa.load_state_dict(load_state, strict=False)
    print(f"[Fusion Load] matched={len(load_state)}, skipped={skipped}, missing={len(missing)}, "
          f"unexpected={len(unexpected)}")
    # 融合部分必须全部加载
    core = [k for k in missing
            if k.split(".")[0] in ("cmt", "modal_fusion", "proj_ct", "proj_tab", "proj_post", "proj_intra", "head_fuse")]
    if core:
        raise RuntimeError(f"[Fusion Load] {len(core)} 个融合参数未加载（如 {core[0]}）：CMAA 结构与 fusion_ckpt 不一致？")


@torch.no_grad()
def build_stage3_features(model, CT, WSI_intra_feat, Table_num, Table_cat, modality_mask, sample_steps: int,
                          n_samples: int = 1):
    """
    Stage-A + 扩散采样（不求梯度）。
    返回 CT_feat, WSI_feat, [x0_gen × n_samples], Table_feat（均 [B,m]）与四模态 mask [B,4]
    （顺序 [CT, WSI_intra, x0_gen, Table]）。Stage-A 只算一次，扩散模型以同一 condition 独立采样 n_samples 次。
    """
    stage_a_out = model._forward_first(CT=CT, WSI_intra_feat=WSI_intra_feat, Table_num=Table_num,
                                       Table_cat=Table_cat, modality_mask=modality_mask)
    x0_out = model.sample_post(stage_a_out, sample_steps, n_samples=n_samples)
    x0_gen = [x0_out] if isinstance(x0_out, torch.Tensor) else x0_out

    mask3 = stage_a_out["modality_mask"].bool()   # [B,3] = [CT, WSI_intra, Table]
    mask4 = torch.stack([mask3[:, 0], mask3[:, 1], torch.ones_like(mask3[:, 0]), mask3[:, 2]], dim=1)  # 生成特征总是可用

    return (stage_a_out["CT_feature_flat"], stage_a_out["WSI_feature_flat"], x0_gen,
            stage_a_out["Table_feature_flat"], mask4)


def run_epoch_stage3(model, loader, mask_manager, device, epoch, log_dir, sample_steps, optimizer=None,
                     n_val_samples=1, split=None, modality_dropout=0.0):
    """
    optimizer 为 None 时为验证。Stage-A / 扩散冻结，no_grad 采样 x0_gen 后训练 CMAA。
    验证时 x0_gen 采样 n_val_samples 次，对融合后的 softmax 概率取平均；不使用真实术后 WSI。
    返回的 loss 为按 batch 数平均。
    """
    train = optimizer is not None
    split = split or ("train" if train else "val")
    if train:
        set_stage3_train_mode(model)
        params = trainable_params(optimizer)
    else:
        model.eval()

    total_loss = 0.0
    buf = {"y_true": [], "y_pred": [], "proba": []}
    ac_ctx = autocast_ctx(device)

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=f"{split.capitalize()} Stage3 Ep{epoch}"):
            CT_in, intra_in, xnum_in, xcat_in, modality_mask = masked_inputs(batch, mask_manager, device)
            y_1h = batch["y"].to(device, non_blocking=True)
            y_idx = y_1h.argmax(dim=-1)

            if train:
                optimizer.zero_grad(set_to_none=True)

            n = 1 if train else max(1, n_val_samples)
            CT_feat, WSI_feat, x0_gens, Table_feat, mask4 = build_stage3_features(
                model, CT_in, intra_in, xnum_in, xcat_in, modality_mask, sample_steps, n_samples=n)
            if train and modality_dropout > 0:
                mask4 = random_modality_drop(mask4, modality_dropout)
            with ac_ctx:
                if n == 1:
                    logits = model.cmaa.fusion(CT_feat, WSI_feat, x0_gens[0], Table_feat,
                                               modality_mask=mask4)["logits"]
                else:
                    prob = torch.stack([
                        model.cmaa.fusion(CT_feat, WSI_feat, x0, Table_feat, modality_mask=mask4)["logits"]
                        .float().softmax(dim=-1) for x0 in x0_gens]).mean(0)
                    logits = prob.clamp_min(1e-8).log()
                loss = criterion(logits, y_idx)

            if train:
                if not ensure_finite("logits", logits) or not ensure_finite("loss", loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()

            total_loss += float(loss.item())
            update_epoch_buffers(buf, sanitize_logits_for_metrics(logits), y_1h)

    n = max(1, len(loader))
    metrics = compute_epoch_metrics_macro(buf, num_classes=N_CLASSES)
    metrics["loss"] = total_loss / n
    metrics["cls_loss"] = total_loss / n
    save_epoch_metrics(log_dir, split, epoch, metrics)
    return metrics


def add_args(p):
    p.add_argument("--excel", default=EXCEL)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--stage2_ckpt", default=STAGE2_RUN,
                   help="Stage-2 checkpoint（扩散模型 + Anchor）：k 折运行根目录（按折取最优）/ 含 {fold_id} 的模板 / 具体文件")
    p.add_argument("--stage1_ckpt", default=None,
                   help="Stage-A 来源的 Stage-1 checkpoint（写法同上）；默认取 Stage-2 checkpoint 中的")
    p.add_argument("--stage1_mask_file", default=STAGE1_MASK_FILE)
    p.add_argument("--agg_ckpt", default=AGG_PRETRAINED)
    p.add_argument("--fusion_ckpt", default=None, help="CMAA 初始化权重（train.py fourmodal 的结果，写法同上）")
    p.add_argument("--eval_ckpt", default=None,
                   help="只评估：加载该 Stage-3 checkpoint（写法同上），在 test 折上评估一次，不训练")
    p.add_argument("--x0_stats_root", default=X0_STATS_ROOT,
                   help="扩散 x0 统计（会被 Stage-2 checkpoint 中保存的值覆盖）")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-5, help="CMAA 融合模块的 lr")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_sample_steps", type=int, default=None,
                   help="DDIM 采样步数；默认取 Stage-2 训练时的 train_ddim_steps")
    p.add_argument("--modality_dropout", type=float, default=0.3,
                   help="训练时每个样本以该概率随机去掉部分模态")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    add_fusion_args(p)
    p.add_argument("--n_val_samples", type=int, default=5,
                   help="验证 / 测试时 x0_gen 独立采样次数，对融合后的 softmax 概率取平均（1 = 单次采样）")
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
    criterion = FocalCrossEntropyLoss(label_smoothing=args.label_smoothing)

    full_ds = build_fourmodal_dataset(args.excel)
    mask_manager = LoadedFixedMaskManager(mask_file=args.stage1_mask_file, expected_len=len(full_ds))
    print("[Stage-3] Loaded fixed modality masks from Stage-1:")
    print(json.dumps(mask_manager.summary(), indent=2, ensure_ascii=False))

    collate_fn = make_collate_fn(ct_crop_h=33)
    fold_best_results = []

    for fold_id, train_idx, val_idx, fold_dir, ckpt_dir in iter_folds(full_ds, args.n_folds, args.seed,
                                                                        args.log_dir, args.folds):
        train_loader, val_loader = make_loaders(full_ds, train_idx, val_idx, collate_fn, args.num_workers,
                                                batch_size=args.batch_size)
        stage2_ckpt = resolve_fold_ckpt(args.stage2_ckpt, fold_id)
        stage1_ckpt = resolve_fold_ckpt(args.stage1_ckpt, fold_id)

        # 采样步数与 Stage-2 训练时的可微 DDIM 步数保持一致
        steps = args.val_sample_steps or stage2_ddim_steps(stage2_ckpt)
        print(f"[Stage-3] DDIM sample steps = {steps}")

        args.stage_a_arch = stage_a_arch(stage1_ckpt or stage2_ckpt)
        model = build_model(args, fold_id, device)
        load_stage_a_from_stage1_ckpt(model, stage1_ckpt or stage2_ckpt, device=device)
        load_diff_from_stage2_ckpt(model, stage2_ckpt, device=device)   # 含扩散 mean/std buffer 与 Anchor
        load_titanagg_from_ckpt(model, resolve_fold_ckpt(args.agg_ckpt, fold_id), device=device)
        load_fusion_from_ckpt(model, resolve_fold_ckpt(args.fusion_ckpt, fold_id), device=device)

        if args.eval_ckpt:
            path = resolve_fold_ckpt(args.eval_ckpt, fold_id)
            sd = torch.load(path, map_location=device, weights_only=False)["model"]
            # 训练时的 checkpoint 中扩散网络另有一份别名 key（diff.model_single.* 与 diff.model.* 为同一模块）
            sd = {k: v for k, v in sd.items() if not k.startswith(("diff.model_single.", "diff.model_multi."))}
            model.load_state_dict(sd, strict=True)
            print(f"[Stage-3] eval only: {path}")
            set_seed(args.seed)
            te = run_epoch_stage3(model, val_loader, mask_manager, device, 0, fold_dir, sample_steps=steps,
                                  n_val_samples=args.n_val_samples, split="eval")
            print(f"[Fold {fold_id}] Test Acc = {te['acc']:.4f}")
            fold_best_results.append(best_stats(te))
            continue

        freeze_module(model.TitanAgg)
        freeze_module(model.gen_classifier)
        optimizer = optim.AdamW(build_param_groups([
            ("cmaa", model.cmaa, args.lr),
            ("stage_a", model.stage_a, 0.0),
            ("diff", model.diff, 0.0),
        ], args.weight_decay))
        scheduler = build_warmup_cosine(optimizer, args.epochs, args.warmup_epochs)
        best = BestTracker()
        topk_ckpts = []

        for epoch in range(1, args.epochs + 1):
            print(f"\n========== Epoch {epoch}/{args.epochs} ==========")
            train_stats = run_epoch_stage3(model, train_loader, mask_manager, device, epoch, fold_dir,
                                           sample_steps=steps, optimizer=optimizer,
                                           modality_dropout=args.modality_dropout)
            val_stats = run_epoch_stage3(model, val_loader, mask_manager, device, epoch, fold_dir,
                                         sample_steps=steps, n_val_samples=args.n_val_samples)
            scheduler.step()
            print_epoch(epoch, train_stats, val_stats)
            best.update(val_stats)

            ckpt_payload = {"epoch": epoch, "val_acc": float(val_stats["acc"]), "model": model.state_dict(),
                            "stage1_ckpt": stage1_ckpt, "stage2_ckpt": stage2_ckpt,
                            "stage1_mask_file": args.stage1_mask_file, "train_ddim_steps": steps,
                            "n_val_samples": args.n_val_samples}
            _maybe_save_topk(ckpt_dir, args.topk, topk_ckpts, epoch, float(val_stats["acc"]), ckpt_payload)

        print(f"[Fold {fold_id}] Best Val Acc = {best.acc:.4f}")
        fold_best_results.append(best.stats)

    save_kfold_summary(args.log_dir, fold_best_results,
                       stage1_ckpt_template=args.stage1_ckpt, stage2_ckpt_template=args.stage2_ckpt,
                       stage1_mask_file=args.stage1_mask_file, lr=args.lr, n_val_samples=args.n_val_samples)
