# -*- coding: utf-8 -*-
"""
Stage-2：残差扩散（CPD）生成伪术后病理特征。

    CT + 冰冻 WSI + 表格 --Stage-A（Stage-1 权重，冻结）--> condition_vector [B,1,m] / condition_tokens [B,3,m]
    术后 WSI（多张 slide）--TitanAgg（PAM，加载 wsi_post 聚合器，冻结）--> x0_real [B,m]
    冰冻 WSI --Anchor Predictor（train.py anchor，冻结）--> z_anchor；扩散目标为残差 r = x0_real - z_anchor
    损失：w_diff · L_diff + w_rec · 0.5 · L_rec（可微 DDIM）+ L_kd（冻结四模态教师，见 trainers/stage2_kd.py）
    epoch >= cls_start_epoch 起，额外用 x0_gen.detach() 训练生成特征分类头 gen_classifier

- 复用 Stage-1 的 fixed_missing_masks.pt。

    python train.py stage2 --stage1_ckpt <stage1_run> --agg_ckpt <agg_run> --x0_stats_root <x0_stats> \
        --residual intra --anchor_ckpt <anchor_dir>/fold_{fold_id}/sel_intra.pth --teacher_ckpt <fourmodal_run>
"""
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from config.configs import CTEncoderConfig, WSIEncoderConfig, TriFusionConfig, DiffusionConfig, OverallModelConfig
from config.paths import EXCEL, AGG_PRETRAINED, X0_STATS_ROOT, STAGE1_RUN, STAGE1_MASK_FILE
from models.integrated_model import PostOpGenAndCls
from utils.missing_modality import apply_missing_to_inputs, ensure_wsi_b1m
from utils.pipeline import (
    N_CLASSES, BestTracker, LoadedFixedMaskManager, autocast_ctx, build_fourmodal_dataset, iter_folds,
    build_param_groups, load_stage_a_from_stage1_ckpt, make_fusion_cfg, stage_a_arch, load_titanagg_from_ckpt, make_collate_fn, make_loaders,
    make_table_cfg, print_epoch, resolve_fold_ckpt, save_kfold_summary, trainable_params,
)
from utils.utils import (
    FocalCrossEntropyLoss, _maybe_save_topk, compute_epoch_metrics_macro, ensure_finite, freeze_module,
    sanitize_logits_for_metrics, save_epoch_metrics, set_seed, unfreeze_classifier, update_epoch_buffers,
)
from trainers.stage2_kd import FrozenTeacher, add_kd_args, kd_enabled, kd_losses
from trainers.stage2_residual import add_residual_args, setup_residual

criterion = FocalCrossEntropyLoss()


def set_stage2_train_mode(model, epoch: int, cls_start_epoch: int, train_stage_a: bool = True):
    """model.train() 会把所有子模块切到 train，这里把冻结模块切回 eval（避免 BN / Dropout 状态变化）。"""
    model.train()
    model.stage_a.train(train_stage_a)
    model.TitanAgg.eval()
    model.diff.train()
    model.gen_classifier.train(epoch >= cls_start_epoch)
    model.cmaa.eval()   # Stage-2 不使用四模态融合
    if model.anchor is not None:   # 残差生成的 Anchor 拟合后冻结
        model.anchor.eval()


def masked_inputs(batch, mask_manager, device):
    """取出 CT / 冰冻 WSI / 表格，按固定缺失 mask（[CT, WSI_intra, Table]）置零缺失模态。"""
    CT = batch["CT"].to(device, non_blocking=True)
    intra = ensure_wsi_b1m(batch["intra_wsi"].to(device, non_blocking=True))  # [B,1,m]
    xnum = batch["x_num"].to(device, non_blocking=True)
    xcat = batch["x_cat"].to(device, non_blocking=True) if batch["x_cat"] is not None else None

    modality_mask = mask_manager.get(batch["__sample_index__"], device=CT.device)
    CT_in, intra_in, xnum_in, xcat_in = apply_missing_to_inputs(
        CT=CT, WSI_intra_feat=intra, Table_num=xnum, Table_cat=xcat,
        modality_mask=modality_mask, zero_ct_input=False, zero_table_input=False,
    )
    return CT_in, intra_in, xnum_in, xcat_in, modality_mask


def post_wsi_features(batch, device):
    """
    术后 WSI 切片特征，交给 TitanAgg 聚合：
        batch_size=1 -> List[Tensor[768]]（单病例，与历史实现一致）
        batch_size>1 -> List[List[Tensor[768]]]（每个病例切片数不同，TitanAgg 逐病例聚合后拼成 [B,768]）
    """
    cases = [[p.to(device, non_blocking=True) for p in case] for case in batch["post_wsi_list"]]
    return cases[0] if len(cases) == 1 else cases


def cls_weight(epoch: int, cls_start_epoch: int, lambda_cls_max: float, lambda_warmup: int) -> float:
    """生成特征分类损失权重：cls_start_epoch 之前为 0，之后在 lambda_warmup 个 epoch 内线性升到 lambda_cls_max。"""
    if epoch < cls_start_epoch:
        return 0.0
    if lambda_warmup <= 0:
        return float(lambda_cls_max)
    t = (epoch - cls_start_epoch + 1) / float(lambda_warmup)
    return float(lambda_cls_max) * max(0.0, min(1.0, t))


def train_one_epoch_stage2(model, loader, optimizer_diff, optimizer_cls, mask_manager, device, epoch, log_dir,
                           cls_start_epoch=1, lambda_cls_max=1.0, lambda_warmup=10, train_stage_a=False, kd=None):
    """
    每个 batch 两步更新：
        1) diffusion：loss = w_diff * diff_loss + w_rec * 0.5 * recon_loss（见 PostOpGenAndCls.forward_stage2）+ L_kd
        2) epoch >= cls_start_epoch：gen_classifier 在 x0_gen.detach() 上做分类（不回传到扩散模型）
    返回的 loss 为按 batch 数平均。
    """
    set_stage2_train_mode(model, epoch, cls_start_epoch, train_stage_a)
    gen_params = trainable_params(optimizer_diff)
    lambda_cls = cls_weight(epoch, cls_start_epoch, lambda_cls_max, lambda_warmup)
    train_cls = epoch >= cls_start_epoch

    total_loss = total_diff = total_rec = total_cls = 0.0
    buf = {"y_true": [], "y_pred": [], "proba": []}
    ac_ctx = autocast_ctx(device)

    for batch in tqdm(loader):
        CT_in, intra_in, xnum_in, xcat_in, modality_mask = masked_inputs(batch, mask_manager, device)
        features = post_wsi_features(batch, device)
        y_1h = batch["y"].to(device, non_blocking=True)
        y_idx = y_1h.argmax(dim=-1)

        # 1) diffusion 更新
        optimizer_diff.zero_grad(set_to_none=True)
        with ac_ctx:
            out = model(stage=2, WSI_post=features, CT=CT_in, WSI_intra_feat=intra_in, Table_num=xnum_in,
                        Table_cat=xcat_in, label=y_idx, epoch=epoch, modality_mask=modality_mask)
            loss_gen = out["total_loss"]
            x0_gen = out["x0_gen"]
            if kd is not None:
                # 教师蒸馏损失（见 trainers/stage2_kd.py）
                extra = kd_losses(kd["args"], kd["teacher"], out, batch["__sample_index__"])
                for k, v in extra.items():
                    loss_gen = loss_gen + v
                    kd["sum"][k] = kd["sum"].get(k, 0.0) + float(v.item())
                kd["n"] += 1

        if not ensure_finite("loss_gen", loss_gen):
            optimizer_diff.zero_grad(set_to_none=True)
            continue

        loss_gen.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 5.0)
        optimizer_diff.step()

        # 2) 生成特征分类头更新
        loss_cls_val = 0.0
        if train_cls:
            optimizer_cls.zero_grad(set_to_none=True)
            with ac_ctx:
                logits_cls = model.gen_classifier(x0_gen.detach())
                loss_cls = criterion(logits_cls, y_idx) * lambda_cls

            if not ensure_finite("logits_cls", logits_cls) or not ensure_finite("loss_cls", loss_cls):
                optimizer_cls.zero_grad(set_to_none=True)
                continue

            loss_cls.backward()
            torch.nn.utils.clip_grad_norm_(model.gen_classifier.parameters(), 5.0)
            optimizer_cls.step()

            loss_cls_val = float(loss_cls.item())
            logits_for_metrics = logits_cls
        else:
            with torch.no_grad():
                logits_for_metrics = model.gen_classifier(x0_gen.detach())

        total_loss += float(loss_gen.item()) + loss_cls_val
        total_diff += float(out["diff_loss"].item())
        total_rec += float(out["recon_loss"].item())
        total_cls += loss_cls_val
        update_epoch_buffers(buf, sanitize_logits_for_metrics(logits_for_metrics), y_1h)

    n = max(1, len(loader))
    metrics = compute_epoch_metrics_macro(buf, num_classes=N_CLASSES)
    metrics["loss"] = total_loss / n
    metrics["diff_loss"] = total_diff / n
    metrics["recon_loss"] = total_rec / n
    metrics["cls_loss"] = (total_cls / n) if train_cls else float("nan")
    metrics["lambda_cls"] = float(lambda_cls)

    save_epoch_metrics(log_dir, "train", epoch, metrics)
    return metrics


@torch.no_grad()
def validate_one_epoch_stage2(model, loader, mask_manager, device, epoch, log_dir, sample_steps=100, split="val",
                              kd=None):
    """
    DDIM 采样 x0_gen 并用 gen_classifier 分类。
    kd 不为空时另算 teacher_gen（冻结教师 + 生成术后特征），--select_metric teacher 时返回的主指标换成它。
    """
    model.eval()
    buf = {"y_true": [], "y_pred": [], "proba": []}
    tbuf = {"y_true": [], "y_pred": [], "proba": []}
    ac_ctx = autocast_ctx(device)
    teacher = kd["teacher"] if kd is not None else None

    for batch in tqdm(loader, desc=f"{split.capitalize()} Stage2 Ep{epoch}"):
        CT_in, intra_in, xnum_in, xcat_in, modality_mask = masked_inputs(batch, mask_manager, device)
        y_1h = batch["y"].to(device)
        y_idx = y_1h.argmax(dim=-1)

        with ac_ctx:
            out = model.inference_stage2(CT=CT_in, WSI_intra_feat=intra_in, Table_num=xnum_in, Table_cat=xcat_in,
                                         steps=sample_steps, modality_mask=modality_mask)
        update_epoch_buffers(buf, sanitize_logits_for_metrics(out["logits"]), y_1h)
        if teacher is not None:
            with ac_ctx:
                t_logits = teacher.forward_gen(teacher.rows(batch["__sample_index__"]), out["x0_gen"])
            update_epoch_buffers(tbuf, sanitize_logits_for_metrics(t_logits.float()), y_1h)

    metrics = compute_epoch_metrics_macro(buf, num_classes=N_CLASSES)
    if teacher is not None:
        tm = compute_epoch_metrics_macro(tbuf, num_classes=N_CLASSES)
        write_kd_row(log_dir, split, epoch, {"teacher_gen_acc": tm["acc"], "teacher_gen_f1": tm["f1"],
                                             "teacher_gen_auroc": tm["auroc"], "gen_cls_acc": metrics["acc"]})
        if kd["args"].select_metric == "teacher":
            metrics = tm
    save_epoch_metrics(log_dir, split, epoch, metrics)
    return metrics


def write_kd_row(log_dir: str, split: str, epoch: int, vals: dict):
    """教师引导相关的额外指标逐 epoch 追加到 {log_dir}/metrics_kd.txt（key=value，便于 grep）。"""
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "metrics_kd.txt"), "a", encoding="utf-8") as f:
        f.write(f"{split} epoch={epoch} " + " ".join(f"{k}={float(v):.6f}" for k, v in vals.items()) + "\n")


def build_model(args, fold_id: int, device: str) -> PostOpGenAndCls:
    """
    构建完整模型。扩散模型的归一化统计：残差模式由 setup_residual 设为训练集残差统计、Stage-3 随 checkpoint 加载；
    给了 --x0_stats_root 时先按折读取 {x0_stats_root}/fold_k/x0_mean_std.json。
    """
    stats_path = f"{args.x0_stats_root}/fold_{fold_id}/x0_mean_std.json" if getattr(args, "x0_stats_root", None) else None
    # Stage-3 也复用本函数
    diff_cfg = DiffusionConfig(x0_stats_path=stats_path,
                               train_ddim_steps=getattr(args, "train_ddim_steps", DiffusionConfig.train_ddim_steps))
    model = PostOpGenAndCls(
        cfg=OverallModelConfig(**getattr(args, "stage_a_arch", {})),   # Stage-A 结构与 Stage-1 一致
        ct_cfg=CTEncoderConfig(feature_extraction=True),
        wsi_cfg=WSIEncoderConfig(),
        table_cfg=make_table_cfg(),
        diff_cfg=diff_cfg,
        tri_cfg=make_fusion_cfg(args),   # Stage-2 不用 CMAA
    ).to(device)
    return model


def add_args(p):
    p.add_argument("--excel", default=EXCEL)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--stage1_ckpt", default=STAGE1_RUN,
                   help="Stage-1 checkpoint：k 折运行根目录（按折取最优）/ 含 {fold_id} 的模板 / 具体文件（所有折共用）")
    p.add_argument("--stage1_mask_file", default=STAGE1_MASK_FILE, help="Stage-1 保存的 fixed_missing_masks.pt")
    p.add_argument("--agg_ckpt", default=AGG_PRETRAINED, help="术后 WSI 聚合器权重（同 --stage1_ckpt 的写法）")
    p.add_argument("--x0_stats_root", default=X0_STATS_ROOT, help="可选：含 fold_k/x0_mean_std.json 的目录")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--cls_start_epoch", type=int, default=1, help="从该 epoch 起训练生成特征分类头")
    p.add_argument("--lambda_cls_max", type=float, default=1.0, help="生成特征分类头损失的最终权重")
    p.add_argument("--lambda_warmup", type=int, default=10, help="分类头损失权重线性升到 lambda_cls_max 的 epoch 数")
    p.add_argument("--lr_diff", type=float, default=1e-4)
    p.add_argument("--lr_stage_a", type=float, default=0.0, help="Stage-A 的 lr；残差模式须为 0（冻结）")
    p.add_argument("--lr_cls", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--train_ddim_steps", type=int, default=DiffusionConfig.train_ddim_steps,
                   help="训练时 recon_loss 的可微 DDIM 采样步数")
    p.add_argument("--val_sample_steps", type=int, default=None,
                   help="验证时 DDIM 采样步数，默认与 --train_ddim_steps 相同")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--folds", type=int, nargs="*", default=None, help="只跑指定折（1-based），默认全部")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    add_kd_args(p)
    add_residual_args(p)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed if getattr(args, "train_seed", None) is None else args.train_seed)   # --train_seed 只影响训练随机性
    if args.val_sample_steps is None:
        args.val_sample_steps = args.train_ddim_steps

    full_ds = build_fourmodal_dataset(args.excel)
    mask_manager = LoadedFixedMaskManager(mask_file=args.stage1_mask_file, expected_len=len(full_ds))
    collate_fn = make_collate_fn(ct_crop_h=33)
    fold_best_results = []

    for fold_id, train_idx, val_idx, fold_dir, ckpt_dir in iter_folds(full_ds, args.n_folds, args.seed,
                                                                        args.log_dir, args.folds):
        train_loader, val_loader = make_loaders(full_ds, train_idx, val_idx, collate_fn, args.num_workers,
                                                batch_size=args.batch_size)

        # Stage-A 来自 Stage-1（跳过其分类头，结构按 Stage-1 checkpoint 记录的融合方式构建），TitanAgg 来自 wsi_post 聚合器
        stage1_ckpt = resolve_fold_ckpt(args.stage1_ckpt, fold_id)
        args.stage_a_arch = stage_a_arch(stage1_ckpt)
        model = build_model(args, fold_id, device)
        load_stage_a_from_stage1_ckpt(model, stage1_ckpt, device=device)
        load_titanagg_from_ckpt(model, resolve_fold_ckpt(args.agg_ckpt, fold_id), device=device)
        if args.residual:
            # 残差生成：加载并冻结 Anchor，扩散 mean/std 换成残差统计量（见 trainers/stage2_residual.py）
            args._fold_id = fold_id
            setup_residual(args, model, train_loader,
                           lambda b: (*masked_inputs(b, mask_manager, device), post_wsi_features(b, device)),
                           device, stage1_ckpt)

        kd = None
        if kd_enabled(args):
            if (mask_manager.masks.sum(1) < 3).any():
                raise ValueError("教师蒸馏假设所有样本三模态齐全（教师不用缺失 mask）")
            teacher = FrozenTeacher(args, resolve_fold_ckpt(args.teacher_ckpt, fold_id), device)
            teacher.build_cache(full_ds, list(train_idx) + list(val_idx), num_workers=args.num_workers)
            kd = {"args": args, "teacher": teacher, "sum": {}, "n": 0}

        # 扩散模型从头训练；Stage-A 为 Stage-1 权重（--lr_stage_a 0 冻结）。
        # TitanAgg 是扩散目标 x0_real 的来源，必须冻结（否则目标可塌缩为常数）；CMAA Stage-2 不用。
        # gen_classifier 在 cls_start_epoch 解冻，只在 x0_gen.detach() 上训练。
        for m in (model.TitanAgg, model.gen_classifier, model.cmaa):
            freeze_module(m)
        train_stage_a = args.lr_stage_a > 0
        optimizer_diff = optim.AdamW(build_param_groups([
            ("diff", model.diff, args.lr_diff),
            ("stage_a", model.stage_a, args.lr_stage_a),
        ], args.weight_decay))
        optimizer_cls = optim.AdamW(model.gen_classifier.parameters(), lr=args.lr_cls, weight_decay=args.weight_decay)
        best = BestTracker()
        topk_ckpts = []

        for epoch in range(1, args.epochs + 1):
            if epoch == args.cls_start_epoch:
                print(">>> Unfreeze gen_classifier")
                unfreeze_classifier(model)
            print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

            if kd is not None:
                kd["sum"], kd["n"] = {}, 0
            train_stats = train_one_epoch_stage2(model, train_loader, optimizer_diff, optimizer_cls, mask_manager,
                                                 device, epoch, fold_dir, cls_start_epoch=args.cls_start_epoch,
                                                 lambda_cls_max=args.lambda_cls_max, lambda_warmup=args.lambda_warmup,
                                                 train_stage_a=train_stage_a, kd=kd)
            if kd is not None and kd["n"]:
                write_kd_row(fold_dir, "train", epoch, {k: v / kd["n"] for k, v in kd["sum"].items()})
            val_stats = validate_one_epoch_stage2(model, val_loader, mask_manager, device, epoch, fold_dir,
                                                  sample_steps=args.val_sample_steps, kd=kd)
            print_epoch(epoch, train_stats, val_stats)
            best.update(val_stats)

            ckpt_payload = {"epoch": epoch, "val_acc": float(val_stats["acc"]), "model": model.state_dict(),
                            "stage1_ckpt": stage1_ckpt, "stage1_mask_file": args.stage1_mask_file,
                            "train_ddim_steps": args.train_ddim_steps, "lr_stage_a": args.lr_stage_a,
                            "teacher_ckpt": resolve_fold_ckpt(args.teacher_ckpt, fold_id) if kd is not None else None,
                            "select_metric": args.select_metric,
                            "residual": args.residual,
                            "anchor_dropout": model.anchor.drop.p if model.anchor is not None else None,
                            "anchor_hidden": model.anchor.hidden if model.anchor is not None else None,
                            **args.stage_a_arch}
            _maybe_save_topk(ckpt_dir, args.topk, topk_ckpts, epoch, float(val_stats["acc"]), ckpt_payload)

        print(f"[Fold {fold_id}] Best Val Acc = {best.acc:.4f}")
        fold_best_results.append(best.stats)

    save_kfold_summary(args.log_dir, fold_best_results,
                       stage1_mask_file=args.stage1_mask_file, stage1_ckpt_template=args.stage1_ckpt)
