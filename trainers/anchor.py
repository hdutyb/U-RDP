# -*- coding: utf-8 -*-
"""
Anchor Predictor 训练（Stage-2 --residual 用 --anchor_ckpt 加载）。

    冻结 Stage-A（--stage1_ckpt）编码 CT / 冰冻 WSI / 表格 -> condition_tokens；冻结 TitanAgg 聚合术后 WSI -> x0_real
    Anchor（models/diffusion/anchor.py）回归 x0_real：min ‖(z_anchor - x0_real) / σ_x0‖²
    训练集内 10% 早停（选 epoch），报告 test 折 z-score 空间 R²；按 5 折 test R² 均值统一选配置

    python train.py anchor --stage1_ckpt ... --log_dir <dir>              # 网格搜索 -> fold_k/anchor_results.json
    python train.py anchor ... --fit intra_h0_d0.2_wd0.1_lr0.0003         # 用选中的配置重训并保存 fold_k/<cfg>.pth
特征缓存在 fold_k/anchor_cache.pt，两次调用共用。
"""
import itertools
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.paths import EXCEL, AGG_PRETRAINED, X0_STATS_ROOT, STAGE1_RUN, STAGE1_MASK_FILE
from models.diffusion.anchor import AnchorPredictor
from trainers.stage2 import build_model, masked_inputs, post_wsi_features
from trainers.stage2_residual import cache_stage_a, z_r2
from utils.missing_modality import IndexedSubset
from utils.pipeline import (LoadedFixedMaskManager, build_fourmodal_dataset, iter_folds, load_stage_a_from_stage1_ckpt,
                            load_titanagg_from_ckpt, make_collate_fn, resolve_fold_ckpt, stage_a_arch)
from utils.utils import set_seed


def add_args(p):
    p.add_argument("--excel", default=EXCEL)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--stage1_ckpt", default=STAGE1_RUN)
    p.add_argument("--stage1_mask_file", default=STAGE1_MASK_FILE)
    p.add_argument("--agg_ckpt", default=AGG_PRETRAINED)
    p.add_argument("--x0_stats_root", default=X0_STATS_ROOT)
    p.add_argument("--modes", nargs="+", default=["intra", "all"])
    p.add_argument("--hidden", type=int, nargs="+", default=[0, 256, 768, 1536], help="0 = 线性")
    p.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.2, 0.5])
    p.add_argument("--wd", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    p.add_argument("--lr", type=float, nargs="+", default=[1e-3, 3e-4])
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--fit", default=None, help="只重训并保存该配置（名字同 anchor_results.json 的 key）")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)


def cfg_name(mode, hidden, dropout, wd, lr):
    return f"{mode}_h{hidden}_d{dropout:g}_wd{wd:g}_lr{lr:g}"


def parse_cfg(name):
    mode, h, d, wd, lr = name.split("_")
    return mode, int(h[1:]), float(d[1:]), float(wd[2:]), float(lr[2:])


def fit_one(toks_tr, real_tr, sx, mode, hidden, dropout, wd, lr, max_epochs, patience, seed):
    """训练集内 10% 早停；返回 (anchor, holdout z-MSE, 停止 epoch)"""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    anchor = AnchorPredictor(real_tr.size(1), mode, dropout, hidden)
    perm = torch.randperm(len(real_tr), generator=g)
    n_ho = max(1, len(real_tr) // 10)
    ho, tr = perm[:n_ho], perm[n_ho:]
    opt = torch.optim.AdamW(anchor.parameters(), lr=lr, weight_decay=wd)

    def zmse(idx):
        return F.mse_loss(anchor(toks_tr[idx]) / sx, real_tr[idx] / sx)

    best, best_sd, best_ep, bad = float("inf"), None, 0, 0
    for ep in range(1, max_epochs + 1):
        anchor.train()
        for b in tr[torch.randperm(len(tr), generator=g)].split(64):
            opt.zero_grad(set_to_none=True)
            zmse(b).backward()
            opt.step()
        anchor.eval()
        with torch.no_grad():
            v = float(zmse(ho))
        if v < best - 1e-5:
            best, best_sd, best_ep, bad = v, {k: t.clone() for k, t in anchor.state_dict().items()}, ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    anchor.load_state_dict(best_sd)
    return anchor.eval(), best, best_ep


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_ds = build_fourmodal_dataset(args.excel)
    mm = LoadedFixedMaskManager(mask_file=args.stage1_mask_file, expected_len=len(full_ds))
    col = make_collate_fn(ct_crop_h=33)

    for fold_id, train_idx, val_idx, fold_dir, _ in iter_folds(full_ds, args.n_folds, args.seed, args.log_dir,
                                                               args.folds):
        test_idx = val_idx   # 4:1 划分：第二个集合即 test 折
        stage1_ckpt = resolve_fold_ckpt(args.stage1_ckpt, fold_id)
        cache_path = os.path.join(fold_dir, "anchor_cache.pt")
        if os.path.isfile(cache_path):
            c = torch.load(cache_path, map_location="cpu")
        else:
            args.stage_a_arch = stage_a_arch(stage1_ckpt)
            model = build_model(args, fold_id, device)
            load_stage_a_from_stage1_ckpt(model, stage1_ckpt, device=device)
            load_titanagg_from_ckpt(model, resolve_fold_ckpt(args.agg_ckpt, fold_id), device=device)
            enc = lambda b: (*masked_inputs(b, mm, device), post_wsi_features(b, device))
            c = {}
            for split, idx in (("train", train_idx), ("test", test_idx)):
                loader = DataLoader(IndexedSubset(full_ds, list(idx)), batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=col)
                c[f"tok_{split}"], c[f"real_{split}"] = (t.cpu() for t in cache_stage_a(model, loader, enc, device))
            c["stage1_ckpt"] = stage1_ckpt
            torch.save(c, cache_path)
            del model
        assert c["stage1_ckpt"] == stage1_ckpt, "anchor_cache.pt 来自不同的 Stage-A，请删除后重跑"
        ttr, rtr, tte, rte = c["tok_train"], c["real_train"], c["tok_test"], c["real_test"]
        sx, mu = rtr.std(0).clamp_min(1e-4), rtr.mean(0)
        base = {"test_r2_intra_as_is": z_r2(tte[:, 1], rte, mu, sx), "test_r2_train_mean": z_r2(mu.expand_as(rte), rte, mu, sx)}

        names = [args.fit] if args.fit else [cfg_name(*g) for g in itertools.product(
            args.modes, args.hidden, args.dropout, args.wd, args.lr)]
        res = {}
        for name in names:
            mode, hidden, dropout, wd, lr = parse_cfg(name)
            anchor, ho, ep = fit_one(ttr, rtr, sx, mode, hidden, dropout, wd, lr, args.max_epochs, args.patience,
                                     args.seed)
            with torch.no_grad():
                r2 = z_r2(anchor(tte), rte, mu, sx)
            res[name] = {"test_r2": r2, "holdout_zmse": ho, "epoch": ep}
            print(f"[Fold {fold_id}] {name}: test R2={r2:.4f} holdout z-MSE={ho:.4f} ep={ep}", flush=True)
            if args.fit:
                torch.save({"anchor": anchor.state_dict(), "mode": mode, "hidden": hidden, "dropout": dropout,
                            "cfg": name, "test_r2": r2, "stage1_ckpt": stage1_ckpt},
                           os.path.join(fold_dir, f"{name}.pth"))
        if not args.fit:
            with open(os.path.join(fold_dir, "anchor_results.json"), "w") as f:
                json.dump({"baselines": base, "results": res}, f, indent=1)
            print(f"[Fold {fold_id}] baselines {base}")
