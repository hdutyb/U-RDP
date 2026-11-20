# -*- coding: utf-8 -*-
"""
Stage-2 残差生成（论文的 Anchor Predictor + 残差扩散）。

    0) train.py anchor（trainers/anchor.py）：在冻结 Stage-A 输出上训练 Anchor Predictor
    1) Stage-2 用 --anchor_ckpt 加载该 Anchor 并冻结
    2) 扩散模型的 mean/std 换成训练集残差 r = x0_real - z_anchor 的统计量；扩散在 r 的 z-score 空间中进行
    3) x0_gen = z_anchor + r̂ 送入原有的 recon / 教师 / gen_classifier（与非残差版接口完全一致）
Anchor 依赖 Stage-A 输出，所以残差模式要求 --lr_stage_a 0，且 Stage-A 与训练 Anchor 时相同。
"""
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.pipeline import autocast_ctx, resolve_fold_ckpt


def add_residual_args(p):
    g = p.add_argument_group("Stage-2 残差生成")
    g.add_argument("--residual", choices=["intra", "all"], default="intra",
                   help="Anchor 输入：intra = 冰冻 WSI（论文做法），all = [CT, 冰冻 WSI, 表格]（对比）")
    g.add_argument("--anchor_ckpt", default=None,
                   help="train.py anchor 训练的 Anchor（写法同 --stage1_ckpt，可含 {fold_id}）；--residual 时必需")


@torch.no_grad()
def cache_stage_a(model, loader, encode, device):
    """encode(batch) -> (CT_in, intra_in, xnum_in, xcat_in, mask, post_feats)；返回 tokens [N,3,m]、x0_real [N,m]（float32）"""
    model.eval()
    toks, reals = [], []
    with autocast_ctx(device):
        for batch in tqdm(loader, desc="[Anchor] cache"):
            CT_in, intra_in, xnum_in, xcat_in, mask, post = encode(batch)
            sa = model.stage_a(CT=CT_in, WSI_intra_feat=intra_in, Table_num=xnum_in, Table_cat=xcat_in,
                               modality_mask=mask)
            toks.append(sa["condition_tokens"].float())
            reals.append(model.TitanAgg(post)[1].squeeze(1).float())
    return torch.cat(toks), torch.cat(reals)


def z_r2(pred, real, ref_mean, sx):
    """z-score 空间（按训练集 x0 的逐维 std）的 R² = 1 - MSE / Var"""
    return 1 - float(F.mse_loss(pred / sx, real / sx)) / float((((real - ref_mean) / sx) ** 2).mean())


def setup_residual(args, model, train_loader, encode, device, stage1_ckpt: str):
    """加载并冻结 Anchor，把 diff 的 mean/std 设为训练集残差统计量。model 已加载 Stage-A / TitanAgg（及可选热启动权重）。"""
    if args.lr_stage_a > 0:
        raise ValueError("--residual 需要 --lr_stage_a 0（Anchor 在冻结的 Stage-A 输出上训练）")
    if args.anchor_ckpt:
        path = resolve_fold_ckpt(args.anchor_ckpt, args._fold_id)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["mode"] != args.residual:
            raise ValueError(f"--residual {args.residual} 与 Anchor checkpoint 的 mode={ck['mode']} 不一致")
        model.enable_anchor(ck["mode"], ck["dropout"], ck["hidden"])
        model.anchor.load_state_dict(ck["anchor"], strict=True)
        print(f"[Anchor] loaded {path}: {ck['cfg']} (test R2={ck.get('test_r2', float('nan')):.3f})")
    else:
        raise ValueError("--residual 需要 --anchor_ckpt（先运行 python train.py anchor）")

    model.anchor.eval()
    for p in model.anchor.parameters():
        p.requires_grad_(False)
    toks, real = cache_stage_a(model, train_loader, encode, device)
    with torch.no_grad():
        za = model.anchor(toks)
        r = real - za
        mean, std = r.mean(0, keepdim=True), r.std(0, keepdim=True).clamp_min(1e-4)
        sx = real.std(0).clamp_min(1e-4)
    model.diff.set_norm(mean.to(model.diff.mean.device), std.to(model.diff.std.device))
    print(f"[Anchor] train R2={z_r2(za, real, real.mean(0), sx):.3f}; residual std mean={std.mean():.4f} "
          f"(x0 std mean={sx.mean():.4f})")
