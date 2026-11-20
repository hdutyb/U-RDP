# -*- coding: utf-8 -*-
"""
一维条件扩散（DDPM 训练 + DDIM 采样），用于由 Stage-A condition 生成术后 WSI 特征。

    train_step: p_losses（噪声/v 回归）+ 可微 DDIM 采样得到 x0_gen，用于重建损失
    sample:     no_grad DDIM 采样，返回反标准化后的 x0_gen [B,m]
所有计算在标准化空间：x0_norm = (x0 - mean) / std（mean/std 由 set_norm 设置）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.configs import DiffusionConfig


class GaussianDiffusion1D(nn.Module):
    """
    model: 去噪网络（CPD，models.diffusion.denoiser_DALL.CondDenoiser_Prior）
    cfg.pred_type: "eps" 预测噪声 / "v" 预测 v = alpha*eps - sigma*x0
    """
    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.m = cfg.m

        self.model = model

        # x0 标准化统计量（buffer，随 checkpoint 保存）
        self.register_buffer("mean", torch.zeros(1, self.m))
        self.register_buffer("std", torch.ones(1, self.m))

        def set_norm(mean: torch.Tensor, std: torch.Tensor):
            self.mean.copy_(mean)
            self.std.copy_(std)

        self.set_norm = set_norm

        # ---------- 噪声日程 beta_t / alpha_bar_t ----------
        T = cfg.timesteps

        if cfg.schedule == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float32)
            abar = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            abar = abar / abar[0]
            betas = 1 - (abar[1:] / abar[:-1])
            betas = betas.clamp(1e-8, 0.999)

        elif cfg.schedule == "linear":
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, dtype=torch.float32)

        else:
            raise ValueError("schedule must be 'cosine' or 'linear'")

        alphas = 1 - betas
        abar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', abar)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(abar))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - abar))

        abar_prev = torch.cat([torch.ones(1), abar[:-1]], dim=0)
        posterior_variance = betas * (1 - abar_prev) / (1 - abar)
        self.register_buffer('posterior_variance', posterior_variance)

    def _cond(self, cond):
        """cond: [B,m] 或 [B,1,m] -> [B,m]"""
        if cond.dim() == 3 and cond.size(1) == 1:
            return cond.squeeze(1)
        if cond.dim() == 2:
            return cond
        raise ValueError(f"Unsupported cond shape: {cond.shape}")

    def _gather_coeff(self, buf, t, B):
        """按时间步 t（int 或 [B]）取日程系数，返回 [B,1]。"""
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            val = buf[int(t)].expand(B)
        else:
            val = buf[t]
        return val.view(B, 1)

    def q_sample(self, x0, t, noise=None):
        """前向加噪：x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * noise"""
        B, m = x0.shape
        if noise is None:
            noise = torch.randn_like(x0)

        fac1 = self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
        fac2 = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
        return fac1 * x0 + fac2 * noise

    def x0_from_xt_eps(self, x_t, t, eps_pred):
        B, m = x_t.shape
        inv = 1.0 / self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
        s1m = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
        x0 = inv * (x_t - s1m * eps_pred)
        return x0.clamp(-1, 1) if self.cfg.clip_x0 else x0

    def _pred_to_x0_eps(self, x_t, t, pred, B):
        """把网络输出（eps 或 v）换算成 (x0_hat, eps_hat)。"""
        if self.cfg.pred_type == "eps":
            alpha = self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
            sigma = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
            x0_hat = (x_t - sigma * pred) / (alpha + 1e-12)
            eps_hat = pred

        elif self.cfg.pred_type == "v":
            alpha = self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
            sigma = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
            x0_hat = alpha * x_t - sigma * pred
            eps_hat = sigma * x_t + alpha * pred

        else:
            raise ValueError(f"Unknown pred_type: {self.cfg.pred_type}")

        if self.cfg.clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)

        return x0_hat, eps_hat

    def p_losses(self, x0, cond, cls=None):
        """随机采一个 t，回归噪声（eps）或 v；x0 为原始空间特征 [B,m]。"""
        B, m = x0.shape
        device = x0.device

        x0_norm = (x0 - self.mean) / self.std

        cond_norm = self._cond(cond)

        t = torch.randint(0, self.cfg.timesteps, (B,), device=device)
        noise = torch.randn_like(x0_norm)
        x_t = self.q_sample(x0_norm, t, noise=noise)

        pred = self.model(x_t, t, cond_norm, cls)

        if self.cfg.pred_type == "eps":
            target = noise
        else:
            alpha = self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
            sigma = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
            target = alpha * noise - sigma * x0_norm

        elem = (pred - target) ** 2 if self.cfg.loss_type == "l2" else (pred - target).abs()
        return elem.mean()

    def _ddim_step_trainable(self, x_t, t, t_prev, cond_norm, cls=None):
        """可微的单步 DDIM（eta=0）；最后一步 t_prev == t 时直接返回 x0_hat。"""
        B, m = x_t.shape
        abar_prev = self._gather_coeff(self.alphas_cumprod, t_prev, B)

        pred = self.model(x_t, t, cond_norm, cls)
        x0_hat, eps = self._pred_to_x0_eps(x_t, t, pred, B)

        sqrt_prev = torch.sqrt(abar_prev)
        sqrt_one_minus_prev = torch.sqrt(1 - abar_prev)

        xt_prev = sqrt_prev * x0_hat + sqrt_one_minus_prev * eps
        same_mask = (t_prev == t).view(B, 1)

        return torch.where(same_mask, x0_hat, xt_prev)

    def _make_subseq(self, steps):
        """DDIM 时间步子序列：从 T-1 均匀递减到 0（首尾必含）。"""
        T = self.cfg.timesteps
        if steps >= T:
            return torch.arange(T - 1, -1, -1)
        idx = torch.round(torch.linspace(T - 1, 0, steps)).long()
        idx = torch.unique_consecutive(idx)
        if idx[0] != T - 1:
            idx = torch.cat([torch.tensor([T - 1]), idx])
        if idx[-1] != 0:
            idx = torch.cat([idx, torch.tensor([0])])
        return idx

    def sample_trainable(self, cond_tokens, cls=None, steps=None):
        """可微 DDIM 采样（保留梯度），返回标准化空间的 x0 [B,m]。"""
        cond_norm = self._cond(cond_tokens)

        B = cond_norm.shape[0]
        device = self.mean.device

        x_t = torch.randn(B, self.cfg.m, device=device)

        steps = steps or self.cfg.train_ddim_steps
        idx = self._make_subseq(steps).to(device)

        for i in range(len(idx)):
            t = idx[i].repeat(B)
            t_prev = (idx[i] if i == len(idx) - 1 else idx[i + 1]).repeat(B)

            x_t = self._ddim_step_trainable(
                x_t, t, t_prev, cond_norm, cls
            )

        return x_t

    @torch.no_grad()
    def sample(self, cond, cls=None, steps=None):
        """no_grad DDIM 采样；cond [B,m] 或 [B,1,m]，返回原始空间 x0_gen [B,m]。"""
        cond_norm = self._cond(cond)

        B = cond_norm.shape[0]
        device = self.mean.device

        x_t = torch.randn(B, self.cfg.m, device=device)

        steps = steps or self.cfg.timesteps
        idx = self._make_subseq(steps).to(device)

        for i in range(len(idx)):
            t = idx[i].repeat(B)
            t_prev = (idx[i] if i == len(idx) - 1 else idx[i + 1]).repeat(B)

            pred = self.model(x_t, t, cond_norm, cls)
            x0_hat, eps = self._pred_to_x0_eps(x_t, t, pred, B)

            abar_prv = self._gather_coeff(self.alphas_cumprod, t_prev, B)

            x_t = torch.where(
                (t_prev == t).view(B, 1),
                x0_hat,
                torch.sqrt(abar_prv) * x0_hat + torch.sqrt(1 - abar_prv) * eps
            )

        return x_t * self.std + self.mean

    def train_step(self, x0, cond, cls=None, steps_ddim=None):
        """
        返回 diff_loss（噪声/v 回归）、recon_loss（可微采样 x0_gen 与 x0 的 MSE）、x0_gen（原始空间 [B,m]）。
        """
        diff_loss = self.p_losses(x0, cond, cls)

        x0_gen_norm = self.sample_trainable(
            cond,
            cls,
            steps=steps_ddim or self.cfg.train_ddim_steps
        )

        x0_gen = x0_gen_norm * self.std + self.mean
        rec_loss = F.mse_loss(x0_gen, x0)

        return {
            "diff_loss": diff_loss,
            "recon_loss": rec_loss,
            "x0_gen": x0_gen,
        }
