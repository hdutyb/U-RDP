"""CPD 共用组件：正弦时间编码、DropPath。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================
#           时间编码
# ===========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale: float = 1.0,
        use_2pi: bool = False
    ):
        super().__init__()
        assert dim >= 2

        self.dim = dim
        self.base = float(base)
        self.scale = float(scale)
        self.use_2pi = use_2pi

        half = dim // 2
        denom = max(half - 1, 1)
        exponents = torch.arange(half, dtype=torch.float32) * (
            -math.log(self.base) / denom
        )
        freqs = torch.exp(exponents)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor):
        """
        t: [B] or [B, 1]
        return: [B, dim]
        """
        t = t.float().view(-1, 1) * self.scale

        if self.use_2pi:
            t = t * (2 * math.pi)

        args = t * self.freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


# ===========================================
#               DropPath
# ===========================================
class DropPath(nn.Module):
    """DropPath for [B, H] or [B, C, H]."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob or 0.0)

    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)

        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * mask
