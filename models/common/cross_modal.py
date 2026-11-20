# -*- coding: utf-8 -*-
"""
论文 Sec. III-C 的跨模态注意力聚合（CMAA）组件，任意模态数 M 通用：
    GatedCrossModalAttention / CrossModalAttentionStack：每个模态 token 作为 query，只关注其余模态（门控值）
    ModalityScoreFusion：共享两层 MLP 逐模态打分，softmax 加权求和
四模态融合（models.FourModalFusionNetwork.CMAA）与三模态 Stage-A（models.featureExtraction.StageAFeatureExtractor）共用。
"""
import torch
from torch import nn


class GatedCrossModalAttention(nn.Module):
    """
    论文 Eq. (10)(11) 的一层门控跨模态注意力。
        Q = W_Q q,  K = W_K C,  V = W_V C,  R = W_R C      C 为其余模态（不含自身）
        o = (σ(R) ⊙ V)ᵀ softmax(Qᵀ K / sqrt(d_h))         softmax 在 context 模态上，σ(R) 逐维门控
        y = LN(q + W_O concat_h(o_h))                       残差 + LayerNorm，无 FFN
    缺失模态不作为 context；其自身输出由调用方置零。
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, (d_model, nhead)
        self.h, self.dh = nhead, d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.r = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, token_mask=None):
        """x: [B,M,D]；token_mask: [B,M] True=存在。返回 [B,M,D]。"""
        B, M, D = x.shape
        split = lambda t: t.view(B, M, self.h, self.dh).transpose(1, 2)        # [B,H,M,dh]
        q, k, v, r = split(self.q(x)), split(self.k(x)), split(self.v(x)), split(self.r(x))
        v = torch.sigmoid(r) * v                                               # 逐维门控 context 值

        scores = q @ k.transpose(-1, -2) / self.dh ** 0.5                      # [B,H,M(query),M(key)]
        allow = ~torch.eye(M, dtype=torch.bool, device=x.device)               # 不关注自身
        allow = allow.view(1, 1, M, M).expand(B, 1, M, M)
        if token_mask is not None:
            allow = allow & token_mask.bool().view(B, 1, 1, M)                 # 缺失模态不作为 context
        scores = scores.float().masked_fill(~allow, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)        # context 全缺失 -> 输出 0
        out = (self.drop(attn).to(v.dtype) @ v).transpose(1, 2).reshape(B, M, D)
        return self.norm(x + self.drop(self.o(out)))


class CrossModalAttentionStack(nn.Module):
    """L 层 GatedCrossModalAttention；每层后把缺失模态 token 置零。"""

    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([GatedCrossModalAttention(d_model, nhead, dropout) for _ in range(num_layers)])

    def forward(self, x, token_mask=None):
        for layer in self.layers:
            x = layer(x, token_mask)
            if token_mask is not None:
                x = x * token_mask.unsqueeze(-1).to(x.dtype)
        return x


class ModalityScoreFusion(nn.Module):
    """
    论文 Eq. (12)：a = softmax_m(ψ(y_m))，f = Σ_m a_m y_m，ψ 为共享两层 MLP（隐层 hidden）。
    tokens [B,M,D]，token_mask [B,M]（缺失模态得分置 -inf）。返回 (f [B,D], a [B,M])。
    """

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.psi = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, tokens, token_mask=None):
        logits = self.psi(tokens).squeeze(-1).float()                          # [B,M]
        if token_mask is not None:
            if (token_mask.sum(dim=1) == 0).any():
                raise ValueError("At least one modality must be available.")
            logits = logits.masked_fill(~token_mask.bool(), float("-inf"))
        alpha = torch.softmax(logits, dim=-1).to(tokens.dtype)
        return torch.sum(tokens * alpha.unsqueeze(-1), dim=1), alpha


def _linear_proj(in_dim, out_dim):
    """模态投影 y = W z + b（论文 Sec. III-C）；包成 Sequential（checkpoint key 为 proj_*.0.*）。"""
    blk = nn.Sequential(nn.Linear(in_dim, out_dim))
    nn.init.xavier_uniform_(blk[0].weight)
    nn.init.zeros_(blk[0].bias)
    return blk
