# ============================================================
#       CPD（Conditional Pathology Decoder）：DALL·E 2 Prior 风格去噪网络
#        输入:  x_t [B,m], cond [B,m] 或 [B,1,m], t [B]
#        输出:  eps / v 预测 [B,m]
# ============================================================

import torch
import torch.nn as nn

from models.diffusion.denoiser import SinusoidalPosEmb, DropPath


# ------------------------------
# 多头 Self-Attn + FFN
# ------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads, mlp_ratio=4.0, dropout=0.0, droppath=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path_attn = DropPath(droppath)

        self.norm2 = nn.LayerNorm(hidden)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden),
            nn.Dropout(dropout)
        )
        self.drop_path_mlp = DropPath(droppath)

    def forward(self, x):
        # Self-attn
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path_attn(attn)

        # FFN
        h2 = self.norm2(x)
        h2 = self.mlp(h2)
        x = x + self.drop_path_mlp(h2)
        return x


# ------------------------------
# Cross-Attention 模块：x←cond
# ------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.0, droppath=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_k = nn.LayerNorm(hidden)

        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path_attn = DropPath(droppath)

        self.norm_out = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden)
        )
        self.drop_path_ffn = DropPath(droppath)

    def forward(self, x, cond):
        """
        x:    [B,1,H]
        cond: [B,C,H]
        """
        q = self.norm_q(x)
        k = self.norm_k(cond)
        v = k

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = x + self.drop_path_attn(attn_out)

        h = self.norm_out(x)
        h = self.ffn(h)
        x = x + self.drop_path_ffn(h)
        return x


class CondDenoiser_Prior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.m = cfg.m
        self.hidden = cfg.hidden
        self.depth = cfg.depth

        # ===== 1. 时间 embedding =====
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(cfg.time_emb_dim),
            nn.Linear(cfg.time_emb_dim, cfg.time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.time_emb_dim * 2, self.hidden)
        )

        # ===== 2. 输入映射 =====
        self.x_in = nn.Linear(cfg.m, self.hidden)
        self.cond_in = nn.Linear(cfg.cond_dim, self.hidden)

        # ===== 3. FiLM 调制（cond→hidden）=====
        self.film_scale = nn.Linear(self.hidden, self.hidden)
        self.film_shift = nn.Linear(self.hidden, self.hidden)

        # ===== 4. Cross-attn tokens =====
        self.cond_norm = nn.LayerNorm(self.hidden)

        # 可选 cls token
        self.use_cls = (cfg.cls_num is not None and cfg.cls_num > 0)
        if self.use_cls:
            self.cls_emb = nn.Embedding(cfg.cls_num, self.hidden)

        # ===== 5. Transformer blocks =====
        self.self_blocks = nn.ModuleList([
            TransformerBlock(
                hidden=self.hidden,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                droppath=cfg.droppath * (i + 1) / cfg.depth
            )
            for i in range(cfg.depth)
        ])

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                hidden=self.hidden,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                droppath=cfg.droppath * (i + 1) / cfg.depth
            )
            for i in range(cfg.depth)
        ])

        # ===== 6. 输出 eps =====
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, cfg.m)
        )

        # ---- 修复：不要 zero init，使用 xavier_uniform ----
        nn.init.xavier_uniform_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    # =========================================================
    # forward
    # =========================================================
    def forward(self, x_t, t, cond, cls=None):
        """
        x_t:   [B, m]
        cond:  [B, m]
        t:     [B]
        """
        # 支持 [B,1,m] 形式
        if x_t.dim() == 3 and x_t.size(1) == 1:
            x_t = x_t.squeeze(1)
        if cond.dim() == 3 and cond.size(1) == 1:
            cond = cond.squeeze(1)

        B = x_t.size(0)

        # ====== 1) 编码 x ======
        h = self.x_in(x_t)  # [B,H]

        # ====== 2) 编码 cond ======
        cond_vec = self.cond_in(cond)  # [B,H]

        # ====== 3) FiLM：注入条件语义 ======
        scale = self.film_scale(cond_vec)
        shift = self.film_shift(cond_vec)
        h = h * (scale + 1) + shift

        # ====== 4) 时间 embedding：加到主干流（关键）======
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.full((B,), int(t), dtype=torch.long, device=x_t.device)
        t_tok = self.time_emb(t)  # [B,H]

        h = h + t_tok            # ❗正确注入时间信息

        # ====== 5) cond_tokens 只包含 cond_vec（不包含 t_emb）======
        cond_tokens = cond_vec.unsqueeze(1)   # [B,1,H]

        if self.use_cls and cls is not None:
            cls = cls.long()
            cls_tok = self.cls_emb(cls).unsqueeze(1)  # [B,1,H]
            cond_tokens = torch.cat([cond_tokens, cls_tok], dim=1)

        cond_tokens = self.cond_norm(cond_tokens)     # [B,L,H]

        # ====== 6) 扩展为序列 ======
        h = h.unsqueeze(1)     # [B,1,H]

        # ====== 7) 主干 Transformer ======
        for blk_self, blk_cross in zip(self.self_blocks, self.cross_blocks):
            h = blk_self(h)
            h = blk_cross(h, cond_tokens)

        h = h.squeeze(1)       # [B,H]

        return self.out(h)     # → [B, m]

