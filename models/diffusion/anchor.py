# -*- coding: utf-8 -*-
"""
Anchor Predictor（论文 Eq. anchor_residual）。

    z_anchor = intra + W · Drop(LN(intra)) + b          hidden = 0（论文最终结构，线性）
             = intra + W2 · Drop[ReLU(W1 · LN(intra))]  hidden > 0（对比用的残差 MLP）
    r        = x0_real - z_anchor                            扩散模型学习生成 r（在 r 的逐维 z-score 空间中）
    x0_gen   = z_anchor + r̂                                 送入下游（gen_classifier / 教师 / Stage-3 CMAA）

intra 与术后 WSI 同为 TITAN 768 维 slide 特征，所以以 intra 为 skip 基底。
Anchor 由 train.py anchor 单独训练（trainers/anchor.py），Stage-2 用 --anchor_ckpt 加载后冻结。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 输入：Stage-A condition_tokens [B,3,m] = [CT, WSI_intra, Table] 中的下标。"intra" 为论文做法；"all" 为对比用的全模态输入
ANCHOR_INPUTS = {"intra": (1,), "all": (0, 1, 2)}


class AnchorPredictor(nn.Module):
    def __init__(self, m: int = 768, mode: str = "intra", dropout: float = 0.2, hidden: int = 768):
        super().__init__()
        self.mode, self.hidden = mode, hidden
        self.idx = ANCHOR_INPUTS[mode]
        d_in = m * len(self.idx)
        self.ln = nn.LayerNorm(d_in)
        if hidden > 0:
            self.W1 = nn.Linear(d_in, hidden)
            self.W2 = nn.Linear(hidden, m)
        else:
            self.W2 = nn.Linear(d_in, m)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.W2.weight)   # 初始 z_anchor = intra
        nn.init.zeros_(self.W2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: Stage-A 的 condition_tokens [B,3,m]（缺失模态已置零）-> z_anchor [B,m]"""
        u = self.ln(torch.cat([tokens[:, i] for i in self.idx], dim=-1).float())
        h = self.drop(F.relu(self.W1(u))) if self.hidden > 0 else self.drop(u)
        return tokens[:, 1].float() + self.W2(h)
