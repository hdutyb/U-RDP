import torch
import torch.nn as nn

from .agg import build_aggregator


# ------------------------------
# 线性分类头 768 -> 3
# ------------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=3, dropout_rate=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x.float())   # [B, in_dim] -> [B, hidden_dim]
        x = self.relu(x)          # [B, hidden_dim] -> [B, hidden_dim]
        x = self.dropout(x)       # [B, hidden_dim] -> [B, hidden_dim]
        x = self.fc2(x)           # [B, hidden_dim] -> [B, out_dim]
        return x


# ------------------------------
# 融合分类头：Linear -> LN -> SiLU -> Dropout -> Linear
# ------------------------------
class Classifier(nn.Module):
    def __init__(self, tri_dim: int, num_classes: int, dropout: float = 0.1, hidden_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(tri_dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(tri_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        for m_ in self.net:
            if isinstance(m_, nn.Linear):
                nn.init.xavier_uniform_(m_.weight)
                nn.init.zeros_(m_.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


class TitanWithAggregator(nn.Module):
    """
    术后 WSI 多切片聚合：一个病例的 N 个预计算 TITAN slide 特征 [768] -> 病例级特征 z [1,768]。
    forward 返回 (logits [1,out_dim], z [1,768])；Stage-2 以 z 作为扩散目标 x0_real。
    agg 可选值见 models.common.agg.AGGREGATORS；norm_out 参数保留但未使用。
    """

    def __init__(self, agg: str = "max_pooling", norm_out: bool = True, out_dim: int = 3):
        super().__init__()
        self.aggregator = build_aggregator(agg)
        self.head = LinearHead(in_dim=768, out_dim=out_dim)

    def forward(self, features):
        # features: List[Tensor[768]]，单个病例的 N 张切片特征 -> (logits [1,out], z [1,768])
        if isinstance(features, (list, tuple)) and isinstance(features[0], torch.Tensor):
            E = torch.stack(list(features), dim=0)  # [N, 768]
            z = self.aggregator(E)                  # [1, 768]
            z_logit = self.head(z)
            return z_logit, z

        # features: List[List[Tensor[768]]]，batch 内 B 个病例（切片数各不相同）
        # 逐病例聚合后拼接 -> (logits [B,out], z [B,768])
        if isinstance(features, (list, tuple)) and isinstance(features[0], (list, tuple)):
            outs = [self.forward(case) for case in features]
            return torch.cat([o[0] for o in outs], dim=0), torch.cat([o[1] for o in outs], dim=0)

        return None
