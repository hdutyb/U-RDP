"""
CT 编码器：r3d_18（3D ResNet，首层改为单通道，BN 全部替换为 GN/IN）+ 线性投影到 out_dim。
feature_extraction=True 时输出特征 [B,out_dim]，否则输出分类 logits。
"""
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from config.configs import CTEncoderConfig
from models.common.common import LinearHead


# 选一个能整除通道数的 GN 分组
def _pick_gn_groups(num_channels: int, preferred: int = 16) -> int:
    if num_channels % preferred == 0:
        return preferred
    for g in range(min(preferred, num_channels), 1, -1):
        if num_channels % g == 0:
            return g
    return 1  # 最坏情况每通道一组

def _patch_first_conv_to_1ch(model: nn.Module):
    # torchvision r3d_18 的首层通常在 model.stem[0]
    conv = model.stem[0]
    if isinstance(conv, nn.Conv3d) and conv.in_channels == 3:
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None),
        )
        with torch.no_grad():
            # 若使用了预训练权重，则把原 3 通道权重在通道维做均值初始化到 1 通道
            w = conv.weight  # [out, 3, kT, kH, kW]
            new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        model.stem[0] = new_conv


def replace_bn_with_gn_or_in(module: nn.Module, gn_groups: int = 16):
    """把所有 BN/SyncBN 替换为 GN（能整除）或 IN（兜底）。完全移除 BN，从而没有 BN momentum。"""
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm3d, nn.SyncBatchNorm)):
            C = child.num_features
            g = _pick_gn_groups(C, gn_groups)
            if g > 1 and (C % g == 0):
                setattr(module, name, nn.GroupNorm(g, C, affine=True))
            else:
                setattr(module, name, nn.InstanceNorm3d(C, affine=True, track_running_stats=False))
        else:
            replace_bn_with_gn_or_in(child, gn_groups)

class CtEncoder(nn.Module):
    def __init__(self, config: CTEncoderConfig):
        super().__init__()
        self.cfg = config
        self.feature_extraction = getattr(config, "feature_extraction")

        weights = R3D_18_Weights.KINETICS400_V1 if getattr(config, "use_pretrained") else None
        base = r3d_18(weights=weights)
        _patch_first_conv_to_1ch(base)  # 3 通道 -> 1 通道（权重取通道均值）
        in_features = base.fc.in_features
        base.fc = nn.Identity()

        # 彻底移除 BN：统一替换为 GN/IN
        replace_bn_with_gn_or_in(base, gn_groups=getattr(config, "gn_groups", 16))

        self.backbone = base
        out_dim = getattr(config, "out_dim", 768)
        n_classes = getattr(config, "n_classes", 3)

        self.proj = nn.Linear(in_features, out_dim)
        self.classifier = LinearHead(out_dim, hidden_dim=512, out_dim=n_classes)

        # 注意：这里会对所有 Conv3d 重新做 kaiming 初始化，覆盖了上面加载的 Kinetics 预训练权重，
        # 因此 use_pretrained=True 实际不起作用（下游都会再加载 train_encoder 训练出的 CT 权重）。
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm3d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)      # [B, C]
        feat = self.proj(feat)       # [B, out_dim]
        if self.feature_extraction:
            return feat
        return self.classifier(feat)

    def make_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-5, backbone_lr_scale: float = 0.1):
        # 1) 拆分 decay / no_decay
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            is_bias = lname.endswith("bias")
            is_norm = ("norm" in lname) or any(k in lname for k in ["bn", "gn", "in", "ln"])
            if is_bias or is_norm:
                no_decay.append(p)
            else:
                decay.append(p)

        # 2) 识别 backbone 参数（按对象身份，而不是值比较）
        backbone_ids = {id(p) for _, p in self.backbone.named_parameters() if p.requires_grad}

        backbone_decay = [p for p in decay if id(p) in backbone_ids]
        backbone_no_decay = [p for p in no_decay if id(p) in backbone_ids]
        head_decay = [p for p in decay if id(p) not in backbone_ids]
        head_no_decay = [p for p in no_decay if id(p) not in backbone_ids]

        param_groups = []
        if backbone_decay:
            param_groups.append({"params": backbone_decay, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay})
        if backbone_no_decay:
            param_groups.append({"params": backbone_no_decay, "lr": lr * backbone_lr_scale, "weight_decay": 0.0})
        if head_decay:
            param_groups.append({"params": head_decay, "lr": lr, "weight_decay": weight_decay})
        if head_no_decay:
            param_groups.append({"params": head_no_decay, "lr": lr, "weight_decay": 0.0})

        return torch.optim.AdamW(param_groups)
