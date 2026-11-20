from dataclasses import dataclass, field
from typing import Tuple, Literal, Dict, Optional, List

# 公共变量
n_classes = 3
output_dim = 768    # 所有模态对齐维度（= TITAN slide 特征维度；CT/表格预训练权重与 x0 统计均为 768）

@dataclass
class CTEncoderConfig:
    out_dim: int = output_dim
    bn_momentum: float = 0.1
    bn_eps: float = 1e-5
    pretrained: str = None
    dropout: float = 0.0
    output_dir: str = None
    # 训练相关参数
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = False
    sync_bn: bool = False
    n_classes: int = n_classes
    gn_groups: int = 16
    feature_extraction: bool = True
    use_pretrained: bool = True

    @classmethod
    def from_args(cls, args):
        """从argparse.Namespace对象创建配置"""
        return cls(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_amp=args.amp,
            sync_bn=args.sync_bn,
            n_classes=args.n_classes,
            output_dir=args.output_dir
        )

@dataclass
class TableEncoderConfig:
    # ===== 输入特征相关 =====
    num_features: int = 4
    """连续特征（数值型列）的数量，如年龄、收入等"""
    cat_cardinalities: Optional[List[int]] = (50, 20)
    """类别特征的取值规模列表，比如 [50, 20] 表示有两列类别特征：
       第一列城市有50种取值，第二列职业有20种取值"""
    emb_dim: int = 32
    """每个类别特征的Embedding维度，所有类别列共享该维度"""
    # ===== 模型结构相关 =====
    hidden_dims: Tuple[int, ...] = (256, 256)
    """MLP隐藏层的宽度列表，表示每一层的神经元个数"""
    out_dim: int = output_dim
    """编码后的特征输出维度（feature embedding size），用于下游融合/分类"""
    activation: Literal["relu", "gelu", "none"] = "relu"
    """激活函数类型，推荐 silu/gelu"""
    dropout: float = 0.1
    """Dropout 比例，0~0.5之间，一般0.1~0.3较合适"""
    use_attention: bool = True
    """是否启用 Transformer 注意力层来增强特征交互"""
    n_heads: int = 4
    """多头注意力的头数，仅在 use_attention=True 时有效"""
    n_attn_layers: int = 1
    """Transformer Encoder 堆叠层数，1~2层即可"""
    feature_extraction: bool = True
    """是否只输出特征向量 (out_dim)，
       True=输出 [B, out_dim]，
       False=输出分类预测 [B, n_classes]"""
    # ===== 任务相关 =====
    n_classes: int = 3
    """分类任务的类别数，如果是二分类就写2"""
    # ===== 训练相关 =====
    lr: float = 1e-3
    """学习率，AdamW通常1e-3起步"""
    weight_decay: float = 1e-4
    """L2正则化系数"""
    batch_size: int = 32
    """每个batch的大小"""
    epochs: int = 100
    """训练轮数"""
    use_amp: bool = False
    """是否使用混合精度 (Automatic Mixed Precision, AMP)"""
    sync_bn: bool = False
    """是否使用同步 BatchNorm（多GPU时可选）"""
    # ===== 预训练 & 输出 =====
    pretrained_path: Optional[str] = None
    """预训练权重路径（如果有的话可以加载）"""
    output_dir: Optional[str] = None
    """训练日志/模型的输出目录"""

    @classmethod
    def from_args(cls, args):
        """从argparse.Namespace对象创建配置"""
        return cls(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_amp=args.amp,
            n_classes=args.n_classes,
            output_dir=args.output_dir
        )

@dataclass
class WSIEncoderConfig:
    embed_dim: int = 1024
    # 预训练 CLAM 相关
    model_type: Literal["clam_sb", "clam_mb"] = "clam_mb"
    model_size: Literal["small", "big"] = "small"
    dropout: float = 0.0
    n_classes: int = n_classes
    B: int = 8
    gate: bool = True
    subtyping: bool = False
    # 冻结后作为特征的聚合方式
    reduce: Literal["mean", "max", "concat"] = "mean"
    freeze: bool = True
    # 训练参数
    max_epochs: int = 100
    bag_weight: float = 0.7
    testing: bool = False
    weighted_sample: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-5
    output_dir: str = None
    opt: Literal["adam", "sgd"] = "adam"

    wsi_post_agg: List[str] = ("new_hybrid", "enhanced_hybrid")
    agg: str = "new_hybrid"

    @classmethod
    def from_args(cls, args):
        """从argparse.Namespace对象创建配置"""
        return cls(
            model_type=args.model_type,
            n_classes=args.n_classes,
            B=args.B,
            max_epochs=args.max_epochs,
            testing=args.testing,
            lr=args.lr,
            weight_decay=args.weight_decay,
            output_dir=args.output_dir
        )

@dataclass
class ViTWSIConfig:
    in_dim: int = 1024        # 输入 patch 特征维度
    out_dim: int = output_dim        # ViT 隐含维度
    depth: int = 8            # Transformer 层数
    n_heads: int = 8          # 注意力头数
    mlp_ratio: float = 4.0
    attn_drop: float = 0.1
    mlp_drop: float = 0.1
    droppath: float = 0.1     # Stochastic Depth
    dropout: float = 0.1      # token embedding + pos 的dropout
    token_dropout: float = 0.0 # 随机丢弃一部分 patch tokens（不含CLS），训练时生效
    n_classes: int = n_classes        # 若进行分类
    normalize_xy: bool = True # 是否对 (x,y) 进行每张 WSI 内的 min-max 归一化
    feature_extraction: bool = False
    # 仅支持 bs=1 & 无 padding；未来要多样本/有padding，可启用 key_padding_mask 并改 DataLoader

@dataclass
class TriFusionConfig:
    # 共同维度
    input_dim: int = output_dim      # 三模态对齐后的通道数（也是 cross-attn 的 embed_dim）
    num_heads: int = 4        # MultiheadAttention 的头数
    dropout: float = 0.1     # 分类头中的 dropout
    num_classes: int = n_classes      # 分类类别数
    use_confidence: bool = True
    aux_loss_weight: float = 0  # 0关闭辅助损失
    # 分类头
    classifier_hidden_ratio: float = 0.25  # tri_dim -> tri_dim*ratio
    # 四模态融合模块（models/FourModalFusionNetwork.CMAA）：门控跨模态注意力 + 逐模态打分
    cmaa_layers: int = 2
    cmaa_heads: int = 8
    cmaa_dropout: float = 0.1
    cmaa_score_hidden: int = 256

@dataclass
class DenoiserPriorConfig:
    """
    配置：DALL·E 2 Prior 风格的噪声预测器
    （FiLM + Self-Attn + Cross-Attn + Transformer block）
    """

    # =====================================================
    # 基础维度
    # =====================================================
    m: int = output_dim                 # x_t / x0 的维度（= 目标 embedding 768）
    cond_dim: int = output_dim          # 条件向量维度
    hidden: int = 1024                  # 主体 hidden（推荐大于 m，例如 1024）

    # =====================================================
    # 时间嵌入
    # =====================================================
    time_emb_dim: int = 256             # 时间 embedding 维度（prior 需要更大）
    time_emb_scale: float = 1.0
    time_emb_base: float = 10000.0
    time_emb_use_2pi: bool = True       # 推荐使用 2π 版本来增强周期性

    # =====================================================
    # 深度 Transformer 主干（Self-Attn）
    # =====================================================
    depth: int = 8                      # 主干总深度（DALL·E 2 = 8~12）
    n_heads: int = 16                   # Self-Attn 头数（hidden=1024 → head=16 正好）
    mlp_ratio: float = 4.0              # FFN 隐层放大倍数
    attn_dropout: float = 0.1
    dropout: float = 0.1
    droppath: float = 0.1               # StochasticDepth（0.1~0.2）

    # =====================================================
    # Cross-Attention（条件 → 目标）
    # =====================================================
    cross_depth: int = 4                # Cross-Attn 层数（可以比 self-attn 少一些）
    cross_heads: int = 8                # Cross-Attn 头数
    cond_refine_heads: int = 4          # 条件 tokens 的 self-attn（refine）

    # =====================================================
    # FiLM 调制（非常关键）
    # =====================================================
    film_dim: int = 512                 # cond → gamma/beta 的中间映射维度
    film_dropout: float = 0.0

    # =====================================================
    # Dropout & 正则
    # =====================================================
    layernorm_eps: float = 1e-6
    initializer_range: float = 0.02     # 权重初始化标准差

    # =====================================================
    # 类别条件（可选）
    # =====================================================
    cls_num: int = 3
    cls_emb_dim: int = output_dim                # 用类条件时的 embedding 维度

    # =====================================================
    # 功能函数（可选）
    # =====================================================
    @classmethod
    def from_args(cls, args) -> "DenoiserPriorConfig":
        d = vars(args)
        return cls(**{k: d[k] for k in d if k in cls.__annotations__})

    @classmethod
    def from_dict(cls, d: dict) -> "DenoiserPriorConfig":
        return cls(**{k: d[k] for k in d if k in cls.__annotations__})


@dataclass
class DiffusionConfig:
    m : int = output_dim
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"      # "cosine" | "linear"
    loss_type: str = "l2"         # "l1" | "l2"   (噪声回归)
    clip_x0: bool = True          # 采样/训练的 x0_hat/x0_gen 是否 clip 到 [-1,1]
    pred_type: str = "v"  # 预测目标，可选: "eps" 或 "v"
    # min-SNR（可选）
    snr_weighting: bool = True
    snr_gamma: float = 4.0        # 常用 3~5
    # 端到端可微 DDIM 的步数。fold1 对比 10/20/30 步效果相当（采样步数须与训练一致），10 步快约 2.7 倍
    train_ddim_steps: int = 10
    seq_len: int = 3    # ← 新增：默认序列长度（这里我们传 n_class）

    x0_stats_path: str = None


# ----------------------
# 训练配置
# ----------------------


@dataclass
class OverallModelConfig:
    # Stage-A 三模态 CMAA（CT / 冰冻 WSI / 表格：线性投影 -> 门控跨模态注意力 -> 逐模态打分加权）
    stage_a_cmaa_layers: int = 2
    stage_a_cmaa_heads: int = 8
    stage_a_cmaa_dropout: float = 0.1   # CMAA 注意力 dropout
    tri_dim: int = 768      # TriModalFusion 输出 D
    m: int = 768            # 扩散目标/生成向量维度
    num_classes: int = 3
    hidden: int = 512
    clf_hidden: int = 512
    clf_dropout: float = 0.1
    # 损失权重
    w_diff_stage2: float = 0.4     # 噪声回归
    w_rec_stage2: float = 2.0      # 端到端重建
    w_main_stage2: float = 1.5     # 生成特征分类损失
    w_main_stage3: float = 2.0     # 主分类损失
