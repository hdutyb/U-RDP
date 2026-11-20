# -*- coding: utf-8 -*-
"""
数据与预训练权重的路径，集中在此修改（各训练脚本的命令行参数默认值都取自这里，运行时可用对应参数覆盖）。

checkpoint 参数约定（见 utils.pipeline.resolve_fold_ckpt）：
    - 可写成含 {fold_id} 的模板，或某次 k 折运行的根目录（自动取 fold_k/checkpoints 下 val 最优权重）；
    - 写成具体 .pth 文件时，所有折都加载同一个文件。
"""

# ---------------- 数据 ----------------
EXCEL = ""                # 病例表 .xlsx（标签 / 表格特征 / 冰冻切片号 / 术后病理前缀 / CT 路径）
EXCEL_CT = EXCEL          # CT 编码器预训练使用的病例表
CT_NPY_DIR = ""           # CT 体数据 .npy 目录
INTRA_WSI_DIR = ""        # 冰冻 WSI：TITAN slide 特征 .h5
POST_WSI_ROOT = ""        # 术后 WSI：TITAN slide 特征 .h5（每例多张）

LOG_ROOT = "./log"        # 训练输出根目录

# ---------------- 各阶段默认输入（通常由命令行参数给出） ----------------
CT_PRETRAINED = None      # stage0 CT 编码器（train.py ct）
TAB_PRETRAINED = None     # stage0 表格编码器（train.py table）
AGG_PRETRAINED = None     # stage0 术后 WSI 聚合器（train.py wsi_post）
X0_STATS_ROOT = None      # 扩散目标 x0 的按折 mean/std（fold_k/x0_mean_std.json）
STAGE1_RUN = None         # Stage-1 运行目录
STAGE1_MASK_FILE = None   # Stage-1 保存的 fixed_missing_masks.pt
STAGE2_RUN = None         # Stage-2 运行目录
