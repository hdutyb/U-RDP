# -*- coding: utf-8 -*-
"""
统一训练入口：python train.py <stage> [该阶段的参数...]

训练流程（按顺序，每折 train/test = 4:1，按 test 折 acc 取最优 epoch）：
  stage 0  单模态预训练
      ct            CT 编码器（3D ResNet18）
      table         表格编码器
      wsi_post      术后 WSI 多 slide 聚合器（PAM）
  stage 1  stage1   CT + 冰冻 WSI + 表格 -> 三模态条件编码器（CMAA）
  教师     fourmodal  四模态 CMAA（使用真实术后 WSI），作为 stage2 的蒸馏教师与 stage3 的初始化
  stage 2  anchor   在冻结的三模态条件编码器输出上训练 Anchor Predictor
           stage2   残差扩散（CPD）生成伪术后病理特征
  stage 3  stage3   CT / 冰冻 WSI / 表格特征 + 伪术后病理特征 -> 四模态 CMAA 融合分类

    python train.py stage3 -h                 # 查看某阶段的全部参数
"""
import argparse
import importlib
import sys

# stage 名 -> (模块, add_args 额外参数, 说明)
STAGES = {
    "ct":          ("trainers.encoder", {"modality": "ct"}, "stage0: CT 编码器预训练"),
    "table":       ("trainers.encoder", {"modality": "table"}, "stage0: 表格编码器预训练"),
    "wsi_post":    ("trainers.wsi_post", {}, "stage0: 术后 WSI 聚合器（扩散目标）"),
    "stage1":      ("trainers.stage1", {}, "stage1: 三模态条件编码器"),
    "fourmodal":   ("trainers.fusion_fourmodal", {}, "教师: 四模态融合（真实术后 WSI）"),
    "anchor":      ("trainers.anchor", {}, "stage2 前置: 训练 Anchor Predictor"),
    "stage2":      ("trainers.stage2", {}, "stage2: 残差扩散生成伪术后病理特征"),
    "stage3":      ("trainers.stage3", {}, "stage3: 四模态 CMAA 融合分类"),
}


def add_global_args(p):
    g = p.add_argument_group("全局（所有阶段通用）")
    g.add_argument("--train_seed", type=int, default=None,
                   help="只控制训练随机性（初始化 / 打乱 / dropout）的种子；数据划分仍由 --seed 决定。默认同 --seed")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in STAGES:
        if argv and argv[0] not in ("-h", "--help"):
            print(f"未知 stage: {argv[0]}\n")
        print(__doc__)
        sys.exit(0 if not argv or argv[0] in ("-h", "--help") else 2)

    stage, rest = argv[0], argv[1:]
    module_name, extra, desc = STAGES[stage]
    module = importlib.import_module(module_name)   # 只导入所选阶段，避免加载全部模型

    parser = argparse.ArgumentParser(prog=f"train.py {stage}", description=f"{desc}\n\n{module.__doc__ or ''}",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    module.add_args(parser, **extra)
    add_global_args(parser)
    args = parser.parse_args(rest)
    print(f"[train.py] stage={stage} args={vars(args)}")
    module.run(args)


if __name__ == "__main__":
    main()
