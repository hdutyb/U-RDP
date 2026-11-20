# -*- coding: utf-8 -*-
"""
Stage-2 的教师蒸馏损失 L_kd（论文 Eq. kd）。

教师 = 冻结的四模态 CMAA（train.py fourmodal 的结果，使用真实术后 WSI 训练）。
只把教师的术后病理输入换成生成的 x0_gen，ct / intra / tab 用教师自己的编码器（冻结、预先缓存），
梯度经冻结教师的融合模块回传到扩散模型：
    kd : T² · KL( softmax(teacher_real/T) ‖ softmax(teacher_gen/T) )
验证时另算 teacher_gen_acc（教师 + 生成术后特征），--select_metric teacher 时用它选 epoch。
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.configs import CTEncoderConfig, TriFusionConfig
from models.FourModalFusionNetwork import CMAA
from utils.missing_modality import IndexedSubset
from utils.pipeline import autocast_ctx, make_collate_fn, make_table_cfg


def add_kd_args(p):
    g = p.add_argument_group("Stage-2 教师蒸馏")
    g.add_argument("--teacher_ckpt", default=None,
                   help="四模态教师（train.py fourmodal 结果，写法同 --stage1_ckpt，如 <run>/fourmodal/fold{fold_id}）；"
                        "不给则不使用蒸馏损失")
    g.add_argument("--teacher_layers", type=int, default=2, help="教师 CMAA 层数（需与教师训练时一致）")
    g.add_argument("--teacher_heads", type=int, default=8, help="教师 CMAA 头数")
    g.add_argument("--teacher_agg", default="new_hybrid", help="教师的术后 WSI 聚合器结构")
    g.add_argument("--w_kd", type=float, default=1.0, help="输出蒸馏 KL 权重")
    g.add_argument("--kd_T", type=float, default=2.0, help="蒸馏温度")
    g.add_argument("--select_metric", choices=["gen_cls", "teacher"], default="teacher",
                   help="按哪个分类器的 val acc 选 epoch：teacher = 冻结教师 + 生成术后特征；gen_cls = 生成特征分类头")


def kd_enabled(args) -> bool:
    return bool(getattr(args, "teacher_ckpt", None))


class FrozenTeacher:
    """冻结的四模态教师 + 每个样本 ct / intra / tab 特征与真实术后输出的缓存（按原始样本下标）。"""

    def __init__(self, args, ckpt_path: str, device: str):
        self.device = device
        self.model = CMAA(ct_config=CTEncoderConfig(feature_extraction=True), table_config=make_table_cfg(),
                          fusion_cfg=TriFusionConfig(cmaa_layers=args.teacher_layers, cmaa_heads=args.teacher_heads),
                          wsi_agg=args.teacher_agg).to(device)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"[Teacher] loaded (strict) {ckpt_path}")
        self.row = {}

    @torch.no_grad()
    def build_cache(self, dataset, indices, num_workers: int = 4):
        """用 fourmodal 训练时的数据处理（CT 不裁剪）预先编码所有样本：ct / intra / tab 与教师真实输出。"""
        loader = DataLoader(IndexedSubset(dataset, list(indices)), batch_size=1, shuffle=False, num_workers=num_workers,
                            collate_fn=make_collate_fn(ct_crop_h=None))
        ct, intra, tab, logit, y, keys = [], [], [], [], [], []
        m = self.model
        with autocast_ctx(self.device):
            for b in tqdm(loader, desc="[Teacher] cache"):
                c = m.CTEncoder(b["CT"].to(self.device))
                t = m.TableEncoder(b["x_num"].to(self.device),
                                   b["x_cat"].to(self.device) if b["x_cat"] is not None else None)
                i = b["intra_wsi"].to(self.device)
                pw = m._agg_post([[x.to(self.device) for x in case] for case in b["post_wsi_list"]])
                out = m.fusion(c, i, pw, t)
                ct.append(c.float()); intra.append(i.float()); tab.append(t.float())
                logit.append(out["logits"].float())
                y.append(b["y"].argmax(-1)); keys += [int(k) for k in b["__sample_index__"]]
        self.ct, self.intra, self.tab = torch.cat(ct), torch.cat(intra), torch.cat(tab)
        self.logit_real = torch.cat(logit)
        self.y = torch.cat(y).to(self.device)
        self.row = {k: r for r, k in enumerate(keys)}
        acc = (self.logit_real.argmax(-1) == self.y).float().mean().item()
        print(f"[Teacher] cached {len(keys)} samples; teacher(real post) acc on these = {acc:.4f}")

    def rows(self, sample_index) -> torch.Tensor:
        return torch.tensor([self.row[int(k)] for k in sample_index], device=self.device)

    def forward_gen(self, r: torch.Tensor, x0_gen: torch.Tensor):
        """把 x0_gen 当作术后 WSI 特征送进冻结教师的融合模块（梯度回传到 x0_gen）。"""
        return self.model.fusion(self.ct[r], self.intra[r], x0_gen, self.tab[r])["logits"]


def kd_losses(args, teacher: FrozenTeacher, out: dict, sample_index) -> dict:
    """L_kd（float32 标量）。"""
    r = teacher.rows(sample_index)
    lg = teacher.forward_gen(r, out["x0_gen"].float()).float()
    T = args.kd_T
    kd = (T * T) * F.kl_div(F.log_softmax(lg / T, -1), F.softmax(teacher.logit_real[r] / T, -1),
                            reduction="batchmean")
    return {"kd": args.w_kd * kd}
