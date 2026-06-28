import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from difix3d.pipelines.difix_ref_vanilla_pipeline import DifixRefVanillaPipeline

def _load_payload_split(payload: Dict[str, Any], split: str) -> Dict[str, Any]:
    # 兼容不同数据集 JSON 的 split 命名（与 difix3d.dataset.PairedDataset 保持一致）
    key = split
    if key == "train" and "train" not in payload and "training" in payload:
        key = "training"
    elif key == "test" and "test" not in payload and "validation" in payload:
        key = "validation"
    if key not in payload:
        raise KeyError(f"Split '{split}' not found in json. Available keys: {list(payload.keys())}")
    return payload[key]

def _pil_to_t01(img: Image.Image, *, height: int, width: int, device: torch.device) -> torch.Tensor:
    """把 PIL 图像 resize 到评测分辨率，并转成 (3,H,W) 的 [0,1] float tensor（在 device 上）。"""
    img = img.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t.to(device=device)

def _psnr_t01(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> float:
    # PSNR: 10 * log10( MAX^2 / MSE )，这里输入已是 [0,1]，所以 MAX=1
    mse = torch.mean((x - y) ** 2).item()
    return float(10.0 * np.log10(1.0 / (mse + eps)))


@torch.no_grad()
def _ssim_t01(x: torch.Tensor, y: torch.Tensor) -> float:
    # SSIM：复用 difix3d.utils.ssim_torch（要求输入是 [0,1]）
    from difix3d.utils.ssim_torch import ssim as ssim_torch

    return float(ssim_torch(x.unsqueeze(0), y.unsqueeze(0), data_range=1.0, size_average=True).item())


@dataclass
class RunningMean:
    # 简单的 running mean 统计器：用于实时显示 “到当前为止” 的均值
    s: float = 0.0
    n: int = 0

    def update(self, v: float) -> None:
        self.s += float(v)
        self.n += 1

    @property
    def mean(self) -> float:
        return self.s / max(1, self.n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str, help="JSON with image/target_image/(ref_image)/prompt")
    parser.add_argument("--split", default="test", choices=["train", "test"], type=str)
    parser.add_argument("--pretrained_name", default="nvidia/difix_ref", type=str)
    parser.add_argument("--pretrained_path", default=None, type=str, help="Finetuned .pkl checkpoint path")
    parser.add_argument("--height", default=112, type=int)
    parser.add_argument("--width", default=544, type=int)
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional random seed. Use this if you want deterministic eval runs while keeping training-aligned stochastic ops enabled.",
    )
    parser.add_argument(
        "--deterministic_vae_encode",
        action="store_true",
        help="If set, use VAE latent_dist.mode() instead of sampling. Default False to match training.",
    )
    parser.add_argument(
        "--deterministic_scheduler_step",
        action="store_true",
        help="If set, use mean-only deterministic DDPM step (no variance noise). Default False to match training.",
    )
    parser.add_argument("--max_samples", default=-1, type=int, help="<=0 means evaluate all")
    parser.add_argument("--save_json", default=None, type=str, help="Optional path to save summary json")
    parser.add_argument("--save_per_sample", action="store_true", help="Also dump per-sample metrics into json")
    args = parser.parse_args()

    # 选择 device；如果用户指定 cuda 但不可用则自动回退 cpu
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        # 说明：这里的 seed 仅用于“让 eval 更可复现”，但默认仍保持训练同款的随机路径
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # 读取 JSON 并选择 split（train/test）
    with open(args.dataset_path, "r") as f:
        payload = json.load(f)
    split_data = _load_payload_split(payload, args.split)
    img_ids = list(split_data.keys())
    if args.max_samples and args.max_samples > 0:
        img_ids = img_ids[: args.max_samples]

    # 加载 DifixRef pipeline（可指定 finetuned .pkl）；两个 deterministic flag 默认 False，用于严格对齐训练
    pipe = DifixRefVanillaPipeline.from_pretrained(
        pretrained_name=args.pretrained_name,
        pretrained_path=args.pretrained_path,
        timestep=199,
        deterministic_vae_encode=args.deterministic_vae_encode,
        deterministic_scheduler_step=args.deterministic_scheduler_step,
    )
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()

    # LPIPS：使用 VGG backbone；这里用的是 [-1,1] 的输入域
    import lpips as lpips_lib

    lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()

    # before = conditioning(image) vs target
    m_before_psnr = RunningMean()
    m_before_ssim = RunningMean()
    m_before_lpips = RunningMean()

    # after = pipeline output vs target
    m_after_psnr = RunningMean()
    m_after_ssim = RunningMean()
    m_after_lpips = RunningMean()

    per_sample: Dict[str, Any] = {}

    for i, img_id in enumerate(tqdm(img_ids, desc=f"Evaluating ({args.split})")):
        rec = split_data[img_id]
        # JSON 字段约定：image=conditioning, target_image=GT, ref_image(可选), prompt(可选)
        inp_path = rec["image"]
        tgt_path = rec["target_image"]
        ref_path = rec.get("ref_image", None)
        prompt = rec.get("prompt", "")

        if not os.path.exists(inp_path):
            raise FileNotFoundError(inp_path)
        if not os.path.exists(tgt_path):
            raise FileNotFoundError(tgt_path)
        if ref_path is not None and not os.path.exists(ref_path):
            raise FileNotFoundError(ref_path)

        inp = Image.open(inp_path)
        tgt = Image.open(tgt_path)
        ref = Image.open(ref_path) if ref_path is not None else None

        # 模型推理：输出一张 PIL（内部会 resize 到 height/width）
        out = pipe(
            inp,
            ref_image=ref,
            prompt=prompt,
            height=args.height,
            width=args.width,
            output_type="pil",
            return_dict=True,
        ).images[0]

        # 指标统一在模型分辨率 (height/width) 上计算，避免不同原图分辨率带来偏差
        x_in = _pil_to_t01(inp, height=args.height, width=args.width, device=device)
        x_hat = _pil_to_t01(out, height=args.height, width=args.width, device=device)
        x_gt = _pil_to_t01(tgt, height=args.height, width=args.width, device=device)

        # before/after 的 PSNR、SSIM
        before_psnr = _psnr_t01(x_in, x_gt)
        before_ssim = _ssim_t01(x_in, x_gt)

        after_psnr = _psnr_t01(x_hat, x_gt)
        after_ssim = _ssim_t01(x_hat, x_gt)

        # LPIPS：要求输入是 [-1,1]，所以把 [0,1] 线性映射过去
        x_gt_n11 = x_gt.unsqueeze(0) * 2.0 - 1.0
        x_in_n11 = x_in.unsqueeze(0) * 2.0 - 1.0
        x_hat_n11 = x_hat.unsqueeze(0) * 2.0 - 1.0
        before_lp = float(lpips_fn(x_in_n11, x_gt_n11).mean().item())
        after_lp = float(lpips_fn(x_hat_n11, x_gt_n11).mean().item())

        # 更新 running mean
        m_before_psnr.update(before_psnr)
        m_before_ssim.update(before_ssim)
        m_before_lpips.update(before_lp)

        m_after_psnr.update(after_psnr)
        m_after_ssim.update(after_ssim)
        m_after_lpips.update(after_lp)
        
        if (i + 1) % 10 == 0:
            # 每 10 个样本打印一次 “到当前为止”的平均指标，便于边跑边看
            print(
                json.dumps(
                    {
                        "progress": f"{i + 1}/{len(img_ids)}",
                        "before": {
                            "psnr_mean": m_before_psnr.mean,
                            "ssim_mean": m_before_ssim.mean,
                            "lpips_mean": m_before_lpips.mean,
                        },
                        "after": {
                            "psnr_mean": m_after_psnr.mean,
                            "ssim_mean": m_after_ssim.mean,
                            "lpips_mean": m_after_lpips.mean,
                        },
                    },
                    ensure_ascii=False,
                )
            )

        if args.save_per_sample:
            # 可选：把每个样本的 before/after 指标都存到 json，方便后续按 degradation level 做分析
            per_sample[img_id] = {
                "before": {"psnr": before_psnr, "ssim": before_ssim, "lpips": before_lp},
                "after": {"psnr": after_psnr, "ssim": after_ssim, "lpips": after_lp},
                "image": inp_path,
                "target_image": tgt_path,
                "ref_image": ref_path,
            }

    summary: Dict[str, Any] = {
        "split": args.split,
        "num_samples": len(img_ids),
        "height": args.height,
        "width": args.width,
        "pretrained_name": args.pretrained_name,
        "pretrained_path": args.pretrained_path,
        "before": {
            "psnr_mean": m_before_psnr.mean,
            "ssim_mean": m_before_ssim.mean,
            "lpips_mean": m_before_lpips.mean,
        },
        "after": {
            "psnr_mean": m_after_psnr.mean,
            "ssim_mean": m_after_ssim.mean,
            "lpips_mean": m_after_lpips.mean,
        },
    }
    if args.save_per_sample:
        summary["per_sample"] = per_sample

    # 最终打印汇总 JSON（也可 --save_json 落盘）
    print(json.dumps(summary, indent=2))

    if args.save_json is not None:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

