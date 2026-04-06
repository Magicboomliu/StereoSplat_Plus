import argparse
import json
import os
from typing import Dict, Tuple

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from dataset_pose_cond import KITTI360_Restoration_Dataset
from ssim_torch import ssim as ssim_torch


def read_json_file(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_split(payload: Dict, split: str) -> Dict:
    key = split
    if key == "train" and "train" not in payload and "training" in payload:
        key = "training"
    elif key == "test" and "test" not in payload and "validation" in payload:
        key = "validation"
    if key not in payload:
        raise KeyError(f"Split '{split}' not found. Available keys: {list(payload.keys())}")
    return payload[key]


def get_paths_from_item(item: Dict) -> Tuple[str, str, str]:
    image_path = item.get("image", item.get("input_image"))
    target_path = item.get("target_image", item.get("output_image"))
    ref_path = item.get("ref_image", None)
    if image_path is None or target_path is None:
        raise KeyError("Each sample must contain image/input_image and target_image/output_image")
    return image_path, target_path, ref_path


def pil_to_neg1_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def psnr_from_mse_neg1_to1(mse: torch.Tensor, eps: float = 1e-10) -> float:
    return float(10.0 * np.log10(4.0 / (float(mse.item()) + eps)))


def compute_psnr_ssim(pred: Image.Image, gt: Image.Image) -> Tuple[float, float]:
    pred_np = np.array(pred.convert("RGB"), dtype=np.float32) / 255.0
    gt_np = np.array(gt.convert("RGB"), dtype=np.float32) / 255.0
    if pred_np.shape != gt_np.shape:
        raise ValueError(f"Shape mismatch: pred={pred_np.shape}, gt={gt_np.shape}")
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_test_filename", type=str, required=True, help="Path to dataset json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split to evaluate")
    parser.add_argument("--height", type=int, default=112, help="Train-like input height")
    parser.add_argument("--width", type=int, default=544, help="Train-like input width")
    parser.add_argument("--ablation_study_name", type=str, default="baseline_no_model", help="Run tag")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_folder", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--output_json_name", type=str, default="baseline_metrics.json", help="Single output JSON filename")
    parser.add_argument("--max_samples", type=int, default=-1, help="Evaluate first N samples, -1 means all")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="original_res",
        choices=["train_like", "original_res"],
        help="train_like: match training preprocessing; original_res: compute metrics on original images",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_net = lpips.LPIPS(net="vgg").to(device).eval()

    results = []
    l_psnr, l_ssim, l_lpips = [], [], []

    if args.eval_mode == "train_like":
        dataset = KITTI360_Restoration_Dataset(
            dataset_path=args.input_test_filename,
            split=args.split,
            tokenizer=None,
            height=args.height,
            width=args.width,
            use_relative_pose=False,
        )
        if args.max_samples > 0:
            dataset.img_ids = dataset.img_ids[: args.max_samples]
        dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for batch_idx, batch in enumerate(tqdm(dl, desc=f"Baseline {args.split} (train_like)")):
            x_src = batch["conditioning_pixel_values"].to(device)  # [B,V,C,H,W]
            x_tgt = batch["output_pixel_values"].to(device)

            # 与训练验证一致：只评估第0视图
            x_src0 = x_src[:, 0]
            x_tgt0 = x_tgt[:, 0]

            with torch.no_grad():
                mse = torch.mean((x_src0.float() - x_tgt0.float()) ** 2)
                psnr = psnr_from_mse_neg1_to1(mse)
                ssim = float(
                    ssim_torch(
                        torch.clamp(x_src0.float() * 0.5 + 0.5, 0.0, 1.0),
                        torch.clamp(x_tgt0.float() * 0.5 + 0.5, 0.0, 1.0),
                        data_range=1.0,
                        size_average=True,
                    ).item()
                )
                lpips_val = float(lpips_net(x_src0.float(), x_tgt0.float()).mean().item())

            sample_id = dataset.img_ids[batch_idx]
            image_path, target_path, ref_path = get_paths_from_item(dataset.data[sample_id])
            l_psnr.append(psnr)
            l_ssim.append(ssim)
            l_lpips.append(lpips_val)
            results.append(
                {
                    "id": str(sample_id),
                    "image": image_path,
                    "target_image": target_path,
                    "ref_image": ref_path,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_val,
                }
            )
    else:
        payload = read_json_file(args.input_test_filename)
        dataset_files = resolve_split(payload, args.split)
        items = list(dataset_files.items())
        if args.max_samples > 0:
            items = items[: args.max_samples]

        for data_id, data_item in tqdm(items, desc=f"Baseline {args.split} (original_res)"):
            image_path, target_path, ref_path = get_paths_from_item(data_item)
            input_img = Image.open(image_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")

            # baseline: pred == input
            pred_img = input_img

            psnr, ssim = compute_psnr_ssim(pred_img, target_img)
            pred_t = pil_to_neg1_tensor(pred_img, device)
            gt_t = pil_to_neg1_tensor(target_img, device)
            with torch.no_grad():
                lpips_val = float(lpips_net(pred_t, gt_t).mean().item())

            l_psnr.append(psnr)
            l_ssim.append(ssim)
            l_lpips.append(lpips_val)
            results.append(
                {
                    "id": str(data_id),
                    "image": image_path,
                    "target_image": target_path,
                    "ref_image": ref_path,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_val,
                }
            )

    summary = {
        "ablation_study_name": args.ablation_study_name,
        "dataset": args.input_test_filename,
        "split": args.split,
        "num_samples": len(results),
        "eval_mode": args.eval_mode,
        "height": args.height,
        "width": args.width,
        "mean_psnr": float(np.mean(l_psnr)) if l_psnr else None,
        "mean_ssim": float(np.mean(l_ssim)) if l_ssim else None,
        "mean_lpips": float(np.mean(l_lpips)) if l_lpips else None,
    }

    out_payload = {"summary": summary, "per_sample": results}
    out_path = os.path.join(args.output_folder, args.output_json_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Samples: {summary['num_samples']}")
    print(f"Mean PSNR : {summary['mean_psnr']:.4f}")
    print(f"Mean SSIM : {summary['mean_ssim']:.6f}")
    print(f"Mean LPIPS: {summary['mean_lpips']:.6f}")
    print(f"Saved to  : {out_path}")
    print("=" * 60)

