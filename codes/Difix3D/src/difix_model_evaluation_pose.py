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
from model_ref_with_pose import DifixRefWithPose
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
    if ref_path is None:
        raise KeyError("Pose evaluation requires ref_image for every sample.")
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


def build_model(args) -> DifixRefWithPose:
    if args.use_model_type not in {"huggingface", "local"}:
        raise ValueError("--use_model_type must be 'huggingface' or 'local'")

    if args.use_model_type == "huggingface":
        model_name = args.model_name
        model_path = None
    else:
        model_name = args.model_name if args.model_name else "nvidia/difix_ref"
        model_path = args.model_path

    model = DifixRefWithPose(
        pretrained_name=model_name,
        pretrained_path=model_path,
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
        deterministic_vae_encode=not args.stochastic_forward,
        deterministic_scheduler_step=not args.stochastic_forward,
    )
    model.set_eval()
    return model


def ensure_pose_views(relative_pose: np.ndarray, has_ref: bool) -> np.ndarray:
    """
    训练数据侧约定：
    - 若有 ref_image：pose 需要 [2,4,4]，第 0 个为主视图 relative_pose，第 1 个为 I（ref视图）
    - 若无 ref_image：pose [1,4,4]
    """
    if relative_pose.shape == (4, 4):
        relative_pose = relative_pose[None, ...]  # [1,4,4]
    if relative_pose.ndim != 3 or relative_pose.shape[-2:] != (4, 4):
        raise ValueError(f"relative_pose must be [4,4] or [V,4,4], got {relative_pose.shape}")
    if has_ref:
        if relative_pose.shape[0] == 1:
            relative_pose = np.concatenate([relative_pose, np.eye(4, dtype=np.float32)[None, ...]], axis=0)
        if relative_pose.shape[0] != 2:
            raise ValueError(f"Expected pose views=2 (with ref), got {relative_pose.shape[0]}")
    else:
        if relative_pose.shape[0] != 1:
            raise ValueError(f"Expected pose views=1 (no ref), got {relative_pose.shape[0]}")
    return relative_pose.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_test_filename", type=str, required=True, help="Path to dataset json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split to evaluate")
    parser.add_argument("--model_name", type=str, default="nvidia/difix_ref", help="HF model name")
    parser.add_argument("--model_path", type=str, default=None, help="Local checkpoint .pkl (for --use_model_type local)")
    parser.add_argument("--height", type=int, default=112, help="Model input height")
    parser.add_argument("--width", type=int, default=544, help="Model input width")
    parser.add_argument("--ablation_study_name", type=str, default="eval_pose_run", help="Run tag")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timestep", type=int, default=199, help="Diffusion timestep")
    parser.add_argument("--lora_rank_vae", type=int, default=4, help="Match training")
    parser.add_argument("--mv_unet", action="store_true", help="Match training")
    parser.add_argument("--stochastic_forward", action="store_true", help="Default deterministic forward")
    parser.add_argument("--use_model_type", type=str, default="local", help="huggingface or local")
    parser.add_argument("--output_folder", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--output_json_name", type=str, default="metrics_pose.json", help="Single output JSON filename")
    parser.add_argument("--save_predictions", action="store_true", help="Save per-sample model outputs")
    parser.add_argument("--max_samples", type=int, default=-1, help="Evaluate first N samples, -1 means all")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="train_like",
        choices=["train_like", "original_res"],
        help="train_like: strict match to training validation pipeline; original_res: sample_with_pose()+PIL metrics",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    pred_dir = os.path.join(args.output_folder, "predictions")
    if args.save_predictions:
        os.makedirs(pred_dir, exist_ok=True)

    model = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_net = lpips.LPIPS(net="vgg").to(device).eval()

    results = []
    l_psnr, l_ssim, l_lpips = [], [], []

    if args.eval_mode == "train_like":
        dataset = KITTI360_Restoration_Dataset(
            dataset_path=args.input_test_filename,
            split=args.split,
            tokenizer=model.tokenizer,
            height=args.height,
            width=args.width,
            use_relative_pose=True,
        )
        if args.max_samples > 0:
            dataset.img_ids = dataset.img_ids[: args.max_samples]
        dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for batch_idx, batch in enumerate(tqdm(dl, desc=f"Evaluating {args.split} (pose train_like)")):
            x_src = batch["conditioning_pixel_values"].to(device)
            x_tgt = batch["output_pixel_values"].to(device)
            prompt_tokens = batch["input_ids"].to(device)
            rel_pose = batch.get("relative_pose", None)
            if rel_pose is None:
                raise KeyError("Dataset missing relative_pose, but pose evaluation requires it.")
            rel_pose = rel_pose.to(device=device, dtype=torch.float32)  # [B,V,4,4]

            with torch.no_grad():
                x_pred = model(x_src, rel_pose, prompt_tokens=prompt_tokens)
                x_pred0 = x_pred[:, 0]
                x_tgt0 = x_tgt[:, 0]
                mse = torch.mean((x_pred0.float() - x_tgt0.float()) ** 2)
                psnr = psnr_from_mse_neg1_to1(mse)
                ssim = float(
                    ssim_torch(
                        torch.clamp(x_pred0.float() * 0.5 + 0.5, 0.0, 1.0),
                        torch.clamp(x_tgt0.float() * 0.5 + 0.5, 0.0, 1.0),
                        data_range=1.0,
                        size_average=True,
                    ).item()
                )
                lpips_val = float(lpips_net(x_pred0.float(), x_tgt0.float()).mean().item())

            sample_id = dataset.img_ids[batch_idx]
            image_path, target_path, ref_path = get_paths_from_item(dataset.data[sample_id])
            if args.save_predictions:
                pred01 = torch.clamp(x_pred0.float().squeeze(0) * 0.5 + 0.5, 0.0, 1.0)
                pred_u8 = (pred01.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                Image.fromarray(pred_u8).save(os.path.join(pred_dir, f"{sample_id}.png"))
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

        for data_id, data_item in tqdm(items, desc=f"Evaluating {args.split} (pose original_res)"):
            image_path, target_path, ref_path = get_paths_from_item(data_item)
            prompt = data_item.get("prompt", "")
            rel_pose = data_item.get("relative_pose", None)
            if rel_pose is None:
                raise KeyError(f"Sample {data_id} missing relative_pose, but pose evaluation requires it.")

            input_img = Image.open(image_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
            ref_img = Image.open(ref_path).convert("RGB")

            rel_pose_np = ensure_pose_views(np.array(rel_pose), has_ref=True)  # [2,4,4]
            with torch.no_grad():
                pred_img = model.sample_with_pose(
                    image=input_img,
                    width=args.width,
                    height=args.height,
                    relative_pose=rel_pose_np,
                    ref_image=ref_img,
                    timesteps=None,
                    prompt=prompt,
                    prompt_tokens=None,
                )

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

            if args.save_predictions:
                pred_img.save(os.path.join(pred_dir, f"{data_id}.png"))

    summary = {
        "ablation_study_name": args.ablation_study_name,
        "dataset": args.input_test_filename,
        "split": args.split,
        "num_samples": len(results),
        "model_name": args.model_name,
        "model_path": args.model_path,
        "use_model_type": args.use_model_type,
        "use_ref": True,
        "use_relative_pose": True,
        "eval_mode": args.eval_mode,
        "height": args.height,
        "width": args.width,
        "timestep": args.timestep,
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

