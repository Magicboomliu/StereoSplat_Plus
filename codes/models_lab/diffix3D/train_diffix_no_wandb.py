import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

# ---- 你自己的模块 ----
from model import Difix, load_ckpt_from_state_dict, save_ckpt
from dataset import PairedDataset
from loss import gram_loss


# -----------------------
# Utils for robust cropping (避免越界 / 尺寸不足导致的随机裁剪报错)
# -----------------------
def _pad_to_min_hw(x: torch.Tensor, min_h: int, min_w: int, mode: str = "reflect"):
    """Pad last two dims (H,W) up to (min_h,min_w). Works for ...,H,W or NCHW."""
    H, W = x.shape[-2], x.shape[-1]
    pad_h = max(0, min_h - H)
    pad_w = max(0, min_w - W)
    if pad_h == 0 and pad_w == 0:
        return x
    # F.pad takes pads as (left, right, top, bottom) for 4 last dims
    return F.pad(
        x,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        mode=mode,
    )

def safe_random_crop(
    x: torch.Tensor,
    crop_h: int,
    crop_w: int,
    *,
    pad_if_needed=True,
    pad_mode="reflect",
):
    """
    Robust random crop on the last two dims (H,W). Returns cropped tensor and (top,left).
    - If pad_if_needed: pad to at least (crop_h,crop_w) then crop exactly that size.
    - Else: crop size becomes (min(H,crop_h), min(W,crop_w)).
    """
    if pad_if_needed:
        x = _pad_to_min_hw(x, crop_h, crop_w, mode=pad_mode)
        H, W = x.shape[-2], x.shape[-1]
        top = int(torch.randint(0, H - crop_h + 1, (1,), device=x.device).item())
        left = int(torch.randint(0, W - crop_w + 1, (1,), device=x.device).item())
        x = x[..., top : top + crop_h, left : left + crop_w]
        return x, (top, left)
    else:
        H, W = x.shape[-2], x.shape[-1]
        ch = min(crop_h, H)
        cw = min(crop_w, W)
        top = 0 if H == ch else int(torch.randint(0, H - ch + 1, (1,), device=x.device).item())
        left = 0 if W == cw else int(torch.randint(0, W - cw + 1, (1,), device=x.device).item())
        x = x[..., top : top + ch, left : left + cw]
        return x, (top, left)


def main(args):
    # tracker 设为 None：不启用任何日志后端
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # ----- 构建模型 -----
    net_difix = Difix(
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    net_difix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 评估/损失用到的网络
    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)

    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    # ----- 优化器参数 -----
    layers_to_opt = []
    # Unet
    layers_to_opt += list(net_difix.unet.parameters())
    # VAE LoRA（部分）
    for n, _p in net_difix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    # VAE Skip connections
    layers_to_opt = (
        layers_to_opt
        + list(net_difix.vae.decoder.skip_conv_1.parameters())
        + list(net_difix.vae.decoder.skip_conv_2.parameters())
        + list(net_difix.vae.decoder.skip_conv_3.parameters())
        + list(net_difix.vae.decoder.skip_conv_4.parameters())
    )

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ----- 数据集 -----
    dataset_train = PairedDataset(
        dataset_path=args.dataset_path,
        split="train",
        tokenizer=net_difix.tokenizer,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    dataset_val = PairedDataset(
        dataset_path=args.dataset_path,
        split="test",
        tokenizer=net_difix.tokenizer,
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # ----- Resume -----
    global_step = 0
    if args.resume is not None:
        if os.path.isdir(args.resume):
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(
                ckpt_files,
                key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")),
            )
            print("=" * 50)
            print(f"Loading checkpoint from {ckpt_files[-1]}")
            print("=" * 50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(net_difix, optimizer, ckpt_files[-1])
        elif args.resume.endswith(".pkl"):
            print("=" * 50)
            print(f"Loading checkpoint from {args.resume}")
            print("=" * 50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(net_difix, optimizer, args.resume)
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("=" * 50)
        print("Training from scratch")
        print("=" * 50)

    # ----- 精度 & 设备 -----
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_difix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)

    # Accelerator 准备
    net_difix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)

    # VGG 输入的标准化
    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # 进度条
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ----- 训练循环 -----
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_difix):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, V, C, H, W = x_src.shape

                # forward
                x_tgt_pred = net_difix(x_src, prompt_tokens=batch["input_ids"])

                # 视角展开到 batch
                x_tgt = rearrange(x_tgt, "b v c h w -> (b v) c h w")
                x_tgt_pred = rearrange(x_tgt_pred, "b v c h w -> (b v) c h w")

                # 重建损失
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lp = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lp

                # Gram 损失（warmup 之后）
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h, crop_w = 200, 200
                        x_tgt_pred_renorm, _ = safe_random_crop(
                            x_tgt_pred_renorm, crop_h, crop_w, pad_if_needed=True, pad_mode="reflect"
                        )

                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm, _ = safe_random_crop(
                            x_tgt_renorm, crop_h, crop_w, pad_if_needed=True, pad_mode="reflect"
                        )

                        loss_gram = gram_loss(
                            x_tgt_pred_renorm.to(weight_dtype),
                            x_tgt_renorm.to(weight_dtype),
                            net_vgg,
                        ) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0, dtype=weight_dtype, device=accelerator.device)

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # 还原回 [B,V,...]
                x_tgt = rearrange(x_tgt, "(b v) c h w -> b v c h w", v=V)
                x_tgt_pred = rearrange(x_tgt_pred, "(b v) c h w -> b v c h w", v=V)

            # 每步结束（若发生了同步优化）
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # 更新进度条 postfix（仅显示，不写日志）
                    pb_logs = {
                        "loss_l2": float(loss_l2.detach().item()),
                        "loss_lpips": float(loss_lp.detach().item()),
                    }
                    if args.lambda_gram > 0:
                        pb_logs["loss_gram"] = float(loss_gram.detach().item())
                    progress_bar.set_postfix(**pb_logs)

                    # Checkpoint
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(
                            args.output_dir, "checkpoints", f"model_{global_step}.pkl"
                        )
                        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, outf)

                    # 简单评估：只计算标量并 print，不写任何日志文件
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
                        for step_v, batch_val in enumerate(dl_val):
                            if step_v >= args.num_samples_eval:
                                break
                            x_src_v = batch_val["conditioning_pixel_values"].to(
                                accelerator.device, dtype=weight_dtype
                            )
                            x_tgt_v = batch_val["output_pixel_values"].to(
                                accelerator.device, dtype=weight_dtype
                            )
                            Bv, Vv, Cv, Hv, Wv = x_src_v.shape
                            assert Bv == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                x_tgt_pred_v = accelerator.unwrap_model(net_difix)(
                                    x_src_v, prompt_tokens=batch_val["input_ids"].to(accelerator.device)
                                )
                                # 用第一个视角
                                x_tgt_v = x_tgt_v[:, 0]
                                x_tgt_pred_v = x_tgt_pred_v[:, 0]
                                loss_l2_v = F.mse_loss(
                                    x_tgt_pred_v.float(), x_tgt_v.float(), reduction="mean"
                                )
                                loss_lp_v = net_lpips(
                                    x_tgt_pred_v.float(), x_tgt_v.float()
                                ).mean()

                                l_l2.append(loss_l2_v.item())
                                l_lpips.append(loss_lp_v.item())

                        val_l2 = float(np.mean(l_l2)) if l_l2 else float("nan")
                        val_lp = float(np.mean(l_lpips)) if l_lpips else float("nan")
                        print(f"[Eval @ step {global_step}] L2={val_l2:.6f}, LPIPS={val_lp:.6f}")
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # loss
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # eval
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for evaluation")
    parser.add_argument("--viz_freq", type=int, default=100, help="(已禁用可视化) 保留占位，不再使用。")

    # trackers（已禁用日志，这两个参数保留占位但不会使用）
    parser.add_argument("--tracker_project_name", type=str, default="difix")
    parser.add_argument("--tracker_run_name", type=str, default="no_log_run")

    # model
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")

    # training
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere GPUs to speed up training.",
    )

    # 日志后端参数已移除，Accelerator 固定为 log_with=None

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers."
    )
    parser.add_argument("--set_grads_to_none", action="store_true")

    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()
    main(args)
