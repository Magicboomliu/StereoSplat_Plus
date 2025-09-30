#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import transformers
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from model import Difix, load_ckpt_from_state_dict, save_ckpt
from dataset import PairedDataset
from loss import gram_loss


# ------------------------- helpers -------------------------
def charbonnier(x, eps: float = 1e-6):
    # robust L1
    return torch.sqrt(x * x + eps)

@torch.no_grad()
def update_ema(src_model: nn.Module, ema_model: nn.Module, decay: float = 0.999):
    for p, p_ema in zip(src_model.parameters(), ema_model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
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

    # ---------------- models ----------------
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
            raise ValueError("xformers is not available, please `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # LPIPS (FP32 + eval)
    net_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    # VGG features for Gram (FP32 + eval)
    from torchvision.models import vgg16
    net_vgg = vgg16(pretrained=True).features.to(accelerator.device)
    for p in net_vgg.parameters():
        p.requires_grad_(False)
    net_vgg.eval()

    # optimizer only for trainable parts
    layers_to_opt = []
    layers_to_opt += list(net_difix.unet.parameters())
    for n, p in net_difix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert p.requires_grad
            layers_to_opt.append(p)
    layers_to_opt += list(net_difix.vae.decoder.skip_conv_1.parameters())
    layers_to_opt += list(net_difix.vae.decoder.skip_conv_2.parameters())
    layers_to_opt += list(net_difix.vae.decoder.skip_conv_3.parameters())
    layers_to_opt += list(net_difix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,   # 已下调默认 1e-4
        eps=args.adam_epsilon,
    )

    # scheduler: cosine + warmup（默认）
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # datasets
    dataset_train = PairedDataset(dataset_path=args.dataset_path, split="train", tokenizer=net_difix.tokenizer)
    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, pin_memory=True
    )
    dataset_val = PairedDataset(dataset_path=args.dataset_path, split="test", tokenizer=net_difix.tokenizer)
    # random.Random(42).shuffle(dataset_val.img_names)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # resume
    global_step = 0
    if args.resume is not None:
        if os.path.isdir(args.resume):
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("=" * 50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("=" * 50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(net_difix, optimizer, ckpt_files[-1])
        elif args.resume.endswith(".pkl"):
            print("=" * 50); print(f"Loading checkpoint from {args.resume}"); print("=" * 50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(net_difix, optimizer, args.resume)
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("=" * 50); print("Training from scratch"); print("=" * 50)

    # precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # move train model到设备（LPIPS/VGG 已保持 FP32，且不通过 accelerator.prepare 包装，避免被半精度化）
    net_difix.to(accelerator.device, dtype=weight_dtype)

    # accelerator prepare
    net_difix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dl_train, lr_scheduler
    )

    # 归一化到 ImageNet 统计（先把 [−1,1] -> [0,1] 再标准化）
    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # trackers
    if accelerator.is_main_process:
        init_kwargs = {"wandb": {"name": args.tracker_run_name, "dir": args.output_dir}}
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)), init_kwargs=init_kwargs)

    # 可选 EMA
    ema_model = None
    if args.use_ema:
        ema_model = accelerator.unwrap_model(Difix(lora_rank_vae=args.lora_rank_vae, timestep=args.timestep, mv_unet=args.mv_unet))
        ema_model.load_state_dict(accelerator.unwrap_model(net_difix).state_dict(), strict=False)
        ema_model.to(accelerator.device, dtype=weight_dtype)
        ema_model.eval()

    # progress
    progress_bar = tqdm(total=args.max_train_steps, initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    # ---------------- training loop ----------------
    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(net_difix):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, V, C, H, W = x_src.shape

                # forward
                x_tgt_pred = net_difix(x_src, prompt_tokens=batch["input_ids"])

                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')

                # --- losses ---
                # 像素项（默认 Charbonnier，更抗异常值）
                if args.pixel_loss == "charbonnier":
                    loss_pix = charbonnier(x_tgt_pred.float() - x_tgt.float()).mean()
                elif args.pixel_loss == "huber":
                    huber = nn.SmoothL1Loss(beta=0.01)
                    loss_pix = huber(x_tgt_pred.float(), x_tgt.float())
                else:
                    loss_pix = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                loss_l2 = loss_pix * args.lambda_l2

                # LPIPS 用 FP32 计算（禁用 AMP）
                with torch.cuda.amp.autocast(enabled=False):
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips

                loss = loss_l2 + loss_lpips

                # Gram（默认关闭；若开启也 FP32 + eval VGG）
                if args.lambda_gram > 0 and global_step > args.gram_loss_warmup_steps:
                    x_pred_01 = x_tgt_pred * 0.5 + 0.5
                    x_tgt_01 = x_tgt * 0.5 + 0.5
                    x_tgt_pred_renorm = t_vgg_renorm(x_pred_01)
                    x_tgt_renorm = t_vgg_renorm(x_tgt_01)

                    crop_h, crop_w = min(400, H), min(400, W)
                    top = random.randint(0, H - crop_h)
                    left = random.randint(0, W - crop_w)
                    x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                    x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)

                    with torch.cuda.amp.autocast(enabled=False):
                        loss_gram = gram_loss(x_tgt_pred_renorm.float(), x_tgt_renorm.float(), net_vgg) * args.lambda_gram
                    loss = loss + loss_gram
                else:
                    loss_gram = torch.tensor(0.0, device=accelerator.device)

                # backward & step
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # 还原形状给可视化
                x_tgt = rearrange(x_tgt, '(b v) c h w -> b v c h w', v=V)
                x_tgt_pred = rearrange(x_tgt_pred, '(b v) c h w -> b v c h w', v=V)

            # post-step
            if accelerator.sync_gradients:
                # EMA
                if ema_model is not None:
                    update_ema(accelerator.unwrap_model(net_difix), ema_model, decay=args.ema_decay)

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_l2": float(loss_l2.detach().item()),
                        "loss_lpips": float(loss_lpips.detach().item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = float(loss_gram.detach().item())

                    # 可视化
                    if args.viz_freq > 0 and global_step % args.viz_freq == 1:
                        logs["train/source"] = [wandb.Image(rearrange(x_src, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)]
                        logs["train/target"] = [wandb.Image(rearrange(x_tgt, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)]
                        logs["train/model_output"] = [wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)]

                    # checkpoint
                    if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, outf)
                        if ema_model is not None:
                            outf_ema = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}_ema.pkl")
                            save_ckpt(ema_model, optimizer, outf_ema)

                    # eval（轻量）
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        with torch.no_grad():
                            for k_step, batch_val in enumerate(dl_val):
                                if k_step >= args.num_samples_eval:
                                    break
                                x_src_v = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                                x_tgt_v = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                                Bv, Vv, Cv, Hv, Wv = x_src_v.shape
                                assert Bv == 1, "Use batch size 1 for eval."

                                # 选择 EMA 或当前模型做推理
                                infer_model = accelerator.unwrap_model(net_difix)
                                if ema_model is not None:
                                    infer_model = ema_model
                                infer_model.eval()
                                x_tgt_pred_v = infer_model(x_src_v, prompt_tokens=batch_val["input_ids"].to(accelerator.device))
                                infer_model.train()

                                if k_step % 10 == 0:
                                    log_dict["sample/source"].append(wandb.Image(rearrange(x_src_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/target"].append(wandb.Image(rearrange(x_tgt_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/model_output"].append(wandb.Image(rearrange(x_tgt_pred_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))

                                # 只评第一个视角
                                x_tgt_v = x_tgt_v[:, 0]
                                x_tgt_pred_v = x_tgt_pred_v[:, 0]

                                # L2/LPIPS（LPIPS FP32）
                                loss_l2_v = F.mse_loss(x_tgt_pred_v.float(), x_tgt_v.float(), reduction="mean")
                                with torch.cuda.amp.autocast(enabled=False):
                                    loss_lpips_v = net_lpips(x_tgt_pred_v.float(), x_tgt_v.float()).mean()

                                l_l2.append(loss_l2_v.item())
                                l_lpips.append(loss_lpips_v.item())

                        logs["val/l2"] = float(np.mean(l_l2))
                        logs["val/lpips"] = float(np.mean(l_lpips))
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect(); torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # losses
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=0.0, type=float, help="默认关闭 Gram；若开启请配合 FP32+eval VGG")
    parser.add_argument("--gram_loss_warmup_steps", default=4000, type=int)
    parser.add_argument("--pixel_loss", choices=["charbonnier", "huber", "mse"], default="charbonnier")

    # dataset
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # eval / viz
    parser.add_argument("--eval_freq", default=200, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100)
    parser.add_argument("--viz_freq", type=int, default=200)
    parser.add_argument("--tracker_project_name", type=str, default="difix")
    parser.add_argument("--tracker_run_name", type=str, required=True)

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
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        help='["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)  # 调低
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    # EMA
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()
    main(args)
