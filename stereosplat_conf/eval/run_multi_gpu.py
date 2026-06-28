"""Multi-GPU inference — same metrics as ``eval/run.py``.

Supports ``eval_mode`` stereosplat / stereosplat_plus / pixel_fusion.
Shards val dataloader, merges per-rank sums, writes ``metric.json``.

Launch:
  accelerate launch --config_file accelerate_configs/inference/multi_gpu.yaml \\
      eval/run_multi_gpu.py --eval_mode stereosplat --pretrained_model_path ...
"""
from __future__ import annotations

import argparse
import importlib
import os
import os.path as osp
import sys
import warnings
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_STEREOSPLAT_ROOT = Path(__file__).resolve().parents[1]
if str(_STEREOSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STEREOSPLAT_ROOT))

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from mmengine import MMLogger
from mmengine.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

from eval.common import (  # noqa: E402
    dataset_module_for_world_center,
    load_state_dict_any,
    setup_import_paths,
)
from eval.routes import (  # noqa: E402
    accumulate_batch_metrics,
    init_metric_accumulators,
    merge_inference_metric_payloads,
    payload_from_metric_accumulators,
    requires_separated_stage1,
    run_batch_inference,
)

setup_import_paths()

from stereosplat.models_lab.StereoSplat.stereosplat import StereoSplat  # noqa: E402
from difix3d import DifixRef  # noqa: E402
from tools.metrics import saved_into_json  # noqa: E402


def build_model(cfg) -> StereoSplat:
    return StereoSplat(
        backbone=cfg.model.backbone,
        neck=cfg.model.neck,
        costvolume_gs=cfg.model.costvolume_gs,
        volume_gs=cfg.model.volume_gs,
        losses_params=cfg.model.losses_params,
        camera_args=cfg.camera_args,
        dataset_params=cfg.dataset_params,
        use_checkpoint=cfg.use_checkpoint,
    )


def load_frozen_stage1(accelerator, cfg, stage1_path: str) -> StereoSplat:
    frozen = build_model(cfg)
    accelerator.print(f"[Stage1] loading frozen model from: {stage1_path}")
    sd = load_state_dict_any(stage1_path, map_location="cpu")
    incompatible = frozen.load_state_dict(sd, strict=True)
    accelerator.print(
        f"[Stage1] loaded. missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    return frozen


def maybe_load_difix(args) -> DifixRef | None:
    if not args.use_diffix3d:
        return None
    if not args.pretrained_diffix_model_path:
        return None
    if not os.path.exists(args.pretrained_diffix_model_path):
        raise FileNotFoundError(
            f"Difix3D checkpoint not found: {args.pretrained_diffix_model_path}"
        )
    net = DifixRef(
        pretrained_name="nvidia/difix_ref",
        pretrained_path=args.pretrained_diffix_model_path,
        timestep=args.timestep,
        mv_unet=args.use_ref,
    )
    net.set_eval()
    return net


def gather_validation_payloads(accelerator, local_payload: dict) -> list[dict]:
    if accelerator.num_processes == 1:
        return [local_payload]
    gathered: list[dict | None] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, local_payload)
    return [p for p in gathered if p is not None]


def validate_args(args) -> None:
    if args.architecture == "separated" and requires_separated_stage1(
        args.eval_mode, args.architecture
    ):
        if not args.stage_1_model_path:
            raise ValueError(
                "architecture=separated requires --stage_1_model_path for frozen Stage1."
            )
    if (
        args.use_diffix3d
        and args.eval_mode == "pixel_fusion"
        and args.architecture == "whole"
        and not args.pretrained_diffix_model_path
    ):
        raise ValueError(
            "pixel_fusion + whole with Difix3D requires --pretrained_diffix_model_path "
            "(or pass --no_difix3d)."
        )
    if args.conf_fusion_margin is not None and not args.conf_pixel_level_fusion:
        raise ValueError("--conf_fusion_margin requires --conf_pixel_level_fusion.")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Multi-GPU inference (metric.json, same format as eval/run.py)"
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--pretrained_diffix_model_path", type=str, default="")
    parser.add_argument("--val_filelist", type=str, default="")
    parser.add_argument(
        "--eval_mode",
        choices=["stereosplat", "stereosplat_plus", "pixel_fusion"],
        default="pixel_fusion",
    )
    parser.add_argument(
        "--architecture",
        choices=["whole", "separated"],
        default="whole",
    )
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--prompt", type=str, default="remove degradation")
    parser.add_argument("--use_ref", action="store_true", default=False)
    parser.add_argument(
        "--no_difix3d",
        action="store_true",
        default=False,
        help="Skip Difix3D pseudo enhancement (raw pseudo views only).",
    )
    parser.add_argument("--self_pseudo", action="store_true", default=False)
    parser.add_argument("--stage_1_model_path", type=str, default=None)
    parser.add_argument("--pseudo_ratio", type=str, nargs="*", default=[])
    parser.add_argument(
        "--supp-view-nums",
        type=str,
        default="all",
        help="Val dataloader supp_view_nums (default all, same as run.py inference).",
    )
    parser.add_argument(
        "--conf_pixel_level_fusion",
        action="store_true",
        default=False,
    )
    parser.add_argument("--conf_fusion_margin", type=float, default=None)
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="soft",
        choices=["soft", "legacy", "per_view_adaptive"],
    )
    parser.add_argument("--fusion_first_margin", type=float, default=999.0)
    parser.add_argument("--fusion_center_margin", type=float, default=0.0)
    parser.add_argument("--fusion_last_margin", type=float, default=0.0)
    parser.add_argument(
        "--fusion_calibration",
        type=str,
        default="none",
        choices=["none", "zscore", "minmax"],
    )
    parser.add_argument("--fusion_temperature", type=float, default=None)
    parser.add_argument("--gs_conf_fusion", action="store_true", default=False)
    parser.add_argument("--gs_fusion_voxel_size", type=float, default=0.1)
    parser.add_argument("--gs_fusion_margin", type=float, default=0.05)
    parser.add_argument(
        "--gs_fusion_conf_agg",
        choices=["mean", "max"],
        default="mean",
    )
    parser.add_argument("--gs_fusion_base_conf_thresh", type=float, default=None)
    parser.add_argument(
        "--output_vis",
        action="store_true",
        default=False,
        help="Save per-batch visuals (main process only; single GPU recommended).",
    )
    args = parser.parse_args(argv)
    args.use_diffix3d = not args.no_difix3d

    args.pseudo_ratio = [float(x) for x in args.pseudo_ratio]
    if not args.pseudo_ratio and args.eval_mode in ("stereosplat_plus", "pixel_fusion"):
        args.pseudo_ratio = [0.5, 1.0]
    args.use_gt_view = False
    args.output_folder = args.output_folder
    validate_args(args)

    if args.output_vis and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("--output_vis is not supported with multiple processes.")

    cfg = Config.fromfile(args.config_path)
    cfg.work_dir = args.output_folder
    cfg.prompt = args.prompt

    MMLogger.get_instance("mmengine", log_level="WARNING")
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1),
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=cfg.work_dir,
            logging_dir=osp.join(cfg.work_dir, "logs"),
        ),
        kwargs_handlers=[kwargs],
    )

    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)

    dataset_config = cfg.dataset_params
    val_filelist = args.val_filelist or dataset_config.val_filelist
    supp_view_nums: int | str = (
        "all" if args.supp_view_nums == "all" else int(args.supp_view_nums)
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_folder, exist_ok=True)
        cfg.dump(osp.join(args.output_folder, osp.basename(args.config_path)))

    datasets = importlib.import_module(
        dataset_module_for_world_center(getattr(cfg, "world_center", None))
    )
    dataset_cls = getattr(datasets, dataset_config.dataset_name)
    val_dataset = dataset_cls(
        datapath=dataset_config.datapath,
        train_filelist=dataset_config.train_filelist,
        val_filelist=val_filelist,
        test_filelist=val_filelist,
        data_version=dataset_config.data_version,
        resolution=dataset_config.resolution,
        split="val",
        sequence=dataset_config.sequence,
        use_center=dataset_config.use_center,
        use_first=dataset_config.use_first,
        use_last=dataset_config.use_last,
        supp_view_nums=supp_view_nums,
        depth_info_dict=dataset_config.depth_info_params,
        camera_model=dataset_config.camera_model,
    )
    val_dataloader = DataLoader(
        val_dataset,
        dataset_config.batch_size_val,
        shuffle=False,
        num_workers=dataset_config.num_workers_val,
        pin_memory=True,
    )

    frozen_stage_1_model = None
    if args.architecture == "separated" and requires_separated_stage1(
        args.eval_mode, args.architecture
    ):
        if not args.stage_1_model_path:
            raise ValueError("--stage_1_model_path is required for architecture=separated.")
        frozen_stage_1_model = load_frozen_stage1(
            accelerator, cfg, args.stage_1_model_path
        )

    pretrained_diffix_model = maybe_load_difix(args)
    if pretrained_diffix_model is not None:
        accelerator.print(
            f"[Difix3D] loaded from {args.pretrained_diffix_model_path}"
        )
    elif args.use_diffix3d:
        accelerator.print("[Difix3D] enabled but no weights loaded (path empty).")
    else:
        accelerator.print("[Difix3D] disabled (--no_difix3d)")

    my_model = build_model(cfg)
    my_model, val_dataloader = accelerator.prepare(my_model, val_dataloader)

    if frozen_stage_1_model is not None:
        frozen_stage_1_model.to(accelerator.device)

    accelerator.print(f"Loading model weights from {args.pretrained_model_path}")
    sd = load_state_dict_any(args.pretrained_model_path, map_location="cpu")
    _model = my_model.module if hasattr(my_model, "module") else my_model
    incompatible = _model.load_state_dict(sd, strict=False)
    if incompatible.missing_keys:
        accelerator.print(
            f"  [warn] missing keys: {incompatible.missing_keys[:5]}"
            f"{'...' if len(incompatible.missing_keys) > 5 else ''}"
        )
    if incompatible.unexpected_keys:
        accelerator.print(
            f"  [warn] unexpected keys: {incompatible.unexpected_keys[:5]}"
            f"{'...' if len(incompatible.unexpected_keys) > 5 else ''}"
        )

    if pretrained_diffix_model is not None:
        pretrained_diffix_model.to(accelerator.device)

    accum = init_metric_accumulators(
        args.eval_mode, args.architecture, args.conf_pixel_level_fusion
    )
    batch_count = 0

    my_model.eval()
    with torch.no_grad():
        iterator = val_dataloader
        if accelerator.is_main_process:
            iterator = tqdm(val_dataloader, desc="[Infer MultiGPU]")

        for batch in iterator:
            bin_token_list = batch["bin_token"]
            evaluation_results_stat = run_batch_inference(
                my_model,
                batch,
                args,
                cfg,
                bin_token_list,
                eval_mode=args.eval_mode,
                architecture=args.architecture,
                pretrained_diffix_model=pretrained_diffix_model,
                frozen_stage_1_model=frozen_stage_1_model,
            )
            accumulate_batch_metrics(accum, evaluation_results_stat, args)
            batch_count += 1

    accelerator.wait_for_everyone()

    local_payload = payload_from_metric_accumulators(accum, batch_count)
    gathered = gather_validation_payloads(accelerator, local_payload)

    if accelerator.is_main_process:
        for i, payload in enumerate(gathered):
            print(
                f"[Infer MultiGPU] rank {i}: {payload['batch_count']} batches"
            )
        results_dict = merge_inference_metric_payloads(gathered)
        out_path = osp.join(args.output_folder, "metric.json")
        saved_into_json(data_dict=results_dict, path=out_path)
        print(f"[Infer MultiGPU] saved metric.json -> {out_path}")
        for branch in ("2v", "mv", "fuse"):
            sec = results_dict.get(branch)
            if not sec:
                continue
            print(
                f"  {branch}: "
                f"mean_psnr={sec.get('mean_psnr', float('nan')):.4f} "
                f"mean_ssim={sec.get('mean_ssim', float('nan')):.4f} "
                f"mean_abs_rel={sec.get('mean_abs_rel', float('nan')):.4f} "
                f"mean_sq_rel={sec.get('mean_sq_rel', float('nan')):.4f}"
            )
        rgb = results_dict.get("rgb")
        if rgb:
            print(
                f"  rgb: first_psnr={rgb.get('first_view_psnr_average', float('nan')):.4f} "
                f"center_psnr={rgb.get('center_view_psnr_average', float('nan')):.4f} "
                f"last_psnr={rgb.get('last_view_psnr_average', float('nan')):.4f} "
                f"all_psnr={rgb.get('all_view_psnr_average', float('nan')):.4f}"
            )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
