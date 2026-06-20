"""Multi-GPU evaluation entry (4-way data-parallel inference + global metrics).

Shards the validation dataloader across GPUs via ``accelerator.prepare``, runs the
same inference stack as ``eval/run.py``, then merges per-rank metric *sums* on the
main process so ``metric.json`` matches single-GPU semantics.

Launch (4 GPUs):
  accelerate launch --config_file accelerate_configs/inference/multi_gpu.yaml \\
      eval/run_multi_gpu.py --output_folder ... [same flags as eval/run.py]

Notes:
  - ``--output_vis`` is disallowed with multiple processes (file collisions).
  - Use ``accelerate_configs/inference/multi_gpu.yaml`` (not gpu_0.yaml).
"""
from __future__ import annotations

import argparse
import importlib
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import os.path as osp
import sys
import warnings
from datetime import timedelta
from pathlib import Path

_STEREOSPLAT_ROOT = Path(__file__).resolve().parents[1]
if str(_STEREOSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STEREOSPLAT_ROOT))

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import (
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from mmengine import MMLogger
from mmengine.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

from eval.argparse_eval import build_eval_argument_parser
from eval.common import (
    config_path_for_stage,
    dataset_module_for_world_center,
    load_state_dict_any,
    maybe_init_wandb,
    setup_import_paths,
)
from eval.routes import (
    accumulate_batch_metrics,
    finalize_metrics,
    init_metric_accumulators,
    requires_separated_stage1,
    run_batch_inference,
    wandb_logs_from_metrics,
)
from eval.run import (
    apply_defaults,
    build_model,
    load_frozen_stage1,
    maybe_load_difix,
    validate_args,
)

setup_import_paths()
torch.autograd.set_detect_anomaly(True)

from tools.metrics import saved_into_json  # noqa: E402


def _resolve_inference_model(model):
    """Unwrap DDP / Accelerate wrapper for inference method dispatch."""
    if hasattr(model, "module"):
        return model.module
    return model


def _all_gather_rank_metrics(
    accum: dict,
    batch_count: int,
) -> list[tuple[dict, int]]:
    """Collect per-rank (accum, batch_count) via ``all_gather_object``.

    Do not use ``accelerate.utils.gather_object`` here: its GPU implementation
    flattens gathered dicts by iterating keys, which breaks metric aggregation.
    """
    if not dist.is_available() or not dist.is_initialized():
        return [(accum, int(batch_count))]

    world_size = dist.get_world_size()
    payloads: list = [None] * world_size
    dist.all_gather_object(
        payloads,
        {"accum": accum, "batch_count": int(batch_count)},
    )
    gathered: list[tuple[dict, int]] = []
    for rank_id, item in enumerate(payloads):
        if not isinstance(item, dict) or "accum" not in item or "batch_count" not in item:
            raise TypeError(
                f"Rank {rank_id} gathered invalid payload: {type(item)!r}"
            )
        gathered.append((item["accum"], int(item["batch_count"])))
    return gathered


def _merge_metric_dicts(target: dict, source: dict) -> None:
    """Recursively sum scalar leaves in nested metric accumulators."""
    for key, val in source.items():
        if isinstance(val, dict):
            bucket = target.setdefault(key, {})
            _merge_metric_dicts(bucket, val)
        else:
            target[key] = target.get(key, 0.0) + float(val)


def _merge_gathered_metrics(
    gathered_payloads: list[tuple[dict, int]],
) -> dict:
    """Sum per-rank metric accumulators and divide by total batch count."""
    total_batches = sum(count for _, count in gathered_payloads)
    if total_batches == 0:
        raise RuntimeError(
            "Validation dataloader produced no batches across all GPUs."
        )

    merged: dict = {}
    for rank_accum, _ in gathered_payloads:
        for section, values in rank_accum.items():
            if not values:
                continue
            bucket = merged.setdefault(section, {})
            _merge_metric_dicts(bucket, values)

    if not merged:
        raise RuntimeError(
            "No metric keys accumulated on any rank; check eval_mode / filelist."
        )

    return finalize_metrics(merged, total_batches)


def main(args=None, defaults: dict | None = None):
    parser = build_eval_argument_parser(
        description="Multi-GPU confidence StereoSplat evaluation"
    )
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    args = apply_defaults(args, defaults)
    args.pseudo_ratio = [float(x) for x in args.pseudo_ratio]

    if args.eval_mode is None:
        if args.conf_pixel_level_fusion:
            args.eval_mode = "pixel_fusion"
        else:
            raise ValueError("Pass --eval_mode or use a legacy validator wrapper with defaults.")

    if args.training_stage is None:
        args.training_stage = "stage2"

    if not args.config_path:
        args.config_path = config_path_for_stage(args.training_stage)

    if not args.pseudo_ratio and (
        args.use_gt_view or args.eval_mode in ("stereosplat_plus", "pixel_fusion")
    ):
        args.pseudo_ratio = [0.5, 1.0]

    validate_args(args)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    cfg = Config.fromfile(args.config_path)
    cfg.work_dir = args.output_folder
    cfg.prompt = args.prompt
    cfg.use_diffix3d_postprocessing = args.use_diffix3d_postprocessing

    tracker_enabled = bool(getattr(args, "use_wandb", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=("wandb" if tracker_enabled else None),
        project_config=ProjectConfiguration(
            project_dir=cfg.work_dir,
            logging_dir=os.path.join(cfg.work_dir, "logs"),
        ),
        kwargs_handlers=[kwargs],
    )

    if args.output_vis and accelerator.num_processes > 1:
        raise ValueError(
            "--output_vis is not supported with multi-GPU inference "
            "(ranks would write the same paths). Run eval/run.py on a single GPU "
            "for visualization, or disable --output_vis here."
        )

    tracker_enabled = maybe_init_wandb(accelerator, args, cfg)

    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)

    if accelerator.is_main_process:
        os.makedirs(args.output_folder, exist_ok=True)
        cfg.dump(osp.join(args.output_folder, osp.basename(args.config_path)))
        accelerator.print(
            f"[MultiGPU] processes={accelerator.num_processes}, "
            f"device={accelerator.device}"
        )

    MMLogger.get_instance("mmengine", log_level="WARNING")

    dataset_config = cfg.dataset_params
    datasets = importlib.import_module(
        dataset_module_for_world_center(getattr(cfg, "world_center", None))
    )
    dataset_cls = getattr(datasets, dataset_config.dataset_name)
    val_filelist = args.demo_filelist if args.output_vis else args.val_filelist
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
        supp_view_nums="all",
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
    if requires_separated_stage1(args.eval_mode, args.architecture):
        frozen_stage_1_model = load_frozen_stage1(
            accelerator, cfg, args.stage_1_model_path
        )

    force_difix = args.eval_mode == "pixel_fusion" and args.architecture == "whole"
    pretrained_diffix_model = maybe_load_difix(args, force=force_difix)

    my_model = build_model(cfg)
    my_model, val_dataloader = accelerator.prepare(my_model, val_dataloader)

    if frozen_stage_1_model is not None:
        frozen_stage_1_model.to(accelerator.device)
        frozen_stage_1_model.eval()

    path = args.pretrained_model_path or getattr(cfg, "pretrained_model_path", None)
    if path:
        accelerator.print(f"[rank {accelerator.process_index}] loading weights: {path}")
        sd = load_state_dict_any(path, map_location="cpu")
        _model = _resolve_inference_model(my_model)
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
    else:
        accelerator.print(
            f"[rank {accelerator.process_index}] no checkpoint; random init."
        )

    if pretrained_diffix_model is not None:
        pretrained_diffix_model.to(accelerator.device)
        pretrained_diffix_model.eval()

    if args.use_gt_view:
        accum = {"rgb": {}, "depth": {}, "oracle_reference": {}}
    else:
        accum = init_metric_accumulators(
            args.eval_mode, args.architecture, args.conf_pixel_level_fusion
        )

    infer_model = _resolve_inference_model(my_model)
    my_model.eval()
    infer_model.eval()

    batch_idx = 0
    progress = tqdm(
        val_dataloader,
        disable=not accelerator.is_main_process,
        desc=f"eval multi-gpu ({accelerator.num_processes} ranks)",
    )

    with torch.no_grad():
        for batch in progress:
            evaluation_results_stat = run_batch_inference(
                infer_model,
                batch,
                args,
                cfg,
                batch["bin_token"],
                eval_mode=args.eval_mode,
                architecture=args.architecture,
                pretrained_diffix_model=pretrained_diffix_model,
                frozen_stage_1_model=frozen_stage_1_model,
            )
            accumulate_batch_metrics(accum, evaluation_results_stat, args)
            batch_idx += 1

    accelerator.wait_for_everyone()

    gathered_payloads = _all_gather_rank_metrics(accum, batch_idx)

    if accelerator.is_main_process:
        for rank_id, (_, rank_count) in enumerate(gathered_payloads):
            accelerator.print(f"[MultiGPU] rank {rank_id}: {rank_count} batches")

    results_dict = None
    if accelerator.is_main_process:
        results_dict = _merge_gathered_metrics(gathered_payloads)

    if not args.output_vis and accelerator.is_main_process and results_dict is not None:
        metric_path = os.path.join(args.output_folder, "metric.json")
        saved_into_json(data_dict=results_dict, path=metric_path)
        accelerator.print(f"[MultiGPU] saved metrics -> {metric_path}")

    if tracker_enabled and accelerator.is_main_process:
        accelerator.log(wandb_logs_from_metrics(results_dict, args), step=0)

    accelerator.wait_for_everyone()
    return results_dict


if __name__ == "__main__":
    main()
