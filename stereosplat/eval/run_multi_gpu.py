"""Multi-GPU Stage2 fusion validation — same logic as training ``fusion_metric.json``.

Shards the val dataloader across GPUs, runs training-aligned two-pass validation
per batch, merges per-rank PSNR lists on the main process, and writes
``fusion_metric.json`` (identical schema to training validation).

Launch:
  accelerate launch --config_file accelerate_configs/inference/multi_gpu.yaml \\
      eval/run_multi_gpu.py --output_folder ... --pretrained_model_path ...
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

setup_import_paths()

from eval.fusion_validation import (  # noqa: E402
    accumulate_batch_fusion_metrics,
    merge_fusion_metric_payloads,
    new_fusion_metric_accumulators,
    payload_from_accumulators,
)
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


def maybe_load_difix(accelerator, args) -> DifixRef | None:
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
    accelerator.print(f"[Difix3D] loaded from {args.pretrained_diffix_model_path}")
    return net


def gather_validation_payloads(accelerator, local_payload: dict) -> list[dict]:
    if accelerator.num_processes == 1:
        return [local_payload]
    gathered: list[dict | None] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, local_payload)
    return [p for p in gathered if p is not None]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Stage2 fusion validation (training-aligned)"
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--pretrained_diffix_model_path", type=str, required=True)
    parser.add_argument("--val_filelist", type=str, default="")
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--prompt", type=str, default="remove degradation")
    parser.add_argument("--use_ref", action="store_true", default=False)
    parser.add_argument("--self_pseudo", action="store_true", default=False)
    parser.add_argument(
        "--stage_1_model_path",
        type=str,
        default=None,
        help="Frozen Stage1 for non-self-pseudo validation (ignored when --self_pseudo).",
    )
    parser.add_argument(
        "--output_vis",
        action="store_true",
        default=False,
        help="Save per-batch visuals (main process only; not recommended multi-GPU).",
    )
    args = parser.parse_args()

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
    if args.val_filelist:
        val_filelist = args.val_filelist
    else:
        val_filelist = dataset_config.val_filelist

    if accelerator.is_main_process:
        os.makedirs(args.output_folder, exist_ok=True)
        cfg.dump(osp.join(args.output_folder, osp.basename(args.config_path)))

    datasets = importlib.import_module(
        dataset_module_for_world_center(getattr(cfg, "world_center", None))
    )
    dataset_cls = getattr(datasets, dataset_config.dataset_name)

    # Match trainer val dataloader (supp_view_nums=3, not "all").
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
        supp_view_nums=3,
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
    if not args.self_pseudo:
        if not args.stage_1_model_path:
            raise ValueError("--stage_1_model_path is required without --self_pseudo.")
        frozen_stage_1_model = load_frozen_stage1(
            accelerator, cfg, args.stage_1_model_path
        )

    pretrained_diffix_model = maybe_load_difix(accelerator, args)

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

    acc_2view, acc_mv, acc_fus = new_fusion_metric_accumulators()
    batch_count = 0

    my_model.eval()
    with torch.no_grad():
        iterator = val_dataloader
        if accelerator.is_main_process:
            iterator = tqdm(val_dataloader, desc="[Val MultiGPU]")

        for batch in iterator:
            val_batch_save_dir = ""
            if args.output_vis and accelerator.is_main_process:
                val_batch_save_dir = osp.join(
                    args.output_folder, f"batch-{batch_count}"
                )
                os.makedirs(val_batch_save_dir, exist_ok=True)

            accumulate_batch_fusion_metrics(
                acc_2view,
                acc_mv,
                acc_fus,
                my_model,
                batch,
                cfg,
                frozen_stage_1_model=frozen_stage_1_model,
                pretrained_diffix_model=pretrained_diffix_model,
                self_pseudo=args.self_pseudo,
                val_batch_save_dir=val_batch_save_dir,
                save_visuals=args.output_vis,
            )
            batch_count += 1

    accelerator.wait_for_everyone()

    local_payload = payload_from_accumulators(
        acc_2view, acc_mv, acc_fus, batch_count
    )
    gathered = gather_validation_payloads(accelerator, local_payload)

    if accelerator.is_main_process:
        for i, payload in enumerate(gathered):
            print(
                f"[Val MultiGPU] rank {i}: {payload['batch_count']} batches"
            )
        fusion_metric = merge_fusion_metric_payloads(gathered)
        out_path = osp.join(args.output_folder, "fusion_metric.json")
        saved_into_json(data_dict=fusion_metric, path=out_path)
        print(f"[Val MultiGPU] saved fusion_metric -> {out_path}")
        for sec, data in fusion_metric.items():
            print(
                f"  {sec}: mean={data['psnr_mean']:.4f} "
                f"center={data['psnr_center']:.4f} last={data['psnr_last']:.4f}"
            )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
