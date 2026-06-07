"""Unified evaluation entry for confidence StereoSplat.

Usage (direct):
  python eval/run.py --training_stage stage2 --eval_mode pixel_fusion \\
      --architecture separated --config_path ... --output_folder ...

Legacy validator scripts are thin wrappers that call main(defaults={...}).
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

# Bootstrap before `import eval.*` (works for `python eval/run.py` and validator wrappers).
_STEREOSPLAT_ROOT = Path(__file__).resolve().parents[1]
if str(_STEREOSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STEREOSPLAT_ROOT))

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from mmengine import MMLogger
from mmengine.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

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

setup_import_paths()
torch.autograd.set_detect_anomaly(True)

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


def maybe_load_difix(args, *, force: bool = False):
    if not force and not args.use_diffix3d:
        return None
    if not args.pretrained_diffix_model_path or not os.path.exists(args.pretrained_diffix_model_path):
        raise FileNotFoundError(
            "The pretrained Difix3D model path does not exist: "
            f"{args.pretrained_diffix_model_path}"
        )
    net = DifixRef(
        pretrained_name="nvidia/difix_ref",
        pretrained_path=args.pretrained_diffix_model_path,
        timestep=args.timestep,
        mv_unet=args.use_ref,
        deterministic_vae_encode=args.deterministic_vae_encode,
        deterministic_scheduler_step=args.deterministic_scheduler_step,
    )
    net.set_eval()
    return net


def apply_defaults(args, defaults: dict | None) -> argparse.Namespace:
    if not defaults:
        return args
    for key, value in defaults.items():
        if getattr(args, key, None) in (None, "", []):
            setattr(args, key, value)
    return args


def validate_args(args) -> None:
    if args.architecture == "separated" and requires_separated_stage1(args.eval_mode, args.architecture):
        if not args.stage_1_model_path:
            raise ValueError(
                "architecture=separated requires --stage_1_model_path for frozen Stage1 model."
            )
    if args.eval_mode == "pixel_fusion" and args.architecture == "whole":
        if not args.pretrained_diffix_model_path:
            raise ValueError(
                "pixel_fusion + whole loads Difix3D weights (legacy behavior); "
                "pass --pretrained_diffix_model_path."
            )


def main(args=None, defaults: dict | None = None):
    parser = argparse.ArgumentParser(description="Confidence StereoSplat evaluation")
    parser.add_argument("--training_stage", choices=["stage1", "stage2"], default=None)
    parser.add_argument(
        "--eval_mode",
        choices=["stereosplat", "stereosplat_plus", "pixel_fusion"],
        default=None,
    )
    parser.add_argument(
        "--architecture",
        choices=["whole", "separated"],
        default="whole",
    )
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, default="")
    parser.add_argument("--stage_1_model_path", type=str, default=None)
    parser.add_argument("--val_filelist", type=str, default="")
    parser.add_argument("--demo_filelist", type=str, default="")
    parser.add_argument("--ablation_type", type=str, default="NMRFStereo")
    parser.add_argument("--dataset_type", type=str, default="First_LiDAR_3_Uniform")
    parser.add_argument("--pretrained_diffix_model_path", type=str, default="")
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--prompt", type=str, default="remove degradation")
    parser.add_argument("--use_ref", action="store_true", default=False)
    parser.add_argument("--use_gt_view", action="store_true", default=False)
    parser.add_argument("--use_diffix3d", action="store_true", default=False)
    parser.add_argument("--use_diffix3d_postprocessing", action="store_true", default=False)
    parser.add_argument("--deterministic_vae_encode", action="store_true", default=False)
    parser.add_argument("--deterministic_scheduler_step", action="store_true", default=False)
    parser.add_argument(
        "--conf_pixel_level_fusion",
        action="store_true",
        default=False,
        help="Per-pixel confidence fusion (pixel_fusion mode, or legacy separated/whole validators).",
    )
    parser.add_argument("--output_vis", action="store_true", default=False)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--pseudo_ratio", type=str, nargs="*", default=[])

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    args = apply_defaults(args, defaults)
    args.pseudo_ratio = [float(x) for x in args.pseudo_ratio]
    args.gpus = torch.cuda.device_count()

    if args.eval_mode is None:
        if args.conf_pixel_level_fusion:
            args.eval_mode = "pixel_fusion"
        else:
            raise ValueError("Pass --eval_mode or use a legacy validator wrapper with defaults.")

    if args.training_stage is None:
        args.training_stage = "stage2"

    if not args.config_path:
        args.config_path = config_path_for_stage(args.training_stage)

    validate_args(args)

    cfg = Config.fromfile(args.config_path)
    cfg.work_dir = args.output_folder
    cfg.prompt = args.prompt
    cfg.use_diffix3d_postprocessing = args.use_diffix3d_postprocessing

    MMLogger.get_instance("mmengine", log_level="WARNING")
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.work_dir,
        logging_dir=os.path.join(cfg.work_dir, "logs"),
    )
    tracker_enabled = bool(getattr(args, "use_wandb", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=("wandb" if tracker_enabled else None),
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    tracker_enabled = maybe_init_wandb(accelerator, args, cfg)

    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)

    dataset_config = cfg.dataset_params
    if accelerator.is_main_process:
        os.makedirs(args.output_folder, exist_ok=True)
        cfg.dump(osp.join(args.output_folder, osp.basename(args.config_path)))

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

    if args.pretrained_model_path:
        cfg.pretrained_model_path = args.pretrained_model_path
    path = getattr(cfg, "pretrained_model_path", None) or None
    if path:
        accelerator.print(f"Loading from checkpoint {path}")
        accelerator.load_state(path, map_location="cpu", strict=False)
        print(f"Successfully loaded from {path}")
    else:
        print("Can't find checkpoint. Randomly initialize model parameters anyway.")

    if pretrained_diffix_model is not None:
        pretrained_diffix_model.to(accelerator.device)

    accum = init_metric_accumulators(
        args.eval_mode, args.architecture, args.conf_pixel_level_fusion
    )

    with torch.no_grad():
        my_model.eval()
        batch_idx = 0
        for batch in tqdm(val_dataloader):
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
            batch_idx += 1

    results_dict = finalize_metrics(accum, batch_idx)

    if not args.output_vis and accelerator.is_main_process:
        saved_into_json(
            data_dict=results_dict,
            path=os.path.join(args.output_folder, "metric.json"),
        )
    if tracker_enabled and accelerator.is_main_process:
        accelerator.log(wandb_logs_from_metrics(results_dict, args), step=0)


if __name__ == "__main__":
    main()
