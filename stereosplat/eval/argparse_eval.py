"""Shared CLI for eval/run.py and eval/run_multi_gpu.py (keep in sync manually)."""
from __future__ import annotations

import argparse


def build_eval_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
    parser.add_argument(
        "--conf_fusion_margin",
        type=float,
        default=None,
        help=(
            "A1 conf fusion: pick plus only when conf_plus > conf_base + margin "
            "(ties prefer base). Omit for legacy conf_plus >= conf_base (ties prefer plus)."
        ),
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="soft",
        choices=["soft", "legacy", "per_view_adaptive"],
        help=(
            "soft=train/val-aligned sigmoid conf fusion (reads fusion_sup_dict); "
            "legacy=global hard conf pick; per_view_adaptive=per-view margins."
        ),
    )
    parser.add_argument(
        "--fusion_first_margin",
        type=float,
        default=999.0,
        help="First-frame margin (999 forces base). Used with per_view_adaptive.",
    )
    parser.add_argument(
        "--fusion_center_margin",
        type=float,
        default=0.0,
        help="Center-frame margin for per_view_adaptive.",
    )
    parser.add_argument(
        "--fusion_last_margin",
        type=float,
        default=0.0,
        help="Last-frame margin for per_view_adaptive.",
    )
    parser.add_argument(
        "--fusion_calibration",
        type=str,
        default="none",
        choices=["none", "zscore", "minmax"],
        help="Per-image conf calibration before fusion (per_view_adaptive).",
    )
    parser.add_argument(
        "--fusion_temperature",
        type=float,
        default=None,
        help="Soft blending temperature; omit for hard selection (per_view_adaptive).",
    )
    parser.add_argument(
        "--gs_conf_fusion",
        action="store_true",
        default=False,
        help=(
            "Fuse G_base and G_plus in 3D (voxel conf winner + margin). "
            "With --conf_pixel_level_fusion: pixel-fuse G_base render vs G_gs_fused render."
        ),
    )
    parser.add_argument(
        "--gs_fusion_voxel_size",
        type=float,
        default=0.1,
        help="Voxel size (meters) for --gs_conf_fusion.",
    )
    parser.add_argument(
        "--gs_fusion_margin",
        type=float,
        default=0.05,
        help="Plus wins a voxel when agg(conf_plus) > agg(conf_base) + margin.",
    )
    parser.add_argument(
        "--gs_fusion_conf_agg",
        choices=["mean", "max"],
        default="mean",
        help="Per-voxel conf aggregation over Gaussians on each side (default: mean).",
    )
    parser.add_argument(
        "--gs_fusion_base_conf_thresh",
        type=float,
        default=None,
        help=(
            "Base-priority gate: plus wins a voxel only if mean/max conf_base < this "
            "(and margin rule). Omit to use relative conf comparison only."
        ),
    )
    parser.add_argument("--output_vis", action="store_true", default=False)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--pseudo_ratio", type=str, nargs="*", default=[])
    return parser
