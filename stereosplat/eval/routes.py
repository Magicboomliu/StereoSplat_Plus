"""Map eval_mode × architecture to stereosplat.py inference functions.

Canonical model methods (see stereosplat.py):
  stereosplat              whole       -> infer_stereosplat_two_gt_views_forward
  stereosplat_plus         whole       -> infer_stereosplat_plus_pose_injection_single_model
  stereosplat_plus         separated   -> infer_stereosplat_plus_frozen_stage1_two_models
  pixel_fusion             whole       -> infer_pixel_fusion_pose_injection_single_model
  pixel_fusion             separated   -> infer_pixel_fusion_pose_injection_frozen_stage1_two_models
  (--use_gt_view)          any         -> infer_oracle_upper_bound_ablation (GT pseudo + RGB/Depth/Conf)
"""
from __future__ import annotations

from typing import Any


def requires_separated_stage1(eval_mode: str, architecture: str) -> bool:
    return architecture == "separated" and eval_mode in ("stereosplat_plus", "pixel_fusion")


def requires_pseudo_ratio(eval_mode: str) -> bool:
    return eval_mode in ("stereosplat_plus", "pixel_fusion") and True


def run_batch_inference(
    model,
    batch,
    args,
    cfg,
    bin_token_list,
    *,
    eval_mode: str,
    architecture: str,
    pretrained_diffix_model=None,
    frozen_stage_1_model=None,
) -> dict[str, Any]:
    if args.use_gt_view:
        return model.infer_oracle_upper_bound_ablation(
            batch,
            args.output_folder,
            bin_token_list,
            pseudo_ratio_index=args.pseudo_ratio,
            cfg=cfg,
            start_images_views=2,
            use_diffix3d=args.use_diffix3d,
            diffix3d_network=pretrained_diffix_model,
            use_ref=args.use_ref,
            vis=args.output_vis,
        )

    if eval_mode == "stereosplat":
        return model.infer_stereosplat_two_gt_views_forward(
            batch,
            args.output_folder,
            bin_token_list,
            cfg=cfg,
            view_num=2,
            matching_nums=2,
            vis=args.output_vis,
        )

    if eval_mode == "stereosplat_plus" and architecture == "whole":
        return model.infer_stereosplat_plus_pose_injection_single_model(
            batch,
            args.output_folder,
            bin_token_list,
            pseudo_ratio_index=args.pseudo_ratio,
            cfg=cfg,
            start_images_views=2,
            use_diffix3d=args.use_diffix3d,
            diffix3d_network=pretrained_diffix_model,
            use_ref=args.use_ref,
            vis=args.output_vis,
        )

    if eval_mode == "stereosplat_plus" and architecture == "separated":
        return model.infer_stereosplat_plus_frozen_stage1_two_models(
            batch,
            args.output_folder,
            bin_token_list,
            pseudo_ratio_index=args.pseudo_ratio,
            cfg=cfg,
            start_images_views=2,
            use_diffix3d=args.use_diffix3d,
            diffix3d_network=pretrained_diffix_model,
            frozen_stage_1_model=frozen_stage_1_model,
            use_ref=args.use_ref,
            vis=args.output_vis,
        )

    if eval_mode == "pixel_fusion" and architecture == "whole":
        return model.infer_pixel_fusion_pose_injection_single_model(
            batch,
            args.output_folder,
            bin_token_list,
            pseudo_ratio_index=args.pseudo_ratio,
            cfg=cfg,
            start_images_views=2,
            use_diffix3d=args.use_diffix3d,
            diffix3d_network=pretrained_diffix_model,
            use_ref=args.use_ref,
            vis=args.output_vis,
            pixel_level_conf_fusion=args.conf_pixel_level_fusion,
        )

    if eval_mode == "pixel_fusion" and architecture == "separated":
        return model.infer_pixel_fusion_pose_injection_frozen_stage1_two_models(
            batch,
            args.output_folder,
            bin_token_list,
            pseudo_ratio_index=args.pseudo_ratio,
            cfg=cfg,
            start_images_views=2,
            use_diffix3d=args.use_diffix3d,
            diffix3d_network=pretrained_diffix_model,
            frozen_stage_1_model=frozen_stage_1_model,
            use_ref=args.use_ref,
            vis=args.output_vis,
            pixel_level_conf_fusion=args.conf_pixel_level_fusion,
        )

    raise ValueError(
        f"Unsupported combination: eval_mode={eval_mode}, architecture={architecture}"
    )


CONF_KEY_ALIASES = {
    "Conf": "conf",
    "Conf_stage1": "conf_stage1",
    "Conf_stage2": "conf_stage2",
    "Conf_2view": "conf_2view",
    "Conf_pseudo_multiview": "conf_pseudo_multiview",
}


def init_metric_accumulators(eval_mode: str, architecture: str, conf_fusion: bool) -> dict:
    accum: dict[str, dict] = {
        "rgb": {},
        "depth": {},
    }
    if eval_mode in ("stereosplat_plus", "pixel_fusion"):
        accum["conf"] = {}
    if eval_mode == "pixel_fusion" and conf_fusion:
        accum["conf_fused"] = {}
    if architecture == "separated":
        accum["conf_stage1"] = {}
        accum["conf_stage2"] = {}
    if eval_mode == "pixel_fusion" and architecture == "whole":
        accum["conf_2view"] = {}
        accum["conf_pseudo_multiview"] = {}
    return accum


def accumulate_batch_metrics(accum: dict, evaluation_results_stat: dict, args) -> None:
    for key in evaluation_results_stat.get("RGB", {}):
        accum["rgb"][key] = accum["rgb"].get(key, 0.0) + evaluation_results_stat["RGB"][key]
    for key in evaluation_results_stat.get("Depth", {}):
        accum["depth"][key] = accum["depth"].get(key, 0.0) + evaluation_results_stat["Depth"][key]

    conf_fusion = bool(getattr(args, "conf_pixel_level_fusion", False))
    if "Conf" in evaluation_results_stat:
        bucket = "conf_fused" if conf_fusion and "conf_fused" in accum else "conf"
        if bucket not in accum:
            accum[bucket] = {}
        for key, val in evaluation_results_stat["Conf"].items():
            accum[bucket][key] = accum[bucket].get(key, 0.0) + val

    for src_key, dst_key in CONF_KEY_ALIASES.items():
        if src_key == "Conf":
            continue
        if src_key in evaluation_results_stat and dst_key in accum:
            for key, val in evaluation_results_stat[src_key].items():
                accum[dst_key][key] = accum[dst_key].get(key, 0.0) + val

    if "Oracle_reference" in evaluation_results_stat:
        bucket = "oracle_reference"
        if bucket not in accum:
            accum[bucket] = {}
        for key, val in evaluation_results_stat["Oracle_reference"].items():
            accum[bucket][key] = accum[bucket].get(key, 0.0) + val


def finalize_metrics(accum: dict, batch_idx: int) -> dict:
    if batch_idx == 0:
        raise RuntimeError("Validation dataloader is empty; cannot compute metrics.")
    results = {}
    for section, values in accum.items():
        if not values:
            continue
        results[section] = {k: v / batch_idx for k, v in values.items()}
    return results


def wandb_logs_from_metrics(metrics: dict, args) -> dict:
    logs = {}
    for k, v in metrics.get("rgb", {}).items():
        logs[f"val/rgb/{k}"] = float(v)
    for k, v in metrics.get("depth", {}).items():
        logs[f"val/depth/{k}"] = float(v)
    conf_fusion = bool(getattr(args, "conf_pixel_level_fusion", False))
    for section in ("conf", "conf_fused", "conf_stage1", "conf_stage2", "conf_2view", "conf_pseudo_multiview"):
        if section not in metrics:
            continue
        prefix = f"val/{section}"
        if section == "conf" and conf_fusion:
            prefix = "val/conf_fused"
        for k, v in metrics[section].items():
            logs[f"{prefix}/{k}"] = float(v)
    return logs
