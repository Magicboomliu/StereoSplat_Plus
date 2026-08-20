"""Map eval_mode × architecture to stereosplat.py inference functions.

Canonical model methods (see stereosplat.py):
  stereosplat              whole       -> infer_stereosplat_two_gt_views_forward
  stereosplat_plus         whole       -> infer_stereosplat_plus_pose_injection_single_model
  stereosplat_plus         separated   -> infer_stereosplat_plus_frozen_stage1_two_models
  pixel_fusion             whole       -> infer_pixel_fusion_pose_injection_single_model
  pixel_fusion             separated   -> infer_pixel_fusion_pose_injection_frozen_stage1_two_models
  bev                      whole       -> get_additional_bev_novel_views_non_progressive
  bev_plus                 whole       -> get_additional_bev_novel_views_progressive_iter_once
  (--use_gt_view)          any         -> infer_oracle_upper_bound_ablation (GT pseudo + RGB/Depth/Conf)
"""
from __future__ import annotations

from typing import Any


def _resolve_model(model) -> Any:
    """Unwrap DDP / Accelerate wrapper so custom infer_* methods are reachable."""
    return model.module if hasattr(model, "module") else model


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
    model = _resolve_model(model)

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

    if eval_mode == "bev":
        model.get_additional_bev_novel_views_non_progressive(
            batch,
            args.output_folder,
            bin_token_list,
            cfg=cfg,
            view_num=2,
            matching_nums=2,
            vis=args.output_vis,
            rescale_h=args.bev_rescale_h,
            rescale_w=args.bev_rescale_w,
        )
        return {}

    if eval_mode == "bev_plus":
        model.get_additional_bev_novel_views_progressive_iter_once(
            batch,
            args.output_folder,
            bin_token_list,
            cfg=cfg,
            start_images_views=2,
            use_diffix3d=args.use_diffix3d,
            diffix3d_network=pretrained_diffix_model,
            use_ref=args.use_ref,
            vis=args.output_vis,
            rescale_h=args.bev_rescale_h,
            rescale_w=args.bev_rescale_w,
        )
        return {}

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
            conf_fusion_margin=args.conf_fusion_margin,
            fusion_mode=args.fusion_mode,
            fusion_first_margin=args.fusion_first_margin,
            fusion_center_margin=args.fusion_center_margin,
            fusion_last_margin=args.fusion_last_margin,
            fusion_calibration=args.fusion_calibration,
            fusion_temperature=args.fusion_temperature,
            gs_conf_fusion=args.gs_conf_fusion,
            gs_fusion_voxel_size=args.gs_fusion_voxel_size,
            gs_fusion_margin=args.gs_fusion_margin,
            gs_fusion_conf_agg=args.gs_fusion_conf_agg,
            gs_fusion_base_conf_thresh=args.gs_fusion_base_conf_thresh,
            output_vis_video=getattr(args, "output_vis_video", False),
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
            conf_fusion_margin=args.conf_fusion_margin,
            fusion_mode=args.fusion_mode,
            fusion_first_margin=args.fusion_first_margin,
            fusion_center_margin=args.fusion_center_margin,
            fusion_last_margin=args.fusion_last_margin,
            fusion_calibration=args.fusion_calibration,
            fusion_temperature=args.fusion_temperature,
            gs_conf_fusion=args.gs_conf_fusion,
            gs_fusion_voxel_size=args.gs_fusion_voxel_size,
            gs_fusion_margin=args.gs_fusion_margin,
            gs_fusion_conf_agg=args.gs_fusion_conf_agg,
            gs_fusion_base_conf_thresh=args.gs_fusion_base_conf_thresh,
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
    "Conf_gs_fused": "conf_gs_fused",
}

INFER_BRANCH_NAMES = ("2v", "mv", "fuse")

# metric.json field order per branch (mean_* = all_view over full V)
INFER_BRANCH_METRIC_KEYS = (
    "first_psnr", "first_ssim", "first_abs_rel", "first_sq_rel",
    "center_psnr", "center_ssim", "center_abs_rel", "center_sq_rel",
    "last_psnr", "last_ssim", "last_abs_rel", "last_sq_rel",
    "mean_psnr", "mean_ssim", "mean_abs_rel", "mean_sq_rel",
)


def init_metric_accumulators(eval_mode: str, architecture: str, conf_fusion: bool) -> dict:
    if eval_mode == "pixel_fusion":
        return {branch: {} for branch in INFER_BRANCH_NAMES}
    accum: dict[str, dict] = {
        "rgb": {},
        "depth": {},
    }
    if eval_mode in ("stereosplat_plus",):
        accum["conf"] = {}
    if architecture == "separated":
        accum["conf_stage1"] = {}
        accum["conf_stage2"] = {}
    return accum


def accumulate_batch_metrics(accum: dict, evaluation_results_stat: dict, args) -> None:
    for branch in INFER_BRANCH_NAMES:
        if branch not in evaluation_results_stat:
            continue
        bucket = accum.setdefault(branch, {})
        for key, val in evaluation_results_stat[branch].items():
            bucket[key] = bucket.get(key, 0.0) + val

    if getattr(args, "eval_mode", None) == "pixel_fusion":
        return

    for key in evaluation_results_stat.get("RGB", {}):
        accum.setdefault("rgb", {})[key] = accum["rgb"].get(key, 0.0) + evaluation_results_stat["RGB"][key]
    for key in evaluation_results_stat.get("Depth", {}):
        accum.setdefault("depth", {})[key] = accum["depth"].get(key, 0.0) + evaluation_results_stat["Depth"][key]

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


def payload_from_metric_accumulators(accum: dict, batch_count: int) -> dict:
    return {"accum": accum, "batch_count": batch_count}


def merge_inference_metric_payloads(payloads: list[dict]) -> dict:
    """Merge per-rank metric sums (same math as single-GPU ``finalize_metrics``)."""
    if not payloads:
        raise RuntimeError("No validation payloads to merge.")

    merged: dict[str, dict[str, float]] | None = None
    total_batches = 0
    for payload in payloads:
        total_batches += int(payload.get("batch_count", 0))
        src = payload["accum"]
        if merged is None:
            merged = {sec: dict(vals) for sec, vals in src.items()}
            continue
        for section, values in src.items():
            bucket = merged.setdefault(section, {})
            for key, val in values.items():
                bucket[key] = bucket.get(key, 0.0) + val

    if total_batches == 0:
        raise RuntimeError("Validation dataloader is empty on all ranks.")
    assert merged is not None
    return finalize_metrics(merged, total_batches)


def finalize_metrics(accum: dict, batch_idx: int) -> dict:
    if batch_idx == 0:
        raise RuntimeError("Validation dataloader is empty; cannot compute metrics.")
    results = {}
    for section, values in accum.items():
        if not values:
            continue
        if section in INFER_BRANCH_NAMES:
            averaged = {k: values[k] / batch_idx for k in INFER_BRANCH_METRIC_KEYS if k in values}
            results[section] = averaged
        else:
            results[section] = {k: v / batch_idx for k, v in values.items()}
    return results


def wandb_logs_from_metrics(metrics: dict, args) -> dict:
    logs = {}
    for branch in INFER_BRANCH_NAMES:
        for k, v in metrics.get(branch, {}).items():
            logs[f"val/{branch}/{k}"] = float(v)
    for k, v in metrics.get("rgb", {}).items():
        logs[f"val/rgb/{k}"] = float(v)
    for k, v in metrics.get("depth", {}).items():
        logs[f"val/depth/{k}"] = float(v)
    conf_fusion = bool(getattr(args, "conf_pixel_level_fusion", False))
    for section in (
        "conf",
        "conf_fused",
        "conf_stage1",
        "conf_stage2",
        "conf_2view",
        "conf_pseudo_multiview",
        "conf_gs_fused",
    ):
        if section not in metrics:
            continue
        prefix = f"val/{section}"
        if section == "conf" and conf_fusion:
            prefix = "val/conf_fused"
        for k, v in metrics[section].items():
            logs[f"{prefix}/{k}"] = float(v)
    return logs
