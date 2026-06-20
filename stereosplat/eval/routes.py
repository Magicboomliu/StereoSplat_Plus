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


def make_training_val_eval_args(cfg, output_folder: str, architecture: str):
    """Namespace mirroring eval/run.py pixel_fusion flags for training validation."""
    from types import SimpleNamespace

    fusion_sup = getattr(cfg.model, "losses_params", None)
    fusion_mode = "soft"
    if fusion_sup is not None:
        fusion_sup = getattr(fusion_sup, "fusion_sup_dict", None)
    if fusion_sup is not None:
        fusion_mode = getattr(fusion_sup, "val_fusion_mode", "soft")
    return SimpleNamespace(
        eval_mode="pixel_fusion",
        architecture=architecture,
        conf_pixel_level_fusion=True,
        conf_fusion_margin=0.0,
        fusion_mode=str(fusion_mode),
        fusion_first_margin=999.0,
        fusion_center_margin=0.0,
        fusion_last_margin=0.0,
        fusion_calibration="none",
        fusion_temperature=None,
        gs_conf_fusion=False,
        gs_fusion_voxel_size=0.1,
        gs_fusion_margin=0.05,
        gs_fusion_conf_agg="mean",
        gs_fusion_base_conf_thresh=None,
        output_folder=output_folder,
        output_vis=False,
        use_gt_view=False,
        use_diffix3d=True,
        use_ref=True,
        pseudo_ratio=[0.5, 1.0],
    )


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

FUSION_BRANCH_RGB_KEYS = {
    "RGB_2view": "2view",
    "RGB_pseudo_multiview": "pseudo_multiview",
    "RGB": "pseudo_fused",
}

FUSION_BRANCH_DEPTH_KEYS = {
    "Depth_2view": "2view",
    "Depth_pseudo_multiview": "pseudo_multiview",
    "Depth": "pseudo_fused",
}

PSNR_SUMMARY_KEYS = {
    "first_view_psnr_average": "psnr_first",
    "center_view_psnr_average": "psnr_center",
    "last_view_psnr_average": "psnr_last",
    "all_view_psnr_average": "psnr_mean",
}


def _init_fusion_branch_accumulators(accum: dict) -> None:
    for branch in ("2view", "pseudo_multiview", "pseudo_fused"):
        accum[branch] = {"rgb": {}, "depth": {}}


def _fusion_metric_summary_from_rgb(rgb_metrics: dict) -> dict:
    return {
        dst: rgb_metrics[src]
        for src, dst in PSNR_SUMMARY_KEYS.items()
        if src in rgb_metrics
    }


def init_metric_accumulators(eval_mode: str, architecture: str, conf_fusion: bool) -> dict:
    accum: dict[str, dict] = {
        "rgb": {},
        "depth": {},
    }
    if eval_mode == "pixel_fusion":
        _init_fusion_branch_accumulators(accum)
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
        accum["conf_gs_fused"] = {}
    return accum


def accumulate_batch_metrics(accum: dict, evaluation_results_stat: dict, args) -> None:
    eval_mode = getattr(args, "eval_mode", None)
    for key in evaluation_results_stat.get("RGB", {}):
        accum["rgb"][key] = accum["rgb"].get(key, 0.0) + evaluation_results_stat["RGB"][key]
        if eval_mode == "pixel_fusion" and "pseudo_fused" in accum:
            accum["pseudo_fused"]["rgb"][key] = (
                accum["pseudo_fused"]["rgb"].get(key, 0.0)
                + evaluation_results_stat["RGB"][key]
            )
    for key in evaluation_results_stat.get("Depth", {}):
        accum["depth"][key] = accum["depth"].get(key, 0.0) + evaluation_results_stat["Depth"][key]
        if eval_mode == "pixel_fusion" and "pseudo_fused" in accum:
            accum["pseudo_fused"]["depth"][key] = (
                accum["pseudo_fused"]["depth"].get(key, 0.0)
                + evaluation_results_stat["Depth"][key]
            )

    if eval_mode == "pixel_fusion":
        for src_key, branch in FUSION_BRANCH_RGB_KEYS.items():
            if src_key == "RGB":
                continue
            if src_key not in evaluation_results_stat or branch not in accum:
                continue
            for key, val in evaluation_results_stat[src_key].items():
                accum[branch]["rgb"][key] = accum[branch]["rgb"].get(key, 0.0) + val
        for src_key, branch in FUSION_BRANCH_DEPTH_KEYS.items():
            if src_key == "Depth":
                continue
            if src_key not in evaluation_results_stat or branch not in accum:
                continue
            for key, val in evaluation_results_stat[src_key].items():
                accum[branch]["depth"][key] = accum[branch]["depth"].get(key, 0.0) + val

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

    def _avg_section(values: dict) -> dict:
        return {k: v / batch_idx for k, v in values.items()}

    results: dict = {}
    fusion_metric: dict = {}

    if "2view" in accum:
        for branch in ("2view", "pseudo_multiview", "pseudo_fused"):
            branch_accum = accum.get(branch)
            if not branch_accum:
                continue
            branch_results = {}
            if branch_accum.get("rgb"):
                branch_results["rgb"] = _avg_section(branch_accum["rgb"])
                fusion_metric[branch] = _fusion_metric_summary_from_rgb(branch_results["rgb"])
            if branch_accum.get("depth"):
                branch_results["depth"] = _avg_section(branch_accum["depth"])
            if branch_results:
                results[branch] = branch_results
        if fusion_metric:
            results["fusion_metric"] = fusion_metric

    for section, values in accum.items():
        if section in ("2view", "pseudo_multiview", "pseudo_fused"):
            continue
        if not values:
            continue
        results[section] = _avg_section(values)

    if "pseudo_fused" in results and "rgb" not in results:
        results["rgb"] = results["pseudo_fused"].get("rgb", {})
    if "pseudo_fused" in results and "depth" not in results:
        results["depth"] = results["pseudo_fused"].get("depth", {})

    return results


def merge_metric_dicts(target: dict, source: dict) -> None:
    """Recursively sum scalar leaves in nested metric accumulators."""
    for key, val in source.items():
        if isinstance(val, dict):
            bucket = target.setdefault(key, {})
            merge_metric_dicts(bucket, val)
        else:
            target[key] = target.get(key, 0.0) + float(val)


def all_gather_rank_metrics(
    accum: dict,
    batch_count: int,
) -> list[tuple[dict, int]]:
    """Collect per-rank (accum, batch_count) via ``all_gather_object``."""
    import torch.distributed as dist

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


def merge_gathered_metrics(
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
            merge_metric_dicts(bucket, values)

    if not merged:
        raise RuntimeError(
            "No metric keys accumulated on any rank; check eval_mode / filelist."
        )

    return finalize_metrics(merged, total_batches)


def wandb_logs_from_metrics(metrics: dict, args) -> dict:
    logs = {}
    eval_mode = getattr(args, "eval_mode", None)
    if eval_mode == "pixel_fusion":
        for branch in ("2view", "pseudo_multiview", "pseudo_fused"):
            branch_metrics = metrics.get(branch, {})
            for metric_type in ("rgb", "depth"):
                for k, v in branch_metrics.get(metric_type, {}).items():
                    logs[f"val/{branch}/{metric_type}/{k}"] = float(v)
        for branch, branch_metrics in metrics.get("fusion_metric", {}).items():
            for k, v in branch_metrics.items():
                logs[f"val/fusion_metric/{branch}/{k}"] = float(v)
    else:
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
