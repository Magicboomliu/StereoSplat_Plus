"""Multi-GPU Stage1 validation helpers (fusion / volume / cv branches).

Each branch reports per-frame stereo-averaged RGB metrics:
  psnr_first, psnr_center, psnr_last, psnr_mean
  ssim_first, ssim_center, ssim_last, ssim_mean

``psnr_mean`` / ``ssim_mean`` pool all first+center+last samples (same as Stage2 val).
"""
from __future__ import annotations

from typing import Any

import torch

_VIEW_KEYS = {
    "center": "center_view",
    "first": "first_view",
    "last": "last_view",
}
_FRAME_KEYS = ("first", "center", "last", "all")


def new_stage1_rgb_accumulator() -> dict[str, dict[str, list[float]]]:
    return {
        "psnr": {k: [] for k in _FRAME_KEYS},
        "ssim": {k: [] for k in _FRAME_KEYS},
    }


def new_stage1_metric_accumulators() -> dict[str, Any]:
    return {
        "fusion": new_stage1_rgb_accumulator(),
        "volume": new_stage1_rgb_accumulator(),
        "cv": new_stage1_rgb_accumulator(),
        "conf_sum": 0.0,
        "conf_count": 0,
    }


def _resolve_model(model) -> Any:
    return model.module if hasattr(model, "module") else model


def _lr_avg_rgb_view(rgb_dict: dict, view_key: str) -> tuple[float, float]:
    view = rgb_dict[view_key]
    psnr = (view["left"]["psnr"] + view["right"]["psnr"]) / 2.0
    ssim = (view["left"]["ssim"] + view["right"]["ssim"]) / 2.0
    return psnr, ssim


def accumulate_batch_stage1_rgb(
    acc_rgb: dict[str, dict[str, list[float]]],
    rgb_dict: dict,
) -> None:
    for view_short, view_key in _VIEW_KEYS.items():
        if view_key not in rgb_dict:
            continue
        psnr, ssim = _lr_avg_rgb_view(rgb_dict, view_key)
        acc_rgb["psnr"][view_short].append(psnr)
        acc_rgb["ssim"][view_short].append(ssim)
        acc_rgb["psnr"]["all"].append(psnr)
        acc_rgb["ssim"]["all"].append(ssim)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def section_from_rgb_accumulator(acc_rgb: dict[str, dict[str, list[float]]]) -> dict:
    return {
        "psnr_first": _mean(acc_rgb["psnr"]["first"]),
        "psnr_center": _mean(acc_rgb["psnr"]["center"]),
        "psnr_last": _mean(acc_rgb["psnr"]["last"]),
        "psnr_mean": _mean(acc_rgb["psnr"]["all"]),
        "ssim_first": _mean(acc_rgb["ssim"]["first"]),
        "ssim_center": _mean(acc_rgb["ssim"]["center"]),
        "ssim_last": _mean(acc_rgb["ssim"]["last"]),
        "ssim_mean": _mean(acc_rgb["ssim"]["all"]),
    }


def payload_from_stage1_accumulators(accum: dict, batch_count: int) -> dict:
    return {
        "accum": {
            "fusion": accum["fusion"],
            "volume": accum["volume"],
            "cv": accum["cv"],
        },
        "conf_sum": accum["conf_sum"],
        "conf_count": accum["conf_count"],
        "batch_count": batch_count,
    }


def merge_stage1_metric_payloads(payloads: list[dict]) -> dict[str, dict]:
    if not payloads:
        raise RuntimeError("No validation payloads to merge.")

    merged = new_stage1_metric_accumulators()
    total_batches = 0

    for payload in payloads:
        total_batches += int(payload.get("batch_count", 0))
        merged["conf_sum"] += float(payload.get("conf_sum", 0.0))
        merged["conf_count"] += int(payload.get("conf_count", 0))

        for stage in ("fusion", "volume", "cv"):
            src = payload["accum"][stage]
            dst = merged[stage]
            for metric in ("psnr", "ssim"):
                for frame in _FRAME_KEYS:
                    dst[metric][frame].extend(src[metric][frame])

    if total_batches == 0:
        raise RuntimeError("Validation dataloader is empty on all ranks.")

    fusion = section_from_rgb_accumulator(merged["fusion"])
    if merged["conf_count"] > 0:
        fusion["mean_conf"] = merged["conf_sum"] / merged["conf_count"]

    return {
        "fusion": fusion,
        "volume": section_from_rgb_accumulator(merged["volume"]),
        "cv": section_from_rgb_accumulator(merged["cv"]),
    }


def gather_stage1_validation_payloads(accelerator, local_payload: dict) -> list[dict]:
    if accelerator.num_processes == 1:
        return [local_payload]
    import torch.distributed as dist

    gathered: list[dict | None] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, local_payload)
    return [p for p in gathered if p is not None]


@torch.no_grad()
def accumulate_batch_stage1_metrics(
    accum: dict,
    model,
    batch,
    cfg,
    *,
    global_iter: int = 0,
    val_batch_save_dir: str | None = None,
    save_visuals: bool = False,
) -> None:
    """Run ``validation_step`` and append RGB metrics for fusion/volume/cv."""
    _model = _resolve_model(model)
    _save_dir = val_batch_save_dir or ""

    metrics_rgb_list, _, _ = _model.validation_step(
        batch,
        _save_dir,
        cfg,
        save_visuals=save_visuals,
    )

    stage_by_index = {0: "fusion", 1: "volume", 2: "cv"}
    for idx, rgb_dict in enumerate(metrics_rgb_list):
        stage = stage_by_index.get(idx)
        if stage is None:
            continue
        accumulate_batch_stage1_rgb(accum[stage], rgb_dict)

    try:
        _, _, fusion_list, _, _ = _model.forward(
            batch,
            "train",
            view_num=2,
            matching_nums=2,
            iter=global_iter,
            cfg=cfg,
        )
        if len(fusion_list) > 2:
            conf = fusion_list[2]
            accum["conf_sum"] += conf.detach().float().mean().item()
            accum["conf_count"] += 1
            if save_visuals and val_batch_save_dir:
                import matplotlib.cm as cm
                import PIL.Image

                conf_save_path = f"{val_batch_save_dir}/conf_map.png"
                sample = conf[0, 0].float().cpu().clamp(0, 1).numpy()
                colored = (cm.viridis(sample)[:, :, :3] * 255).astype("uint8")
                PIL.Image.fromarray(colored).save(conf_save_path)
    except Exception:
        pass
