"""Training-aligned Stage2 fusion validation (fusion_metric.json).

Mirrors ``train_kitti360_stereosplat_stage2_with_difix3d.py`` validation:
  Pass 1 — ``validation_step`` with view_num=2 (2view PSNR, L/R average)
  Pass 2 — ``forward_stage2_with_difix3d`` mode='val', view_num=6 (pseudo mv + fused)
"""
from __future__ import annotations

from typing import Any

import torch


def new_fusion_metric_accumulators() -> tuple[dict, dict, dict]:
    def _new_acc() -> dict[str, list[float]]:
        return {v: [] for v in ("first", "center", "last", "all")}

    return _new_acc(), _new_acc(), _new_acc()


def _resolve_model(model) -> Any:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def accumulate_batch_fusion_metrics(
    acc_2view: dict,
    acc_pseudo_mv: dict,
    acc_pseudo_fus: dict,
    model,
    batch,
    cfg,
    *,
    frozen_stage_1_model=None,
    pretrained_diffix_model=None,
    self_pseudo: bool = True,
    val_batch_save_dir: str | None = None,
    save_visuals: bool = False,
) -> None:
    """Append one val-batch PSNR samples into the three accumulators."""
    _model = _resolve_model(model)
    _save_dir = val_batch_save_dir or ""

    m2_rgb, _, _ = _model.validation_step(
        batch,
        _save_dir,
        cfg,
        view_num=2,
        matching_nums=2,
        save_visuals=save_visuals,
    )
    _r2 = m2_rgb[0] if m2_rgb else {}
    for _view in ("first", "center", "last"):
        _vk = f"{_view}_view"
        if _vk not in _r2:
            continue
        _p = (
            _r2[_vk]["left"]["psnr"] + _r2[_vk]["right"]["psnr"]
        ) / 2.0
        acc_2view[_view].append(_p)
        acc_2view["all"].append(_p)

    _prog_frozen = None if self_pseudo else frozen_stage_1_model
    _prog_out = _model.forward_stage2_with_difix3d(
        batch,
        "val",
        view_num=6,
        matching_nums=3,
        iter=0,
        frozen_stage_1_model=_prog_frozen,
        pretrained_diffix_model=pretrained_diffix_model,
        mix_psuedo_views_ratio=1.0,
        mix_difix3d_ratio=1.0,
        use_self_for_pseudo=self_pseudo,
        cfg=cfg,
    )
    _pl = _prog_out[1]
    for _view in ("first", "center", "last"):
        _mv_k = f"val/psnr_multiview_{_view}"
        _fus_k = f"val/psnr_fused_{_view}"
        if _mv_k in _pl:
            acc_pseudo_mv[_view].append(_pl[_mv_k])
            acc_pseudo_mv["all"].append(_pl[_mv_k])
        if _fus_k in _pl:
            acc_pseudo_fus[_view].append(_pl[_fus_k])
            acc_pseudo_fus["all"].append(_pl[_fus_k])


def _mean(lst: list[float]) -> float | None:
    return sum(lst) / len(lst) if lst else None


def section_from_accum(acc: dict) -> dict[str, float | None]:
    return {
        "psnr_first": _mean(acc["first"]),
        "psnr_center": _mean(acc["center"]),
        "psnr_last": _mean(acc["last"]),
        "psnr_mean": _mean(acc["all"]),
    }


def fusion_metric_from_accumulators(
    acc_2view: dict,
    acc_pseudo_mv: dict,
    acc_pseudo_fus: dict,
) -> dict:
    return {
        "2view": section_from_accum(acc_2view),
        "pseudo_multiview": section_from_accum(acc_pseudo_mv),
        "pseudo_fused": section_from_accum(acc_pseudo_fus),
    }


def merge_fusion_metric_payloads(payloads: list[dict]) -> dict:
    """Merge per-rank accumulators (concat lists, then global mean)."""
    if not payloads:
        raise RuntimeError("No validation payloads to merge.")

    acc_2view, acc_mv, acc_fus = new_fusion_metric_accumulators()
    for payload in payloads:
        for key, dst in (
            ("acc_2view", acc_2view),
            ("acc_pseudo_mv", acc_mv),
            ("acc_pseudo_fus", acc_fus),
        ):
            src = payload[key]
            for view in ("first", "center", "last", "all"):
                dst[view].extend(src[view])

    total_batches = sum(int(p.get("batch_count", 0)) for p in payloads)
    if total_batches == 0:
        raise RuntimeError("Validation dataloader is empty on all ranks.")

    return fusion_metric_from_accumulators(acc_2view, acc_mv, acc_fus)


def payload_from_accumulators(
    acc_2view: dict,
    acc_pseudo_mv: dict,
    acc_pseudo_fus: dict,
    batch_count: int,
) -> dict:
    return {
        "acc_2view": acc_2view,
        "acc_pseudo_mv": acc_pseudo_mv,
        "acc_pseudo_fus": acc_pseudo_fus,
        "batch_count": batch_count,
    }
