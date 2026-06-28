"""Helpers for conf_tune (split conf_head) inference."""
from __future__ import annotations

from typing import Callable, Iterable, NamedTuple


class LoadCheckResult(NamedTuple):
    ok: bool
    message: str


def cfg_uses_split_conf_head(cfg) -> bool:
    cv = cfg.model.costvolume_gs.get("gaussain_head_kwargs", {})
    vol = cfg.model.volume_gs.get("gs_decoder", {})
    return bool(cv.get("use_split_conf_head")) and bool(vol.get("use_split_conf_head"))


def _has_any(keys: Iterable[str], needles: tuple[str, ...]) -> bool:
    return any(any(n in k for n in needles) for k in keys)


def validate_split_conf_load(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    log: Callable[[str], None] | None = print,
) -> LoadCheckResult:
    """Fail fast when unified ckpt is loaded into a split-head model (or vice versa)."""
    unified_markers = ("gaussian_head.", "gs_decoder.gs_decoder.")
    split_markers = (
        "shared_hidden.",
        "rgb_geom_head.",
        "conf_head.",
        "gs_rgb_decoder.",
        "conf_decoder.",
    )

    missing = list(missing_keys)
    unexpected = list(unexpected_keys)

    if _has_any(missing, unified_markers):
        msg = (
            "Split-head model but checkpoint looks unified "
            f"(missing e.g. {missing[:3]}). "
            "Use conf_tune weights or run migrate before inference."
        )
        if log:
            log(f"[ConfTuneInfer] ERROR: {msg}")
        return LoadCheckResult(False, msg)

    if _has_any(unexpected, unified_markers):
        msg = (
            "Unified-head checkpoint loaded into split-head model "
            f"(unexpected e.g. {list(unexpected)[:3]}). "
            "Use run_multi_gpu_tune.py + conf_tune config/weights."
        )
        if log:
            log(f"[ConfTuneInfer] ERROR: {msg}")
        return LoadCheckResult(False, msg)

    critical = [
        k for k in missing
        if ("conf_head." in k or "conf_decoder." in k or "shared_hidden." in k
                or "rgb_geom_head." in k or "gs_rgb_decoder." in k)
    ]
    if critical:
        msg = f"Split-head checkpoint incomplete; missing {critical[:5]}"
        if log:
            log(f"[ConfTuneInfer] ERROR: {msg}")
        return LoadCheckResult(False, msg)

    if log:
        log(
            "[ConfTuneInfer] split conf_head load OK "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    return LoadCheckResult(True, "ok")
