"""Multi-GPU inference for conf_tune (split conf_head) Stage2 checkpoints.

Same inference path as ``eval/run_multi_gpu.py`` (``infer_pixel_fusion_pose_injection_single_model``),
but:
  - defaults ``--config_path`` to input_invariant_stereosplat_stage2_conf_tune.py
  - enables ``--split_conf_head`` (build split model + strict load validation)

The forward / fusion code in stereosplat.py is unchanged; only the Gaussian head
module layout differs (rgb_geom_head + conf_head vs unified gaussian_head).
"""
from __future__ import annotations

import sys
from pathlib import Path

_STEREOSPLAT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    _STEREOSPLAT_ROOT
    / "src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2_conf_tune.py"
)


def _argv_with_conf_tune_defaults(argv: list[str]) -> list[str]:
    out = list(argv)
    has_config = any(
        a == "--config_path" or a.startswith("--config_path=") for a in out[1:]
    )
    if not has_config:
        out[1:1] = ["--config_path", str(_DEFAULT_CONFIG)]
    if "--split_conf_head" not in out:
        out.append("--split_conf_head")
    return out


if __name__ == "__main__":
    from eval.run_multi_gpu import main

    sys.argv = _argv_with_conf_tune_defaults(sys.argv)
    main(sys.argv[1:])
