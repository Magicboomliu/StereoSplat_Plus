#!/usr/bin/env python3
"""Conf-only Stage2 fine-tuning (route A).

Thin wrapper: forces --conf_tune and --self_pseudo, then delegates to the
standard Stage2 trainer.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    argv = list(sys.argv)
    if "--conf_tune" not in argv:
        argv.insert(1, "--conf_tune")
    if "--self_pseudo" not in argv:
        argv.insert(1, "--self_pseudo")
    sys.argv = argv
    runpy.run_path(
        str(Path(__file__).resolve().parent / "train_kitti360_stereosplat_stage2_with_difix3d.py"),
        run_name="__main__",
    )
