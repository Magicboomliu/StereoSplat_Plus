# StereoSplat Plus

Feed-forward 3D Gaussian Splatting for autonomous driving scenes (KITTI-360).

### Dataset Preparation

- [KITTI360 Datasets (Original)](https://www.cvlibs.net/datasets/kitti-360/)

- Processed KITTI360 Downalod
    - [Bin Splits Files 8 meter](https://www.cvlibs.net/datasets/kitti-360/): including camera instrinsics and extrinsics as well as images and depth paths for each timestamp.
    - [Processed images & depths & Point Clouds](x): processed images and depths and sparse depth map captured from raw LiDAR from KITTI360



## Quick Start

```bash
cd stereosplat_conf
pixi install -e cu118          # Python deps (accelerate, torch, wandb, ...)
pixi run -e cu118 setup        # Build diff-gaussian-rasterization-conf (required)
```

**Training** (edit checkpoint / data paths inside the scripts first):

```bash
# Stage 1 — StereoSplat with conf
bash scripts/train/complete/train_stereosplat.sh

# Stage 2 — StereoSplat+ with Difix3D + self-pseudo
bash scripts/train/complete/train_stereosplat_plus.sh
```

**Evaluation** (camera-ready checkpoints; use `bash`, not `sh`):

```bash
# 2-view baseline (--eval_mode stereosplat)
bash scripts/evaluation/evaluations/stereosplat.sh

# Pixel fusion + Difix3D + self-pseudo (--conf_pixel_level_fusion)
bash scripts/evaluation/evaluations/stereosplat_plus.sh
```

Ablation variants live under `scripts/train/ablations/` and `scripts/evaluation/ablations/`.

**Unit tests** (core `stereosplat.py` helpers):

```bash
pixi run -e cu118 test-stereosplat
```
