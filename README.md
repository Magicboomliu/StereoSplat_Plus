# StereoSplat Plus (Pixi)

Feed-forward 3D Gaussian Splatting for autonomous driving scenes (KITTI-360).  
This repository contains three components:

| Component | Description |
|-----------|-------------|
| **(1) StereoSplat** | Feed-forward dual-branch 3DGS training and evaluation |
| **(2) StereoSplat-Conf** | StereoSplat with per-Gaussian confidence supervision (Path 3: conf-guided fusion) |
| **(3) StereoSplat+** | StereoSplat inference enhanced with Difix3D image restoration |
| **(4) Difix3D** | Reference-guided image restoration model training and evaluation |

---

## Quick Start

```bash
cd stereosplat
pixi install -e cu118          # install all Python deps (accelerate, torch, wandb, ...)
pixi run -e cu118 setup        # compile rasterizer + install mmcv/mmdet3d
bash scripts/train/stereosplat/train.sh   # train StereoSplat-Conf (Model 1 with conf)
```

See [`stereosplat/README.md`](stereosplat/README.md) for full documentation.
