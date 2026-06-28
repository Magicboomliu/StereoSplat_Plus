# StereoSplat Plus (Pixi)

Feed-forward 3D Gaussian Splatting for autonomous driving scenes (KITTI-360).

This repository is the **confidence-enabled StereoSplat+** release: **15D Gaussians** (14 geometry/appearance + 1 confidence) with a custom rasterizer. Training and evaluation are built around the conf model end-to-end.

---

## Components

| Component | Description |
|-----------|-------------|
| **StereoSplat (conf)** | Stage 1: all GT views + confidence supervision |
| **StereoSplat+ (Stage 2)** | Self-pseudo training with optional Difix3D enhancement |
| **StereoSplat+ inference** | `stereosplat_plus`: pose injection + `pseudo_ratio` |
| **Pixel-level fusion** | `pixel_fusion`: per-pixel conf fusion of two renders (optional `--conf_fusion_margin`) |
| **GS voxel fusion** | `--gs_conf_fusion`: fuse G_base and G_plus in 3D voxels, then render once |
| **GS + pixel** | `--gs_conf_fusion` + `--conf_pixel_level_fusion`: GS fusion then pixel fusion |
| **Oracle upper bound** | `--use_gt_view`: GT pixel-wise pick between G_base / G_plus (ablation) |
| **Difix3D** | Reference-guided pseudo-view refinement (`difix3d/` standalone Pixi sub-project) |

---

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

---

## Evaluation modes

| `eval_mode` | Meaning | Example script |
|-------------|---------|----------------|
| `stereosplat` | 2-view GT → single forward → render metrics | `evaluation/evaluations/stereosplat.sh` |
| `stereosplat_plus` | Full-trajectory render → `pseudo_ratio` pseudo views → optional Difix → reinject → second forward | `evaluation/evaluations/stereosplat_plus.sh` (uses pixel fusion flags) |
| `pixel_fusion` | Fuse two renders per-pixel by confidence | via `eval/run.py` / `eval/run_multi_gpu.py` CLI |

Unified stack: `eval/run_multi_gpu.py` or `eval/run.py` → `eval/routes.py` → `infer_*` in `stereosplat.py`.

**Fusion extensions** (CLI on `pixel_fusion` + `whole`):

- **Pixel A1**: `--conf_pixel_level_fusion --conf_fusion_margin 0.05` (prefer base on ties)
- **GS**: `--gs_conf_fusion` + `--gs_fusion_voxel_size` / `--gs_fusion_margin` / `--gs_fusion_conf_agg`
- **Combined**: enable both GS and pixel fusion flags

**`pseudo_ratio`** (shared by `stereosplat_plus` and `pixel_fusion`): `--pseudo_ratio 0.5 1.0` selects center stereo then last stereo as pseudo views (default). If omitted, `eval/run.py` fills `[0.5, 1.0]` automatically.

Model entry points: `infer_stereosplat_two_gt_views_forward`, `infer_stereosplat_plus_pose_injection_single_model`, `infer_pixel_fusion_pose_injection_single_model`, etc.

---

## Documentation

| Doc | Contents |
|-----|----------|
| **[stereosplat_conf/README.md](stereosplat_conf/README.md)** | Install, training, inference matrix, shell cheat sheet, visualization |
| **[stereosplat_conf/eval/README.md](stereosplat_conf/eval/README.md)** | Call chain, full CLI, FAQ |
| **[docker/README.md](docker/README.md)** | Pixi environment in Docker |
| **[difix3d/README.md](difix3d/README.md)** | Difix3D standalone train/eval |

---

## Layout

```
StereoSplat_Plus/
├── README.md
├── stereosplat_conf/                 # Main project (train + eval + model)
│   ├── eval/                         # eval/run.py, eval/run_multi_gpu.py
│   ├── trainer/
│   │   ├── train_kitti360_stereosplat_with_conf.py
│   │   └── train_kitti360_stereosplat_plus_with_difix3d.py
│   ├── scripts/
│   │   ├── train/complete/           # train_stereosplat*.sh
│   │   ├── train/ablations/
│   │   └── evaluation/
│   │       ├── evaluations/          # camera-ready eval
│   │       └── ablations/
│   ├── tests/                        # pytest for stereosplat helpers
│   └── src/stereosplat/
├── difix3d/
└── docker/
```
