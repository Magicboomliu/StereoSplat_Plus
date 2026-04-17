# StereoSplat (Pixi)

## Requirements

- Linux (`linux-64`)
- `git`
- A NVIDIA GPU(>12G)

## Install Pixi

Install Pixi (official installer), then verify it is on your `PATH`.

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi --version
```

## Build the environment

Create/sync the Pixi environment for this project, then run the one-shot setup (installs PyTorch/CUDA deps, builds rasterizer, and installs Python requirements).

```bash
cd stereosplat
pixi install
pixi run setup
```

## Train StereoSplat (KITTI-360)

We provide a simple training entrypoint script.

```bash
cd stereosplat/scripts/train/stereosplat
bash train.sh
```

Notes:
- The script uses `pixi run -e cu118 ...` internally.
- Optional overrides (paths, `exp_name`, W&B settings, etc.) are defined at the top of `scripts/train/stereosplat/train.sh`.

## Validation 

- evaluations feedforward 

```bash
cd stereosplat/scripts/evaluation/stereosplat
bash render_inside_bin.sh
```

## StereoSplat Plus inference / evaluation (with Difix3D restoration)

This entrypoint runs the StereoSplat validation pipeline with optional Difix3D-based restoration.

```bash
cd stereosplat/scripts/evaluation/stereosplat_plus
bash render_inside_bin.sh
```

Notes:
- This script uses `pixi run -e cu118 ...` internally.
- Optional flags can be appended in the script (e.g., `--use_diffix3d`, `--use_ref`, `--output_vis`,
  `--deterministic_vae_encode`, `--deterministic_scheduler_step`).