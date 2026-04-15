# StereoSplat (Pixi)

## Requirements

- Linux (`linux-64`)
- `git`

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
sh train.sh
```

Notes:
- The script uses `pixi run -e cu118 ...` internally.
- Optional overrides (paths, `exp_name`, W&B settings, etc.) are defined at the top of `scripts/train/stereosplat/train.sh`.

## Validation (render inside bin)

We provide a simple validation/evaluation entrypoint script that renders forward views and writes averaged metrics to a json file.

```bash
cd stereosplat/scripts/evaluation/stereosplat
sh render_inside_bin.sh
```

Notes:
- Edit paths (config/output/filelists/checkpoint) at the top of `scripts/evaluation/stereosplat/render_inside_bin.sh`.
- Metrics are saved to `--output_folder/metric.json` (when not using `--output_vis`).