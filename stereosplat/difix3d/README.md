# Difix3D (Image Restoration)

This folder contains Difix3D training and evaluation scripts used by **StereoSplat Plus**.

## Pixi installation (recommended)

Difix3D is typically run inside the **StereoSplat pixi environment** (CUDA 11.8 / `cu118`).

```bash
cd /path/to/StereoSplat_Plus/stereosplat
pixi install -e cu118
```

If you run commands from a different working directory, you can force the correct pixi manifest:

```bash
pixi run --manifest-path /path/to/StereoSplat_Plus/stereosplat/pyproject.toml -e cu118 <cmd>
```

> Important: run scripts with `bash script.sh` or `./script.sh`. Do **not** use `sh script.sh`.

## Training

Script:

- `scripts/train_difix3d.sh`

Run:

```bash
cd /path/to/StereoSplat_Plus/stereosplat
bash difix3d/scripts/train_difix3d.sh
```

Edit the script to set dataset paths, output directory, checkpoints, and GPU ids.

## Inference / Evaluation

Script:

- `scripts/eval_difix3d.sh`

Run:

```bash
cd /path/to/StereoSplat_Plus/stereosplat
bash difix3d/scripts/eval_difix3d.sh
```

Edit the script to set:

- `--dataset_path`
- `--pretrained_path` (e.g., `model_50001.pkl`)
- output JSON path
