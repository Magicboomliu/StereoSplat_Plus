StereoSplat Plus (Pixi Version)

This repository contains three commonly used components:

- **StereoSplat training and inference/evaluation** (`stereosplat/`)
- **StereoSplat Plus inference/evaluation with Difix3D restoration**
  (`stereosplat/validator/...plus_diffix.py` + `stereosplat/scripts/evaluation/stereosplat_plus/`)
- **Difix3D / image restoration training and inference/evaluation**
  (`stereosplat/difix3d/` and related scripts/code under `difix3d/`)

> Important: run scripts with `bash script.sh` or `./script.sh`. **Do not** use `sh script.sh` (it may break due to bash-incompatible syntax).

## Environment (Pixi)

The main pixi environment for this repository is defined in:

- `stereosplat/pyproject.toml` (recommended: `-e cu118`)

For concrete training/inference commands, data paths, and parameter descriptions, please refer to the READMEs in each subfolder:

- **StereoSplat / StereoSplat Plus**: `stereosplat/README.md` (add one there if missing)
- **Difix3D / Image Restoration**: `stereosplat/difix3d/README.md` (add one there if missing)
