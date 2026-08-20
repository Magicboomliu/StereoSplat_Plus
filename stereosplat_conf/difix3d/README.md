# Difix3D (Image Restoration)

This folder contains Difix3D training and evaluation scripts used by **StereoSplat Plus**.

StereoSplat 评估里通过 `eval/run.py` 的 `--use_diffix3d` 加载本目录权重；S+ / pixel_fusion 的 `pseudo_ratio` 与 Shell 见 **[../../README.md](../../README.md)**。

## Pixi installation (recommended)

Difix3D is typically run inside the **StereoSplat pixi environment** (CUDA 11.8 / `cu118`).

```bash
cd /path/to/StereoSplat_Plus/stereosplat_conf
pixi install -e cu118
```

If you run commands from a different working directory, you can force the correct pixi manifest:

```bash
pixi run --manifest-path /path/to/StereoSplat_Plus/stereosplat_conf/pyproject.toml -e cu118 <cmd>
```

> Important: run scripts with `bash script.sh` or `./script.sh`. Do **not** use `sh script.sh`.

## Training

Script:

- `scripts/train_difix3d.sh`

Run:

```bash
cd /path/to/StereoSplat_Plus/stereosplat_conf
bash difix3d/scripts/train_difix3d.sh
```

Edit the script to set dataset paths, output directory, checkpoints, and GPU ids.

Difix3D finetuning expects a dataset manifest JSON (`training` / `test` splits with image paths). See `filenames/Validation_Set/all_results_dict.example.json` for the schema; set `DIFIX_DATASET_JSON` to your generated file (do not commit machine-specific paths).

## Inference / Evaluation

Script:

- `scripts/eval_difix3d.sh`

Run:

```bash
cd /path/to/StereoSplat_Plus/stereosplat_conf
bash difix3d/scripts/eval_difix3d.sh
```

Edit the script to set:

- `--dataset_path`
- `--pretrained_path` (e.g., `model_50001.pkl`)
- output JSON path
