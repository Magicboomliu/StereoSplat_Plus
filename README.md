# StereoSplat+

**[Paper (arXiv:2607.08808)](https://arxiv.org/abs/2607.08808)** · IROS 2026

Feed-forward **3D Gaussian Splatting** for autonomous driving scenes, built on KITTI-360. Official implementation of **[StereoSplat+: Feed-Forward Stereo Gaussian Splatting with Diffusion-Assisted Progressive Inference](https://arxiv.org/abs/2607.08808)** (Liu & Okutomi, IROS 2026).

This repository implements **StereoSplat with confidence (15D Gaussians)** and **StereoSplat+**, which augments stereo inputs with self-pseudo views, Difix3D enhancement, and per-pixel confidence fusion at inference time.

> **Main code lives in [`stereosplat_conf/`](stereosplat_conf/)**. See [`stereosplat_conf/README.md`](stereosplat_conf/README.md) for training/eval details and [`stereosplat_conf/eval/README.md`](stereosplat_conf/eval/README.md) for CLI flags.


## Overview

| Stage | Method | Key idea |
|-------|--------|----------|
| **Stage 1** | StereoSplat-Conf | 15D Gaussians (14 geom/appearance + 1 `conf`); custom rasterizer outputs `rendered_conf` |
| **Stage 2** | StereoSplat+ | Self-pseudo stereo views + Difix3D refinement during training; optional pixel-level conf fusion at inference |

**Inference modes**

| Mode | CLI `--eval_mode` | Description |
|------|-------------------|-------------|
| 2-view baseline | `stereosplat` | Two GT stereo views → single forward → render novel views |
| StereoSplat+ (no pixel fusion) | `stereosplat_plus` | Pseudo trajectory + Difix3D + second forward |
| **Full S+ pipeline** | `pixel_fusion` (default) | Above + per-pixel conf fusion of 2-view vs pseudo-multiview renders |

---

## Repository layout

```
StereoSplat_Plus/
├── stereosplat_conf/          # Core training, evaluation, and visualization
│   ├── src/stereosplat/       # Models, configs, metrics
│   ├── trainer/               # Stage 1 & Stage 2 training scripts
│   ├── eval/                  # run.py (single-GPU) & run_multi_gpu.py
│   ├── scripts/
│   │   ├── train/             # complete/ and ablations/
│   │   ├── evaluation/        # metric evaluation shells
│   │   └── visualization/     # RGB / depth export (--output_vis)
│   ├── difix3d/               # Difix3D code used by eval/train (imported at runtime)
│   ├── filenames/kitti360/    # Train/val/demo filelists (paths only)
│   └── diff-gaussian-rasterization-conf/  # Custom CUDA rasterizer (conf channel)
├── difix3d/                   # Standalone Pixi project (optional; duplicate of above)
└── docker/                    # Optional container setup (CUDA 11.8)
```

---

## Requirements

- **OS**: Linux (tested on Ubuntu)
- **GPU**: NVIDIA GPU with CUDA **11.8**
- **Python**: 3.10 (managed via [Pixi](https://pixi.sh))
- **Compiler**: GCC/G++ 11 (for CUDA extensions)

---

## Installation

### Option A — Pixi (recommended)

```bash
cd stereosplat_conf
pixi install -e cu118
pixi run -e cu118 setup    # Build diff-gaussian-rasterization-conf (required)
```

Run `setup` before any training or evaluation. The PyPI rasterizer does **not** support the confidence channel.

### Option B — Docker

```bash
# From repository root
docker build -t stereosplat-plus:cu118 -f docker/Dockerfile .
docker run --gpus all -it --rm -v "$PWD":/workspace/StereoSplat_Plus stereosplat-plus:cu118
```

See [`docker/README.md`](docker/README.md) for entering the Pixi environment inside the container.

---

## Data preparation

This repo does **not** ship KITTI-360 imagery or preprocessed `.bin` sequences. You need:

1. **[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)** — raw dataset (images, calibrations, LiDAR).
2. **Processed sequences** — per-scene `.bin` files listing image/depth paths and poses (8 m splits used in our experiments).
3. **Filelists** — text files under `stereosplat_conf/filenames/kitti360/` pointing to your local `.bin` paths.

Before running any script, edit **data paths** inside the shell scripts or pass them via CLI (`--datapath`, `--val_filelist`, etc.).

---

## Pretrained weights

Download checkpoints from Google Drive, place them locally, and set paths in the shell scripts or CLI.

### StereoSplat / StereoSplat+

| Model | Split | Link |
|-------|-------|------|
| StereoSplat-Conf | Complete training set | [Drive](https://drive.google.com/drive/folders/1sm3rWZ0IcgiP3dZ-XgRZgCL0f0wi-ooU?usp=sharing) |
| StereoSplat+-Conf | Complete training set | [Drive](https://drive.google.com/drive/folders/12lMwnNCBrI76M53eFrYIgWfqEA1FmVDO?usp=sharing) |
| StereoSplat-Conf | Ablations | [Drive](https://drive.google.com/drive/folders/1rk8RCTS96JV4O1iE_KJb1wcx23SF4elB?usp=sharing) |
| StereoSplat+-Conf | Ablations | [Drive](https://drive.google.com/drive/folders/1TaG-n7EjGt36VmA7ynpei4tTPS_nFKa7?usp=sharing) |

### Utility models (required for full pipeline)

| Model | Used by | Download | File to use |
|-------|---------|----------|-------------|
| **Refined Difix3D** | Stage 2 train, `stereosplat_plus`, `pixel_fusion` eval/vis | [**model_130001.pkl**](https://drive.google.com/file/d/15UOotc_7WRJ_Mg9T3g0enx9Kuh_Yn9Cm/view?usp=drive_link) (~4.9 GB) | `model_130001.pkl` |
| **UniMatch** (depth init) | Stage 1 / Stage 2 training | [**depth_estimation_224x840**](https://drive.google.com/drive/folders/1zy7PVENps22YavP2sDaNlVmBrjfBko5U?usp=drive_link) folder (~406 MB) | `checkpoint-90000/model.safetensors` |

**Refined Difix3D** — finetuned from [`nvidia/difix_ref`](https://huggingface.co/nvidia/difix_ref), step **130001**. Download the `.pkl`, save anywhere, set `--pretrained_diffix_model_path` / `--pretrained_difix3d`.

**UniMatch** — download the [depth_estimation_224x840](https://drive.google.com/drive/folders/1zy7PVENps22YavP2sDaNlVmBrjfBko5U?usp=drive_link) folder, then use `checkpoint-90000/model.safetensors` (set `--unimatch-weights-path`).

Difix3D also needs the **Hugging Face base weights** `nvidia/difix_ref` on first load (VAE / UNet backbone). Training/eval shells set `HF_HUB_OFFLINE=1`; either pre-download the model into your HF cache or remove those exports for an online first run.

Set in scripts / CLI:

- Training: `--pretrained_difix3d /path/to/model_130001.pkl`
- Eval / vis: `--pretrained_diffix_model_path /path/to/model_130001.pkl --use_ref`
- Training (UniMatch): `--unimatch-weights-path /path/to/checkpoint-90000/model.safetensors`

---

## Difix3D

Difix3D restores degraded **pseudo stereo** renders before they are re-injected into StereoSplat+.

| Location | Role |
|----------|------|
| [`stereosplat_conf/difix3d/`](stereosplat_conf/difix3d/) | **Used by StereoSplat** — `eval/run.py` imports `DifixRef` from here |
| [`difix3d/`](difix3d/) | Optional standalone Pixi project (Python 3.11) for training Difix in isolation |

**When it runs**

- **Stage 2 training** (`train_kitti360_stereosplat_plus_with_difix3d.py`): with probability `mix_difix3d_ratio` (default 0.9), pseudo views are enhanced before mixing into the batch.
- **Inference** (`pixel_fusion`, `stereosplat_plus`, `bev_plus`): trajectory render → select pseudo stereo → Difix3D (`--use_ref`) → second forward; `pixel_fusion + whole` **requires** Difix weights even without `--use_diffix3d`.

**Train / eval Difix standalone**

```bash
cd stereosplat_conf
bash difix3d/scripts/train_difix3d.sh   # finetune on paired restoration JSON
bash difix3d/scripts/eval_difix3d.sh    # PSNR/SSIM/LPIPS on validation set
```

See [`stereosplat_conf/difix3d/README.md`](stereosplat_conf/difix3d/README.md) for dataset JSON format and config (`configs/train_difix_ref.yaml`: resolution 112×544, `timestep=199`).

---

## Quick start

All commands assume `cd stereosplat_conf` and a working Pixi `cu118` environment.

### Evaluation (metrics)

Set checkpoint paths via env vars or edit each script, then run with **`bash`** (not `sh`):

| Env var | Used for |
|---------|----------|
| `STEREOSPLAT_CHECKPOINT` | StereoSplat / StereoSplat+ model weights |
| `DIFIX3D_WEIGHTS` | `model_130001.pkl` (S+ eval/vis only) |

```bash
export STEREOSPLAT_CHECKPOINT=/path/to/stereosplat_plus_conf_checkpoint
export DIFIX3D_WEIGHTS=/path/to/model_130001.pkl

# 2-view baseline
bash scripts/evaluation/evaluations/stereosplat.sh

# StereoSplat+ with pixel fusion + Difix3D + self-pseudo
bash scripts/evaluation/evaluations/stereosplat_plus.sh
```

Ablation splits: `scripts/evaluation/ablations/`.

### Visualization (RGB + depth)

Uses `eval/run.py` on a **single GPU** with `--output_vis`. Demo filelist: `filenames/kitti360/trainval/demo.txt`.

```bash
# 2-view StereoSplat
bash scripts/visualization/stereosplat.sh

# Full StereoSplat+ pipeline (pixel fusion + Difix3D)
bash scripts/visualization/stereosplat_plus.sh
```

Outputs are written under `outputs/visualization/<method>/`. Depth maps use the KITTI-style disparity colormap (see `metrics.convert_depth_to_disp`).

Optional BEV renders: `scripts/visualization/stereosplat_bev.sh`, `stereosplat_plus_bev.sh`.

### Training

Edit paths inside each script, or export env vars before running:

| Env var | Used for |
|---------|----------|
| `KITTI360_DATAPATH` | KITTI-360 root |
| `UNIMATCH_WEIGHTS` | `checkpoint-90000/model.safetensors` |
| `DIFIX3D_WEIGHTS` | `model_130001.pkl` (Stage 2 only) |
| `STAGE1_CHECKPOINT` | Stage 1 weights (Stage 2 only) |
| `STEREOSPLAT_CHECKPOINT` | Eval / visualization model weights |
| `DIFIX3D_WEIGHTS` | Eval / visualization Difix weights |
| `WANDB_API_KEY` | WandB (optional; set `use_wandb=true` in script) |

```bash
# Stage 1 — StereoSplat with confidence
bash scripts/train/complete/train_stereosplat.sh

# Stage 2 — StereoSplat+ with Difix3D + self-pseudo
bash scripts/train/complete/train_stereosplat_plus.sh
```

Ablation training: `scripts/train/ablations/`.

### Unit tests

```bash
pixi run -e cu118 test-stereosplat
```

---

## Direct CLI example

Multi-GPU evaluation:

```bash
pixi run -e cu118 accelerate launch \
  --config_file accelerate_configs/inference/multi_gpu.yaml \
  eval/run_multi_gpu.py \
  --eval_mode stereosplat \
  --architecture whole \
  --config_path src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py \
  --output_folder outputs/eval/my_run \
  --val_filelist filenames/kitti360/train_complete/val.txt \
  --pretrained_model_path /path/to/checkpoint
```

Single-GPU visualization (StereoSplat+ with Difix3D):

```bash
pixi run -e cu118 accelerate launch \
  --config_file accelerate_configs/inference/gpu_0.yaml \
  eval/run.py \
  --eval_mode pixel_fusion \
  --demo_filelist filenames/kitti360/trainval/demo.txt \
  --pretrained_model_path /path/to/stereosplat_plus_ckpt \
  --pretrained_diffix_model_path /path/to/model_130001.pkl \
  --use_ref --self_pseudo \
  --conf_pixel_level_fusion --fusion_mode soft \
  --output_folder outputs/vis/demo \
  --output_vis
```

Useful flags: `--conf_pixel_level_fusion`, `--fusion_mode soft`, `--self_pseudo`, `--pretrained_diffix_model_path`, `--no_difix3d`, `--timestep 199`. Full list → [`stereosplat_conf/eval/README.md`](stereosplat_conf/eval/README.md).

---

## Open-source checklist

Machine-specific paths have been removed from scripts and configs. Before publishing:

- Set `KITTI360_DATAPATH`, `UNIMATCH_WEIGHTS`, `DIFIX3D_WEIGHTS`, etc. locally (see **Quick Start** above).
- For Difix3D finetuning, generate `all_results_dict.json` from `difix3d/filenames/Validation_Set/all_results_dict.example.json` — do not commit local copies.
- Remove or rotate any **WandB API keys** if present in your environment.
- Add a top-level **LICENSE** if you intend public release (the custom rasterizer under `diff-gaussian-rasterization-conf/` carries its own [LICENSE.md](stereosplat_conf/diff-gaussian-rasterization-conf/LICENSE.md) derived from 3D Gaussian Splatting).

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{liu2026stereosplatplus,
  title     = {StereoSplat+: Feed-Forward Stereo Gaussian Splatting with Diffusion-Assisted Progressive Inference},
  author    = {Liu, Zihua and Okutomi, Masatoshi},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  eprint    = {2607.08808},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2607.08808}
}
```

---

## Acknowledgements

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)
- [Difix3D](https://github.com/nv-tlabs/Difix3D) for pseudo-view enhancement
- [UniMatch](https://github.com/autonomousvision/unimatch) for depth initialization
