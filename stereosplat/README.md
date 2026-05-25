# StereoSplat (Pixi)

Feed-forward 3D Gaussian Splatting for KITTI-360 driving scenes.  
Dual-branch architecture: Cost-Volume pixel GS + TPVFormer volume GS, with optional confidence-supervised training (StereoSplat-Conf) and Difix3D-based progressive inference (StereoSplat+).

---

## Requirements

- Linux (`linux-64`)
- NVIDIA GPU (≥ 12 GB VRAM)
- `git`, `curl`

---

## 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi --version
```

---

## 2. Build the Environment

```bash
cd stereosplat

# Step 1: Resolve and install all Python dependencies (including accelerate, wandb, torch, etc.)
pixi install -e cu118

# Step 2: Compile the custom diff-gaussian-rasterization (conf rendering support)
#         and install mmcv / mmdet / mmdet3d
pixi run -e cu118 setup
```

> **Note**: `requirements.txt` packages are now declared in `[tool.pixi.feature.common.pypi-dependencies]`  
> and are handled automatically by `pixi install`. You no longer need to run `pip-req` manually.

---

## 3. Training

### 3a. StereoSplat Base (GT views only, no conf)

Original input-view-invariant training without confidence supervision.

```bash
# Edit paths/settings at the top of the script first
cd stereosplat/scripts/train/stereosplat
bash train.sh
```

### 3b. StereoSplat-Conf (recommended — with confidence supervision)

Trains the same architecture with an additional per-Gaussian `conf` channel (15D Gaussians).  
Conf is supervised via a self-supervised photometric soft label:

```
conf_gt = exp(-λ · mean_L1(rendered_rgb, gt_rgb))   [stop-gradient]
L_conf  = MSE(rendered_conf, conf_gt)
```

This produces **Model 1 with conf**, which is the foundation for Path 3 (conf-guided fusion in StereoSplat+).

```bash
cd stereosplat/scripts/train/stereosplat
bash train.sh          # already points to train_kitti360_stereosplat_with_conf.py
```

Key config knobs (`src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_conf_loss` | `True` | Enable conf supervision |
| `conf_lambda` | `10.0` | Sharpness of soft label (higher → conf drops faster with error) |
| `fusion_sup_dict.weight_conf` | `0.1` | Weight of conf MSE loss |

---

## 4. Validation / Evaluation

Both scripts use `pixi run -e cu118 python -m accelerate.commands.launch` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (same as training).

### StereoSplat-Conf (Model 1)

```bash
cd stereosplat/scripts/evaluation/stereosplat
# Edit pretrained_model_path inside the script first
bash render_inside_bin.sh
```

### StereoSplat+ (with Difix3D restoration)

```bash
cd stereosplat/scripts/evaluation/stereosplat_plus
bash render_inside_bin.sh
```

Optional flags (edit in the script):

| Flag | Description |
|------|-------------|
| `--output_vis` | Save per-scene visualizations (RGB, depth, depth-error, **conf map**) |
| `--use_diffix3d` | Enable Difix3D image restoration (StereoSplat+ only) |
| `--use_ref` | Use stereo reference for restoration |
| `--deterministic_vae_encode` / `--deterministic_scheduler_step` | Deterministic Difix3D inference |

When `--output_vis` is set, each scene folder contains:

```
<scene>/
├── rendered_images/     # RGB renders (first/last/center stereo pairs)
├── rendered_depth/      # Depth renders (plasma colormap)
├── GT Images/
├── GT Depth/
├── Rendered_Depth_Error/
└── rendered_conf/       # Confidence maps (plasma colormap) ← new (conf model only)
```

---

## 5. Gaussian Layout (15D)

After conf training, each Gaussian has 15 parameters:

| Dims | Field |
|------|-------|
| 0:3 | `mean` (XYZ world position) |
| 3:6 | `rgb` (color, sigmoid) |
| 6:7 | `opacity` (sigmoid) |
| 7:11 | `rotation` (unit quaternion) |
| 11:14 | `scale` (exp × 0.01) |
| **14:15** | **`conf` (sigmoid, self-supervised)** |

The modified `diff-gaussian-rasterization` rasterizes conf via the same α-compositing as depth:

```
rendered_conf[pixel] = Σ conf_i · α_i · T_i
```

---

## 6. Environment Structure

```
stereosplat/
├── src/stereosplat/
│   ├── configs/stereosplat/          # mmengine configs
│   ├── data/                         # KITTI-360 dataloaders (5 world-center variants)
│   └── models_lab/StereoSplat/       # main model (encoder, volume, gaussian, losses)
├── trainer/
│   ├── train_kitti360_stereosplat.py              # Stage 1 (GT only)
│   ├── train_kitti360_stereosplat_with_conf.py    # Stage 1 + conf supervision ← new
│   ├── train_kitti360_stereosplat_stage2_no_difix3d.py
│   └── train_kitti360_stereosplat_stage2_with_difix3d.py
├── validator/                        # evaluation scripts
├── scripts/                          # bash launch scripts
├── diff-gaussian-rasterization/      # custom rasterizer (rgb+depth+alpha+conf)
├── difix3d/                          # Difix3D image restoration
├── accelerate_configs/               # single/multi-GPU accelerate configs
└── pyproject.toml                    # pixi environment + tasks
```

---

## 7. Quick Reference

```bash
# Install everything
pixi install -e cu118 && pixi run -e cu118 setup

# Train with conf (Model 1 with conf)
bash scripts/train/stereosplat/train.sh

# Evaluate StereoSplat-Conf (edit pretrained_model_path first)
bash scripts/evaluation/stereosplat/render_inside_bin.sh

# Evaluate StereoSplat+ (edit pretrained_model_path first)
bash scripts/evaluation/stereosplat_plus/render_inside_bin.sh

# Save visualizations including conf maps
#   → uncomment --output_vis in render_inside_bin.sh
```
