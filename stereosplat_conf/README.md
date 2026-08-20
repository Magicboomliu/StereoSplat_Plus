# StereoSplat+ with Confidence

**Confidence-enabled StereoSplat (StereoSplat_Plus)** — code for [StereoSplat+ (arXiv:2607.08808)](https://arxiv.org/abs/2607.08808), IROS 2026.

In this repo, we support the original gaussain splatting with an additional confidence attribute.
And we use the rasterization function defined in [diff-gaussian-rasterization-conf](diff-gaussian-rasterization-conf)


| Item | Detail |
|------|--------|
| Gaussian dim | **15D** (14 geometry/appearance + 1 `conf`) |
| Rasterizer | Custom `diff-gaussian-rasterization-conf` → `rendered_conf` |
| Training | Conf-only; entry `train_kitti360_stereosplat_with_conf.py` |
| Evaluation | `eval/run_multi_gpu.py` (shells) or `eval/run.py`; details in **[eval/README.md](eval/README.md)** |

---

## Install

```bash
cd stereosplat_conf
pixi install -e cu118
pixi run -e cu118 setup    # Build diff-gaussian-rasterization-conf (required)
```

> Run `setup` before training or evaluation. The PyPI rasterizer does not support `conf`.

---

## Training

All trainers produce **15D conf models** (`gs_dim=15`, `use_conf_loss=True` in config).

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train/complete/train_stereosplat.sh` | Stage 1 — full val split, camera-ready outputs |
| `scripts/train/complete/train_stereosplat_plus.sh` | Stage 2 — self-pseudo + Difix3D, full val split |
| `scripts/train/ablations/train_stereosplat.sh` | Stage 1 ablation — single-sequence trainval split |
| `scripts/train/ablations/train_stereosplat_plus.sh` | Stage 2 ablation — same split as above |

Edit paths inside each script (`datapath`, `work_dir`, `stage_1_model_path`, Difix3D weights, etc.) before launching.

```bash
# Trainin the StereoSplat
bash scripts/train/complete/train_stereosplat.sh

# Train the StereoSplat-Plus
bash scripts/train/complete/train_stereosplat_plus.sh
```

### StereoSplat — all GT views + conf supervision

| Item | Value |
|------|-------|
| Trainer | `trainer/train_kitti360_stereosplat_with_conf.py` |
| Config | `src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py` |
| `world_center` | `First_LiDAR_3_Uniform` |

Conf self-supervision:

```
conf_gt = exp(-λ · mean_L1(rendered_rgb, gt_rgb))   [stop-gradient]
L_conf  = MSE(rendered_conf, conf_gt)
```

### Extend to Stereosplat-plus — pseudo-GT mix + Difix3D

| Item | Value |
|------|-------|
| Trainer | `trainer/train_kitti360_stereosplat_plus_with_difix3d.py` |
| Config | `src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py` |
| `world_center` | `First_Stage2` |

Each iteration (when `view_num > 2`):

1. Weighted random `view_num` (2: 10%; 3/4/5/6: 22.5% each).
2. With probability `mix_psuedo_views_ratio` (default **0.9**), render a **pseudo view** from the current model.
3. Independently, with probability `mix_difix3d_ratio` (default **0.9**), enhance the pseudo view with **Difix3D**.
4. Inject pseudo views as extra inputs (`imgs[:, 2:]`) and train the student.

Pseudo views are **inputs only** — no teacher distillation.

#### Current shell mode: self-pseudo (`--self_pseudo`)

Both `train_stereosplat_plus.sh` scripts pass `--self_pseudo`:

- One trainable model; pseudo views rendered from its own weights (`no_grad` + temporary `eval`).
- `--stage_1_model_path` initializes student weights (overwritten on `resume_from` resume).
- Lower memory than a frozen Stage1 copy.

The trainer still supports **dual-model** mode (omit `--self_pseudo`): a frozen Stage1 renders pseudo views while a separate Stage2 student trains. Not wired in the current shell scripts — enable via CLI if needed.


#### Key CLI flags (Stage 2 trainer)

| Flag | Description |
|------|-------------|
| `--stage_1_model_path` | Student init (self-pseudo) or frozen Stage1 weights (dual-model) |
| `--mix_psuedo_views_ratio` | Pseudo mix probability (default `0.9`) |
| `--mix_difix3d_ratio` | Difix3D apply probability (default `0.9`) |
| `--pretrained_difix3d` | Difix3D checkpoint; use with `--use_ref` |
| `--self_pseudo` | Self-bootstrap single-model training |
| `--resume-from` | `""` / `latest` / checkpoint path |

Multi-GPU uses `accelerate_configs/accelerate_config.yaml`. Scripts set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` for local Difix3D cache.

#### Validation output

Each `step-N/` folder writes **`fusion_metric.json`** with three sections:

| Section | Meaning |
|---------|---------|
| `2view` | Pure 2-view GT baseline (no pseudo, no Difix) |
| `pseudo_multiview` | 6-view input (2 GT + 4 pseudo), multiview GS render |
| `pseudo_fused` | Same as above + pixel-wise conf fusion (2-view GS vs multiview GS) |

Target: `pseudo_fused.psnr_mean > 2view.psnr_mean`. Wandb keys: `val/2view/psnr_mean`, `val/pseudo_multiview/psnr_mean`, `val/pseudo_fused/psnr_mean`.

---

## Evaluation

Shell scripts call **`eval/run_multi_gpu.py`** with `accelerate_configs/inference/multi_gpu.yaml`. Use **`bash`**, not `sh`. Checkpoints: test with `[ -e path ]` (directory checkpoints are valid).

### Scripts

| Script | What it runs |
|--------|----------------|
| `scripts/evaluation/evaluations/stereosplat.sh` | 2-view baseline — `--eval_mode stereosplat`, `whole` |
| `scripts/evaluation/evaluations/stereosplat_plus.sh` | Pixel fusion + Difix3D + `--self_pseudo` + `--fusion_mode soft` |
| `scripts/evaluation/ablations/stereosplat.sh` | Same as above, ablation val filelist + weights |
| `scripts/evaluation/ablations/stereosplat_plus.sh` | Same as above, ablation split |

```bash
cd stereosplat_conf

# 2-view metrics (camera-ready weights)
bash scripts/evaluation/evaluations/stereosplat.sh

# StereoSplat+ with pixel fusion + Difix3D
bash scripts/evaluation/evaluations/stereosplat_plus.sh
```

Edit `pretrained_model_path`, `pretrained_diffix_model_path`, and `output_folder` inside each script.

### Two evaluation modes in use

**① `stereosplat`** (`evaluations/stereosplat.sh`)

- 2 GT stereo views → single forward → render novel views → PSNR/SSIM/LPIPS.
- Model: `infer_stereosplat_two_gt_views_forward()`.
- No Difix3D, no conf fusion.

**② `pixel_fusion`** (default in `run_multi_gpu.py`; used by `stereosplat_plus.sh`)

- Full S+ pipeline: 2-view GS → trajectory render → pseudo stereo (`pseudo_ratio`, default `0.5 1.0`) → Difix3D → reinject → second forward.
- Per-pixel conf fusion of **2-view GS render** vs **pseudo-multiview GS render** (`--conf_pixel_level_fusion`).
- Current shells: `--fusion_mode soft`, `--use_ref`, `--self_pseudo`.
- Model: `infer_pixel_fusion_pose_injection_single_model()`.

`stereosplat_plus` (S+ without pixel fusion) is also available via `--eval_mode stereosplat_plus` on `run_multi_gpu.py` / `run.py` but is not used by the bundled shells.

### Call chain

```
eval/run_multi_gpu.py  (or eval/run.py)
  → eval/routes.py
  → stereosplat.py  infer_*()
```

### Direct CLI example

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

StereoSplat+ / pixel fusion example:

```bash
pixi run -e cu118 accelerate launch \
  --config_file accelerate_configs/inference/multi_gpu.yaml \
  eval/run_multi_gpu.py \
  --config_path src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py \
  --output_folder outputs/eval/my_plus_run \
  --val_filelist filenames/kitti360/train_complete/val.txt \
  --pretrained_model_path /path/to/stage2_ckpt \
  --pretrained_diffix_model_path /path/to/difix.pkl \
  --use_ref --self_pseudo \
  --conf_pixel_level_fusion --fusion_mode soft
```

### Useful eval flags

| Flag | Description |
|------|-------------|
| `--eval_mode` | `stereosplat` / `stereosplat_plus` / `pixel_fusion` (default `pixel_fusion`) |
| `--architecture` | `whole` (default) or `separated` (frozen Stage1 + Stage2) |
| `--pseudo_ratio` | Pseudo stereo selection, e.g. `0.5 1.0` (center + last; auto-filled if omitted) |
| `--self_pseudo` | Match self-pseudo trained checkpoints |
| `--no_difix3d` | Skip Difix3D enhancement |
| `--conf_pixel_level_fusion` | Enable 2D per-pixel conf fusion |
| `--conf_fusion_margin` | A1 margin: pick plus only if `conf_plus > conf_base + margin` |
| `--fusion_mode` | `soft` (default in shells) / `legacy` / `per_view_adaptive` |
| `--gs_conf_fusion` | 3D voxel GS fusion (CLI only; no bundled shell) |
| `--output_vis` | Save images (`eval/run.py` single-GPU; **not** multi-GPU) |

Full parameter list, FAQ, and output format → **[eval/README.md](eval/README.md)**.

---

## Layout (train + eval)

```
stereosplat_conf/
├── eval/
│   ├── run_multi_gpu.py      # multi-GPU eval (used by shells)
│   └── run.py                # single-process / advanced CLI
├── trainer/
│   ├── train_kitti360_stereosplat_with_conf.py
│   └── train_kitti360_stereosplat_plus_with_difix3d.py
├── scripts/
│   ├── train/
│   │   ├── complete/         # train_stereosplat.sh, train_stereosplat_plus.sh
│   │   └── ablations/
│   └── evaluation/
│       ├── evaluations/      # stereosplat.sh, stereosplat_plus.sh
│       └── ablations/
├── tests/                    # pixi run -e cu118 test-stereosplat
└── src/stereosplat/
```


## Models Zoo (Google Drive)

### Utility Models Weights

| Model | Download | File to use |
|-------|----------|-------------|
| Refined Difix3D (pseudo view enhancement) | [model_130001.pkl](https://drive.google.com/file/d/15UOotc_7WRJ_Mg9T3g0enx9Kuh_Yn9Cm/view?usp=drive_link) | `model_130001.pkl` |
| UniMatch (depth init) | [depth_estimation_224x840](https://drive.google.com/drive/folders/1zy7PVENps22YavP2sDaNlVmBrjfBko5U?usp=drive_link) | `checkpoint-90000/model.safetensors` |


### Ablations Pre-trained Weights
- [StereoSplat-Conf](https://drive.google.com/drive/folders/1rk8RCTS96JV4O1iE_KJb1wcx23SF4elB?usp=sharing)
- [StereoSplat-Plus-Conf](https://drive.google.com/drive/folders/1TaG-n7EjGt36VmA7ynpei4tTPS_nFKa7?usp=sharing) 

### Completed Training Set Pre-trained Weights
- [StereoSplat-Conf](https://drive.google.com/drive/folders/1sm3rWZ0IcgiP3dZ-XgRZgCL0f0wi-ooU?usp=sharing)
- [StereoSplat-Plus-Conf](https://drive.google.com/drive/folders/12lMwnNCBrI76M53eFrYIgWfqEA1FmVDO?usp=sharing) 
