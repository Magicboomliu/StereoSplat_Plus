# StereoSplat+ with Confidence (Pixi)

**本仓库是带 confidence 的 StereoSplat 专用版本（StereoSplat_Plus），不是原版 14D 无 conf 的 StereoSplat。**

- 每个 Gaussian 为 **15 维**（14 维几何/外观 + 1 维 `conf`）
- 自定义 `diff-gaussian-rasterization` 在渲染时同步输出 `rendered_conf`（α-compositing，与 depth 同路）
- 训练、评估、pixel-level fusion 均假设 checkpoint 为 **15D conf 模型**

> **不支持无 conf 训练**：仓库内没有 `train_kitti360_stereosplat.py`（14D）入口；`train.sh` 固定调用 `train_kitti360_stereosplat_with_conf.py`，config 中 `gs_dim=15`、`use_conf_loss=True`。若关闭 conf loss 或改用 `gs_dim=14`，需自行改 config 并确认与 rasterizer / 解码器维度一致，**不在本仓库默认支持范围内**。

架构：Cost-Volume pixel GS + TPVFormer volume GS；Stage2 可选 Difix3D pseudo-GT mix；推理侧支持 StereoSplat / StereoSplat+ / pixel-level conf fusion。

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

# Step 2: Compile the custom diff-gaussian-rasterization (必须，含 conf 渲染通道)
#         and install mmcv / mmdet / mmdet3d
pixi run -e cu118 setup
```

> **Note**: `pixi run setup` 会编译本仓库自带的 `diff-gaussian-rasterization`（相对原版多出 **conf** 输出）。未编译或误用 PyPI 原版 rasterizer 会导致训练/评估异常。  
> Python 依赖由 `pixi install` 自动处理，无需再手动 `pip install -r requirements.txt`。

---

## 3. Training（均为 15D conf 模型）

### 3a. Stage1 — GT views + conf 监督

Input-invariant 训练，输入均为 GT view；在 14D Gaussian 上增加第 15 维 `conf`，并用自监督 photometric soft label 约束：

```
conf_gt = exp(-λ · mean_L1(rendered_rgb, gt_rgb))   [stop-gradient]
L_conf  = MSE(rendered_conf, conf_gt)
```

```bash
cd stereosplat
# 编辑 scripts/train/stereosplat/train.sh 中的路径后执行
bash scripts/train/stereosplat/train.sh
# → trainer/train_kitti360_stereosplat_with_conf.py
# → config: input_invariant_stereosplat_default.py (gs_dim=15, use_conf_loss=True)
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `gs_dim` | `15` | Gaussian 维度，**本仓库固定为 15** |
| `use_conf_loss` | `True` | 是否加 conf MSE（可改 False 做消融，但仍是 15D 模型） |
| `conf_lambda` | `10.0` | soft label 锐度 |
| `fusion_sup_dict.weight_conf` | `0.1` | conf loss 权重 |

### 3b. Stage2 — pseudo-GT mix + Difix3D（依赖 Stage1 conf checkpoint）

冻结 Stage1 conf 模型生成 3DGS，随机混入 GT view 继续训练：

```bash
bash scripts/train/stereosplat/train_stereosplat_stage2.sh
# → trainer/train_kitti360_stereosplat_stage2_with_difix3d.py
# → config: input_invariant_stereosplat_stage2.py
# → 必须提供 stage_1_model_path（15D conf 权重）
```

---

## 4. Validation / Evaluation

**All evaluation logic lives in `eval/`** (entry: `eval/run.py`).  
See **[eval/README.md](eval/README.md)** for the full guide: stage × eval_mode × architecture matrix, call chains, CLI, legacy path mapping, and examples.

### Quick start (recommended shells)

```bash
cd stereosplat

# Stage2 | plain 2-view StereoSplat (Stage2 checkpoint)
bash scripts/evaluation/stage2/stereosplat_whole_s2.sh

# Stage2 | StereoSplat+ | separated (frozen Stage1 + Stage2)
bash scripts/evaluation/stage2/stereosplat_plus_separated.sh

# Stage2 | pixel-level confidence fusion | separated
bash scripts/evaluation/stage2/pixel_fusion_separated.sh

# Stage1 | pixel fusion ablation (Stage1 checkpoint)
bash scripts/evaluation/stage1/pixel_fusion.sh
```

### Direct entry (most flexible)

```bash
pixi run -e cu118 accelerate launch --config-file accelerate_configs/inference/gpu_0.yaml \
  eval/run.py \
  --training_stage stage2 \
  --eval_mode pixel_fusion \
  --architecture separated \
  --output_folder /path/to/output \
  --pretrained_model_path /path/to/stage2/latest \
  --stage_1_model_path /path/to/stage1/checkpoint-145000 \
  --val_filelist filenames/kitti360/train_complete/val.txt \
  --pretrained_diffix_model_path /path/to/difix.pkl \
  --use_diffix3d --use_ref --conf_pixel_level_fusion
```

Legacy paths (`validator/*.py`, `scripts/evaluation/stereosplat_plus/conf_fusion/`) still work as thin wrappers.

### Common flags

| Flag | Description |
|------|-------------|
| `--output_vis` | Save per-scene visualizations (uses `demo.txt` by default; skips `metric.json`) |
| `--use_diffix3d` | Enable Difix3D restoration (S+ / pixel_fusion) |
| `--use_ref` | Stereo reference for Difix3D |
| `--conf_pixel_level_fusion` | Per-pixel conf fusion (`pixel_fusion` mode) |

Default paths (checkpoints, Difix3D): edit `scripts/evaluation/_common.sh`.

---

## 5. Gaussian Layout（15D，本仓库标准格式）

每个 Gaussian **必须**按 15 维布局（与 rasterizer、decoder、`gs_dim=15` 一致）：

| Dims | Field |
|------|-------|
| 0:3 | `mean` (XYZ world position) |
| 3:6 | `rgb` (color, sigmoid) |
| 6:7 | `opacity` (sigmoid) |
| 7:11 | `rotation` (unit quaternion) |
| 11:14 | `scale` (exp × 0.01) |
| **14:15** | **`conf` (sigmoid, self-supervised)** |

本仓库 `diff-gaussian-rasterization` 在 forward 中额外输出 `conf` 图（与 rgb / depth / alpha 同一套 α-blending）：

```
rendered_conf[pixel] = Σ conf_i · α_i · T_i
```

Pixel-level fusion、conf 可视化、`metric.json` 中的 conf 指标都依赖该渲染结果。**14D 旧 checkpoint 不能直接用于本仓库默认流程。**

---

## 6. Environment Structure

```
stereosplat/
├── src/stereosplat/
│   ├── configs/stereosplat/          # mmengine configs
│   ├── data/                         # KITTI-360 dataloaders (5 world-center variants)
│   └── models_lab/StereoSplat/       # main model (encoder, volume, gaussian, losses)
├── trainer/
│   ├── train_kitti360_stereosplat_with_conf.py    # Stage1（唯一 Stage1 入口，15D conf）
│   └── train_kitti360_stereosplat_stage2_with_difix3d.py
├── eval/                             # evaluation logic (eval/run.py)
├── validator/                        # legacy wrappers → eval/run.py
├── scripts/                          # bash launch scripts (stage1/, stage2/)
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

# Stage1 conf 训练
bash scripts/train/stereosplat/train.sh

# Stage2 pseudo-GT mix 训练（需先完成 Stage1 conf）
bash scripts/train/stereosplat/train_stereosplat_stage2.sh

# Evaluate (see eval/README.md for full matrix)
bash scripts/evaluation/stage2/stereosplat_whole_s2.sh
bash scripts/evaluation/stage2/pixel_fusion_separated.sh

# Visualize on demo.txt (9 bins): use *_vis() in shell or add --output_vis
#eval_stage2_stereosplat_whole_s2_vis   # see scripts/evaluation/stage2/stereosplat_whole_s2.sh
```
