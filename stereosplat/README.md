# StereoSplat+ with Confidence

**本仓库是带 confidence 的 StereoSplat 专用版本（StereoSplat_Plus）**，不是原版 14D 无 conf 的 StereoSplat。

| 要点 | 说明 |
|------|------|
| Gaussian 维度 | **15D**（14 维几何/外观 + 1 维 `conf`） |
| Rasterizer | 自定义 `diff-gaussian-rasterization`，同步输出 `rendered_conf` |
| 训练 | 仅 conf 版；无 14D `train_kitti360_stereosplat.py` 入口 |
| 评估 | 统一入口 `eval/run.py`；详见 **[eval/README.md](eval/README.md)** |

---

## 目录

- [安装](#安装)
- [训练](#训练)
- [推理方式详解](#推理方式详解)
- [评估速查（Shell 与 CLI）](#评估速查shell-与-cli)
- [可视化](#可视化)
- [目录结构](#目录结构)
- [15D Gaussian 布局](#15d-gaussian-布局)
- [文档索引](#文档索引)

---

## 安装

```bash
cd stereosplat
pixi install -e cu118
pixi run -e cu118 setup    # 编译含 conf 通道的 diff-gaussian-rasterization（必须）
```

> 未执行 `setup` 或误用 PyPI 原版 rasterizer 会导致训练/评估异常。

---

## 训练

均为 **15D conf 模型**（`gs_dim=15`，config 中 `use_conf_loss=True`）。

### Stage1 — 全 GT view + conf 监督

```bash
# 编辑 scripts/train/stereosplat/train.sh 中的路径
bash scripts/train/stereosplat/train.sh
```

| 项 | 值 |
|----|-----|
| Trainer | `trainer/train_kitti360_stereosplat_with_conf.py` |
| Config | `input_invariant_stereosplat_default.py` |
| 数据 | 全部 GT view，input-invariant |

Conf 自监督：

```
conf_gt = exp(-λ · mean_L1(rendered_rgb, gt_rgb))   [stop-gradient]
L_conf  = MSE(rendered_conf, conf_gt)
```

### Stage2 — pseudo-GT mix + Difix3D

```bash
bash scripts/train/stereosplat/train_stereosplat_stage2.sh
```

| 项 | 值 |
|----|-----|
| Trainer | `trainer/train_kitti360_stereosplat_stage2_with_difix3d.py` |
| Config | `input_invariant_stereosplat_stage2.py` |
| 依赖 | 冻结 **Stage1 conf checkpoint**（`--stage_1_model_path`） |

---

## 推理方式详解

评估 = **在 KITTI-360 val bins 上跑推理并算指标**（或 `--output_vis` 只出图）。  
统一入口：`eval/run.py` → `eval/routes.py` → `stereosplat.py` 中对应函数。

### 推理管线总览（三种 mode 递进）

```
输入：每个 bin 取 2-view GT stereo
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ ① stereosplat（基础）                                          │
│    2-view → forward 一次 → 3DGS → 渲染 bin 内 novel views      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ ② stereosplat_plus（S+）                                      │
│    在 ① 基础上：pseudo view → Difix3D 修复 → re-inject        │
│    → 再 forward → 渲染 novel views                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ ③ pixel_fusion（S+ + conf 融合）                               │
│    在 ② 基础上：两路 render 按 per-pixel conf 融合              │
│    （whole：2-view GS vs pseudo-multiview GS；                  │
│      separated：Stage1 render vs Stage2 render）              │
└─────────────────────────────────────────────────────────────┘
```

### 描述一次推理的三个参数

| 参数 | CLI | 含义 |
|------|-----|------|
| 训练阶段 | `--training_stage {stage1,stage2}` | 用哪套 config / dataloader（**不是**用哪个 ckpt 的唯一依据） |
| 评估模式 | `--eval_mode {stereosplat,stereosplat_plus,pixel_fusion}` | 走上面哪条管线 |
| 模型架构 | `--architecture {whole,separated}` | 单 checkpoint 还是「冻结 Stage1 + Stage2」 |

**权重**由 `--pretrained_model_path` / `--stage_1_model_path` 单独指定，可与 `training_stage` 组合（例如 Stage2 dataloader + Stage1 ckpt 做对照）。

---

### 方式 ①：stereosplat（纯 2-view forward）

| 项 | 说明 |
|----|------|
| **做什么** | 只用 2 张 GT view 做一次 feed-forward，建 3DGS，直接在 forward views 上渲染评估 |
| **模型函数** | `validation_on_the_forward_views()` |
| **架构** | 仅 `whole`（单模型） |
| **Difix3D** | 不使用 |
| **conf 融合** | 无 |
| **权重** | 仅 `--pretrained_model_path` |

**Stage1 vs Stage2 区别**：config / dataloader 不同（`First_LiDAR_3_Uniform` vs `First_Stage2`），不是算法不同。

| 变体 | training_stage | 主 checkpoint | 用途 |
|------|----------------|---------------|------|
| Stage1 基础评估 | stage1 | Stage1 | Stage1 模型本身能力 |
| Stage2 基础评估 | stage2 | Stage2 | Stage2 模型本身能力 |
| Stage2 对照（S1 权重） | stage2 | **Stage1** | 同一 Stage2 dataloader 下，看 Stage1 ckpt 的基础 GS 质量 |

对应 Shell：`stage1/stereosplat.sh`、`stage2/stereosplat_whole_s2.sh`、`stage2/stereosplat_whole_s1.sh`

---

### 方式 ②：stereosplat_plus（StereoSplat+）

在 stereosplat 基础上增加 **pseudo view → Difix enhance → re-inject → 再 forward**。  
分两种实现路径（**whole 与 separated 底层函数不同**）：

#### ②-A whole — 单 checkpoint progressive（推荐基线）

| 项 | 说明 |
|----|------|
| **做什么** | 同一 checkpoint 内两次前向：2-view 建 GS → 代码内固定选 pseudo pose（center+last）→ 可选 Difix → reinject → 再评估 |
| **模型函数** | `validation_on_the_forward_views_progressive_iter_once_revised()` |
| **架构** | `whole` |
| **权重** | 仅 `--pretrained_model_path` |
| **Difix3D** | 可选：`--use_diffix3d --use_ref` |
| **pseudo_ratio** | 不使用（pose 策略固定在函数内） |

对应 Shell：`stage1/stereosplat_plus.sh`、`stage2/stereosplat_plus_whole.sh`

#### ②-B separated — 冻结 Stage1 + Stage2 双模型

| 项 | 说明 |
|----|------|
| **做什么** | **Stage1 冻结**：2-view → 初始 3DGS / pseudo 渲染；**Stage2 主模型**：re-inject 后推理与评估 |
| **模型函数** | `stereosplatplus_pose_view_selection_injection_two_seperated_models(pixel_level_conf_fusion=False)` |
| **架构** | `separated`（**仅 Stage2 评估**，Stage1 无 separated） |
| **权重** | `--pretrained_model_path`（Stage2）+ `--stage_1_model_path`（冻结 Stage1） |
| **Difix3D** | 可选 |
| **pseudo_ratio** | `--pseudo_ratio 0.5 1.0`（控制 pseudo 视角比例） |

对应 Shell：`stage2/stereosplat_plus_separated.sh`  
（Stage1 只有 whole，无 separated S+。）

---

### 方式 ③：pixel_fusion（S+ + 逐像素 confidence 融合）

在 S+ 管线上，对 **两路 novel-view 渲染** 按 confidence **逐像素**选更好的一路（`fuse_renders_by_conf_pixelwise`）。

| 融合开关 | CLI | 行为 |
|----------|-----|------|
| fusion **off** | 不加 `--conf_pixel_level_fusion` | 走完整 S+ 管线，但不融合（与改 fusion 前的 deactivate 一致） |
| fusion **on** | `--conf_pixel_level_fusion` | 逐像素 `conf_b >= conf_a` 取 B；fused_conf 取被选中路的 conf |

#### ③-A whole — 单 checkpoint，融合两路内部 render

| 项 | 说明 |
|----|------|
| **融合的两路** | 同 checkpoint：**2-view GS 渲染** vs **pseudo-multiview GS 渲染** |
| **模型函数** | `stereosplatplus_difix3d_pose_view_selection_injection(pixel_level_conf_fusion=...)` |
| **架构** | `whole` |
| **权重** | 仅 `--pretrained_model_path` |
| **Difix3D** | 通常开启（`--use_diffix3d --use_ref`）；whole pixel_fusion 会预加载 Difix 权重 |
| **pseudo_ratio** | `--pseudo_ratio 0.5 1.0` |

> **注意**：whole 的 `pixel_fusion` 与 whole 的 `stereosplat_plus` **不是同一个函数**。  
> - S+ whole → progressive（固定 pseudo）  
> - pixel_fusion whole → pose injection + 可选 conf 融合  

对应 Shell：`stage1/pixel_fusion.sh`、`stage2/pixel_fusion_whole.sh`

#### ③-B separated — 融合 Stage1 vs Stage2 渲染

| 项 | 说明 |
|----|------|
| **融合的两路** | **Stage1 冻结模型渲染** vs **Stage2 模型渲染**（同一相机） |
| **模型函数** | `stereosplatplus_pose_view_selection_injection_two_seperated_models(pixel_level_conf_fusion=...)` |
| **架构** | `separated` |
| **权重** | Stage2 + 冻结 Stage1（同 ②-B） |
| **Difix3D** | 通常开启 |
| **pseudo_ratio** | `--pseudo_ratio 0.5 1.0` |

对应 Shell：`stage2/pixel_fusion_separated.sh`

---

### 消融：Oracle 上界

| 项 | 说明 |
|----|------|
| **CLI** | `--use_gt_view` |
| **模型函数** | `oracle_upper_bound_ablation()` |
| **含义** | 用 GT view 作上界对照，与上述 ①②③ 正交，可叠加任意 mode 做消融 |

---

### 完整对照表（9 种标准推理 + Shell）

| # | Stage | eval_mode | arch | 主 ckpt | 冻结 S1 | Difix | conf fusion | 模型函数（简写） | Shell |
|---|-------|-----------|------|---------|---------|-------|-------------|------------------|-------|
| 1 | 1 | stereosplat | whole | S1 | — | ✗ | ✗ | `validation_on_the_forward_views` | `stage1/stereosplat.sh` |
| 2 | 1 | stereosplat_plus | whole | S1 | — | 可选 | ✗ | `..._progressive_iter_once_revised` | `stage1/stereosplat_plus.sh` |
| 3 | 1 | pixel_fusion | whole | S1 | — | 通常 ✓ | 可选 | `stereosplatplus_difix3d_pose_view_selection_injection` | `stage1/pixel_fusion.sh` |
| 4 | 2 | stereosplat | whole | S2 | — | ✗ | ✗ | `validation_on_the_forward_views` | `stage2/stereosplat_whole_s2.sh` |
| 5 | 2 | stereosplat | whole | **S1** | — | ✗ | ✗ | `validation_on_the_forward_views` | `stage2/stereosplat_whole_s1.sh` |
| 6 | 2 | stereosplat_plus | whole | S2 | — | 可选 | ✗ | `..._progressive_iter_once_revised` | `stage2/stereosplat_plus_whole.sh` |
| 7 | 2 | stereosplat_plus | separated | S2 | **S1** | 可选 | ✗ | `..._two_seperated_models` | `stage2/stereosplat_plus_separated.sh` |
| 8 | 2 | pixel_fusion | whole | S2 | — | 通常 ✓ | 可选 | `stereosplatplus_difix3d_pose_view_selection_injection` | `stage2/pixel_fusion_whole.sh` |
| 9 | 2 | pixel_fusion | separated | S2 | **S1** | 通常 ✓ | 可选 | `..._two_seperated_models` | `stage2/pixel_fusion_separated.sh` |

函数全名见 `eval/routes.py`。每个 Shell 内通常有 `*_vis()` 变体（加 `--output_vis`，用 `demo.txt`）。

---

### 易混淆点

1. **`training_stage` ≠ 用哪个 checkpoint**  
   - `stage2/stereosplat_whole_s1.sh`：`training_stage=stage2`（Stage2 dataloader），但 `pretrained_model_path=Stage1`。

2. **whole 下 S+ 与 pixel_fusion 是两套实现**  
   - S+ → progressive，无 `pseudo_ratio`  
   - pixel_fusion → pose injection + `pseudo_ratio` + 可选 fusion

3. **separated 仅 Stage2 的 S+ / pixel_fusion**  
   - Stage1 训练只有单模型，评估也只有 `whole`。

4. **14D 旧权重**  
   - 本仓库 rasterizer / `gs_dim=15` 不支持直接加载无 conf 旧 ckpt 跑上述流程。

更细的调用链、CLI、FAQ → **[eval/README.md](eval/README.md)**

---

## 评估速查（Shell 与 CLI）

评估逻辑在 **`eval/`**，主入口 **`eval/run.py`**。  
Shell 在 `scripts/evaluation/stage{1,2}/`；默认路径在 `scripts/evaluation/_common.sh`。

### 实验矩阵 → Shell（同上表 #1–#9）

| Stage | eval_mode | architecture | 推荐 Shell |
|-------|-----------|--------------|------------|
| 1 | stereosplat | whole | `stage1/stereosplat.sh` |
| 1 | stereosplat_plus | whole | `stage1/stereosplat_plus.sh` |
| 1 | pixel_fusion | whole | `stage1/pixel_fusion.sh` |
| 2 | stereosplat | whole (S2 ckpt) | `stage2/stereosplat_whole_s2.sh` |
| 2 | stereosplat | whole (S1 ckpt 对照) | `stage2/stereosplat_whole_s1.sh` |
| 2 | stereosplat_plus | whole | `stage2/stereosplat_plus_whole.sh` |
| 2 | stereosplat_plus | separated | `stage2/stereosplat_plus_separated.sh` |
| 2 | pixel_fusion | whole | `stage2/pixel_fusion_whole.sh` |
| 2 | pixel_fusion | separated | `stage2/pixel_fusion_separated.sh` |

### 快速运行

```bash
cd stereosplat

# Stage2 基础 2-view
bash scripts/evaluation/stage2/stereosplat_whole_s2.sh

# Stage2 S+ separated
bash scripts/evaluation/stage2/stereosplat_plus_separated.sh

# Stage2 pixel-level conf fusion
bash scripts/evaluation/stage2/pixel_fusion_separated.sh
```

### 直接调 eval/run.py

```bash
pixi run -e cu118 accelerate launch \
  --config-file accelerate_configs/inference/gpu_0.yaml \
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

### 常用参数

| Flag | 说明 |
|------|------|
| `--output_vis` | 可视化模式（见下节） |
| `--use_diffix3d` / `--use_ref` | 启用 Difix3D + stereo ref |
| `--conf_pixel_level_fusion` | 逐像素 conf 融合（`pixel_fusion` 模式） |

### 旧路径（仍可用，不推荐新实验使用）

| 旧 Shell / Validator | 等效语义 |
|----------------------|----------|
| `stereosplat/render_inside_bin.sh` | stage2 stereosplat whole |
| `stereosplat_plus/render_inside_bin_whole_model.sh` | stage2 stereosplat_plus whole |
| `conf_fusion/pixel_level/sep_model.sh` | stage2 pixel_fusion separated |
| `conf_fusion/pixel_level/whole_mode.sh` | stage2 pixel_fusion whole |
| `conf_fusion/pixel_level/whole_model_stage1.sh` | stage1 pixel_fusion whole |
| `validator/*.py` | 薄 wrapper → `eval/run.py` |

---

## 可视化

**已实现**：`eval/run.py` 支持 `--output_vis`，模型侧 `vis=True` 保存图像。

| 项 | 说明 |
|----|------|
| 数据列表 | 默认 `filenames/kitti360/train_complete/demo.txt`（9 bins） |
| 扩展列表 | `demo_more.txt`（27 bins），`--demo_filelist` 手动指定 |
| 输出 | `output_folder/<bin_token>/rendered_images/` 等 |
| 指标 | 可视化模式**不写** `metric.json` |

每个 `stage{1,2}/*.sh` 均提供 `*_vis()` 函数，底部注释切换即可：

```bash
# 编辑 scripts/evaluation/stage2/stereosplat_whole_s2.sh：
#eval_stage2_stereosplat_whole_s2
eval_stage2_stereosplat_whole_s2_vis
```

或任意命令末尾加 `--output_vis`。

---

## 目录结构

```
stereosplat/
├── eval/                    # ★ 评估逻辑（eval/run.py）
├── trainer/
│   ├── train_kitti360_stereosplat_with_conf.py          # Stage1
│   └── train_kitti360_stereosplat_stage2_with_difix3d.py # Stage2
├── validator/               # 兼容 wrapper → eval/run.py
├── scripts/
│   ├── train/stereosplat/   # train.sh, train_stereosplat_stage2.sh
│   └── evaluation/
│       ├── _common.sh       # 默认路径 + launch 助手
│       ├── stage1/          # ★ Stage1 评估（3 个 shell）
│       └── stage2/          # ★ Stage2 评估（6 个 shell）
├── src/stereosplat/         # 模型、config、dataloader
├── diff-gaussian-rasterization/  # 含 conf 的 rasterizer
└── difix3d/
```

---

## 15D Gaussian 布局

| Dims | Field |
|------|-------|
| 0:3 | mean (XYZ) |
| 3:6 | rgb |
| 6:7 | opacity |
| 7:11 | rotation |
| 11:14 | scale |
| **14:15** | **conf** |

```
rendered_conf[pixel] = Σ conf_i · α_i · T_i
```

14D 旧 checkpoint **不能**直接用于本仓库默认流程。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| **本文件** | 安装、训练、**推理方式详解（9 种 + Oracle）**、Shell 速查、可视化 |
| **[eval/README.md](eval/README.md)** | 评估深入：调用链 mermaid、CLI 全参数、legacy 映射、FAQ |
| `scripts/evaluation/_common.sh` | 默认 checkpoint / Difix3D / filelist 路径 |

---

## Quick Reference

```bash
pixi install -e cu118 && pixi run -e cu118 setup

bash scripts/train/stereosplat/train.sh
bash scripts/train/stereosplat/train_stereosplat_stage2.sh

bash scripts/evaluation/stage2/stereosplat_whole_s2.sh
bash scripts/evaluation/stage2/pixel_fusion_separated.sh

# 可视化：shell 里启用 *_vis() 或加 --output_vis
```
