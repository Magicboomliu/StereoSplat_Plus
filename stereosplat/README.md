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

> **详细文档**（当前 loss / 权重 / 收敛趋势）：[scripts/train/stereosplat/README_STAGE2.md](scripts/train/stereosplat/README_STAGE2.md)

| 项 | 值 |
|----|-----|
| Trainer | `trainer/train_kitti360_stereosplat_stage2_with_difix3d.py` |
| Config | `input_invariant_stereosplat_stage2.py` |
| Shell | `scripts/train/stereosplat/train_stereosplat_stage2.sh`（**一个脚本，两个函数**） |

#### 核心逻辑

Stage2 = **加载一个权重 → 生成 pseudo view → 回灌 → 训练**。每个 iter：

1. **加权随机**选 `view_num`：`view_num=2` 占 10%，`3/4/5/6` 各占 22.5%（保证 multi-view 占主导）。
2. 以 `mix_psuedo_views_ratio`（默认 `0.9`）概率，用一个模型渲染出 **pseudo view**（新视角图）。
3. 独立地，再以 `mix_difix3d_ratio`（默认 `0.9`）概率，把 pseudo view 过 **Difix3D** 增强。
4. pseudo view **作为输入回灌**（替换 `imgs[:, 2:]`），训练当前模型。

> pseudo view 只作 **输入**，不是蒸馏目标（不存在 teacher logits 监督）。

#### Loss 组成与权重

| 名称 | 公式/说明 | 权重 |
|------|----------|------|
| `recon_gs` | MSE(rendered_rgb, gt_rgb) on multiview GS | `weight_recon=1.0` |
| `perceptual_gs` | VGG/LPIPS on multiview GS | `weight_percep=0.1` |
| `conf_loss` | MSE(rendered_conf, exp(-λ·L1)) — conf 自监督 | `weight_conf=0.5` |
| `depth_est_loss` | 深度估计监督 | `branch_weight=0.1` |
| `recon_pixel_fused` | MSE(pixel_fused_rgb, gt_rgb) — **像素融合监督** | `weight_fusion_sup=1.5` |
| `perceptual_pixel_fused` | LPIPS(pixel_fused_rgb, gt_rgb) — **融合感知质量** | `weight_fusion_sup_percep=0.3` |
| `conf_comparative` | MSE(conf, sigmoid(λ·(err_2v − err_mv))) — **引导 conf 反映两路质量差** | `weight_conf_comparative=0.3` |

> `recon_pixel_fused` / `perceptual_pixel_fused` / `conf_comparative` 仅当 `view_num > 2`（有 pseudo view）且处于 `train` 模式时计算。

#### 两种变体（同一 trainer / forward，靠开关区分）

| | 变体 1：双模型（默认） | 变体 2：自举 self |
|---|---|---|
| Shell 函数 | `Train_StereoSplat_Stage2_With_Conf_And_Difix3D` | `Train_StereoSplat_Stage2_Self_Pseudo` |
| 开关 | 无 | `--self_pseudo` |
| pseudo 来源 | **冻结 Stage1**（额外一份权重，全程不变） | **模型自己**当前权重（`no_grad`+临时 `eval` 关 dropout，渲染后恢复 `train`） |
| `--stage_1_model_path` 作用 | 加载到**冻结 Stage1** | 作为 **student 初始化权重**（resume 时被覆盖） |
| 显存 | 多一份冻结模型 | 省一份 |
| work_dir | `stage2_resume/` | `stage2_self_pseudo/` |
| 特点 | pseudo 分布稳定 | self-training，pseudo 随训练变（moving target） |

#### 启动规则

编辑 `train_stereosplat_stage2.sh` **底部** 选一个函数（注释 / 取消注释）：

```bash
Train_StereoSplat_Stage2_With_Conf_And_Difix3D       # 双模型（默认）
# Train_StereoSplat_Stage2_Self_Pseudo               # 自举（--self_pseudo）
```

- 两个函数各自完整独立（变量不共享），`work_dir` 分开，互不干扰。
- `resume_from="latest"`：work_dir 有 checkpoint 就续训（加载全部 optimizer / global_iter 状态）；没有则
  - 双模型：用 `--stage_1_model_path` 初始化 student，另加载冻结 Stage1；
  - 自举：用 `--stage_1_model_path` 初始化 student（只加载权重，不恢复 optimizer）。
- `resume_from=""`（空字符串，等价于 None）：强制从头开始，用 `--stage_1_model_path` 初始化。
- 多卡走 `accelerate_config.yaml`（4 GPU，fp16），**不要**手动设 `CUDA_VISIBLE_DEVICES`。
- 已内置 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`，Difix3D 基座走本地 HF 缓存避免 504。

#### 关键 CLI 选项（两个 trainer 共有 + Stage2 专有）

| 参数 | 说明 |
|------|------|
| `--stage_1_model_path` | 冻结 Stage1 权重 / 或 self 模式的 student 初始化权重 |
| `--mix_psuedo_views_ratio` | pseudo view 混入概率，脚本默认 **`0.9`** |
| `--mix_difix3d_ratio` | Difix3D 应用概率（独立于 pseudo 混入），脚本默认 **`0.9`** |
| `--pretrained_difix3d` | Difix3D 权重；配 `--use_ref` 用参考帧 |
| `--self_pseudo` | 开启自举单模型（见变体 2） |
| `--resume_from` | `""` / `latest` / 具体 checkpoint 路径 |

> 推理脚本**无需改动**：两种变体产出的 checkpoint 与普通 Stage2 完全同构（15D conf），eval 时只需把 `--pretrained_model_path` 指向对应 work_dir 的 checkpoint。

#### Validation 输出格式（每个 `step-N` 目录）

每次 validation 只产出 **一个文件**：`fusion_metric.json`，包含三个 section：

```json
{
  "2view": {
    "psnr_first":  ...,
    "psnr_center": ...,
    "psnr_last":   ...,
    "psnr_mean":   ...
  },
  "pseudo_multiview": {
    "psnr_first":  ..., "psnr_center": ..., "psnr_last": ..., "psnr_mean": ...
  },
  "pseudo_fused": {
    "psnr_first":  ..., "psnr_center": ..., "psnr_last": ..., "psnr_mean": ...
  }
}
```

| Section | 说明 |
|---------|------|
| `2view` | 纯 2-view GT（first stereo）baseline，无 pseudo，无 Difix |
| `pseudo_multiview` | **6 view 输入**（2 GT first + 4 pseudo: center+last stereo），multiview 3DGS 直接渲染 |
| `pseudo_fused` | 同上，但用 pixel-wise conf 融合（2-view GS vs multiview GS） |

> **目标**：`pseudo_fused.psnr_mean > 2view.psnr_mean`（融合后优于纯 2-view baseline）。  
> Progressive pass 用 `view_num=6, mix_psuedo_views_ratio=1.0, mix_difix3d_ratio=1.0`，与 inference 对齐。

Wandb 对应 key：`val/2view/psnr_mean`、`val/pseudo_multiview/psnr_mean`、`val/pseudo_fused/psnr_mean`。

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
│    在 ① 基础上：全轨迹渲染 → pseudo_ratio 选 pseudo stereo   │
│    → 可选 Difix3D → re-inject → 再 forward → 渲染评估         │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ ③ pixel_fusion（S+ + 2D conf 融合）                            │
│    在 ② 基础上：两路 render 按 per-pixel conf 融合              │
│    （whole：2-view GS vs pseudo-multiview GS；                  │
│      separated：Stage1 render vs Stage2 render）              │
│    可选 A1：--conf_fusion_margin（base 优先 + margin）          │
├─────────────────────────────────────────────────────────────┤
│ ③′ GS voxel fusion（3D conf 融合，eval 扩展）                  │
│    G_base vs G_plus 按体素 mean(conf) + margin + base_thresh   │
│    → G_gs_fused → 单次渲染                                     │
├─────────────────────────────────────────────────────────────┤
│ ③″ GS + Pixel 联合（eval 扩展）                                │
│    先 ③′，再 G_base 渲染 vs G_gs_fused 渲染做 pixel 融合       │
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
| **模型函数** | `infer_stereosplat_two_gt_views_forward()` |
| **架构** | 仅 `whole`（单模型） |
| **Difix3D** | 不使用 |
| **conf 融合** | 无 |
| **权重** | 仅 `--pretrained_model_path` |

**stereosplat 模式下，输入就是 2 张 GT stereo view（first view pair）。**

Dataloader 在 val 时固定 `input_view_indices = [1, 0, 2]`（first / center / last），模型里 `prepare_input_multiview(view_num=2)` 取 `index=[0,3]`，即 **左右目各一张 first frame 的 GT 图**。  
Stage1 / Stage2 两个 dataloader 文件在这段 val 逻辑上**几乎相同**；Stage2 版仅多塞了一个 `input_info_for_psuedo_view_rendering` 字段，**stereosplat 纯 forward 不会用到**。

因此 **stereosplat 模式不存在你说的那种「Stage2 输入协议 vs Stage1 输入协议」差别**——输入都是 2 GT view。  
`world_center=First_Stage2` 主要是 **Stage2 训练**（pseudo-GT mix、S+、pixel_fusion）时用的 config 标记；对 stereosplat 评估来说，换 config 基本不改变这 2 张输入图。

| Shell | 加载的权重 | 说明 |
|-------|------------|------|
| `stage1/stereosplat_two_gt_views_forward.sh` | Stage1 | 评 Stage1 模型的基础 2-view 能力 |
| `stage2/stereosplat_two_gt_views_forward.sh` | Stage2 | 评 Stage2 模型的基础 2-view 能力（算法与输入同上，**仅权重不同**） |

> 用 Stage1 权重做 stereosplat 评估 → 跑 `stage1/stereosplat_two_gt_views_forward.sh` 即可。

---

### 方式 ②：stereosplat_plus（StereoSplat+）

在 stereosplat 基础上增加 **pseudo view → Difix enhance → re-inject → 再 forward**。  
分两种实现路径（**whole 与 separated 底层函数不同**）：

#### ②-A whole — 单 checkpoint pose injection（推荐基线）

| 项 | 说明 |
|----|------|
| **做什么** | 同一 checkpoint 内两次前向：2-view 建 GS → 全轨迹渲染 → 按 `pseudo_ratio` 选第二/第三组 pseudo stereo → 可选 Difix → reinject → 再评估 |
| **模型函数** | `infer_stereosplat_plus_pose_injection_single_model()` |
| **架构** | `whole` |
| **权重** | 仅 `--pretrained_model_path` |
| **Difix3D** | 可选：`--use_diffix3d --use_ref` |
| **pseudo_ratio** | `--pseudo_ratio 0.5 1.0`（默认 center+last，与原 progressive 行为等价） |

对应 Shell：`stage1/stereosplat_plus_progressive_single_model.sh`、`stage2/stereosplat_plus_progressive_single_model.sh`

#### ②-B separated — 冻结 Stage1 + Stage2 双模型

| 项 | 说明 |
|----|------|
| **做什么** | **Stage1 冻结**：2-view → 初始 3DGS / pseudo 渲染；**Stage2 主模型**：re-inject 后推理与评估 |
| **模型函数** | `infer_stereosplat_plus_frozen_stage1_two_models()` |
| **架构** | `separated`（**仅 Stage2 评估**，Stage1 无 separated） |
| **权重** | `--pretrained_model_path`（Stage2）+ `--stage_1_model_path`（冻结 Stage1） |
| **Difix3D** | 可选 |
| **pseudo_ratio** | `--pseudo_ratio 0.5 1.0`（控制 pseudo 视角比例） |

对应 Shell：`stage2/stereosplat_plus_progressive_frozen_stage1_two_models.sh`（冻结 Stage1 + Stage2 双模型）  
（Stage1 只有 whole，无 separated S+。）

---

### 方式 ③：pixel_fusion（S+ + 逐像素 confidence 融合）

在 S+ 管线上，对 **两路 novel-view 渲染** 按 confidence **逐像素**选更好的一路（`fuse_renders_by_conf_pixelwise`）。

| 融合开关 | CLI | 行为 |
|----------|-----|------|
| fusion **off** | 不加 `--conf_pixel_level_fusion` | 走完整 S+ 管线，但不融合（与改 fusion 前的 deactivate 一致） |
| fusion **on（legacy）** | `--conf_pixel_level_fusion` | 逐像素 `conf_b >= conf_a` 取 B（平局偏 plus） |
| fusion **on（A1）** | `--conf_pixel_level_fusion --conf_fusion_margin 0.05` | `conf_b > conf_a + margin` 才选 B（平局偏 base） |

#### ③-A whole — 单 checkpoint，融合两路内部 render

| 项 | 说明 |
|----|------|
| **融合的两路** | 同 checkpoint：**2-view GS 渲染** vs **pseudo-multiview GS 渲染** |
| **模型函数** | `infer_pixel_fusion_pose_injection_single_model(pixel_level_conf_fusion=...)` |
| **架构** | `whole` |
| **权重** | 仅 `--pretrained_model_path` |
| **Difix3D** | 通常开启（`--use_diffix3d --use_ref`）；whole pixel_fusion 会预加载 Difix 权重 |
| **pseudo_ratio** | `--pseudo_ratio 0.5 1.0` |

> **注意**：whole 的 `pixel_fusion` 与 whole 的 `stereosplat_plus` **不是同一个函数**。  
> - S+ whole → pose injection + `pseudo_ratio`（无 conf 融合）  
> - pixel_fusion whole → 同上 + 可选 `--conf_pixel_level_fusion` 逐像素融合  

对应 Shell：`stage1/pixel_fusion_pose_injection_single_model.sh`、`stage2/pixel_fusion_pose_injection_single_model.sh`

#### ③-B separated — 融合 Stage1 vs Stage2 渲染

| 项 | 说明 |
|----|------|
| **融合的两路** | **Stage1 冻结模型渲染** vs **Stage2 模型渲染**（同一相机） |
| **模型函数** | `infer_pixel_fusion_pose_injection_frozen_stage1_two_models(pixel_level_conf_fusion=...)` |
| **架构** | `separated` |
| **权重** | Stage2 + 冻结 Stage1（同 ②-B） |
| **Difix3D** | 通常开启 |
| **pseudo_ratio** | `--pseudo_ratio 0.5 1.0` |

对应 Shell：`stage2/pixel_fusion_pose_injection_frozen_stage1_two_models.sh`

---

### 方式 ③′：GS voxel conf 融合（3D，Stage1 whole 已实现）

在 S+ 管线第二次 forward 得到 **G_base** `[N1,15]` 与 **G_plus** `[N2,15]` 后，在 **3D 体素**内融合，再 **只渲染一次**。

| 项 | 说明 |
|----|------|
| **实现** | `utilsdir/gaussain_fusion.py` → `fuse_gaussians_by_voxel_conf_margin` |
| **CLI** | `--gs_conf_fusion`（与 `--eval_mode pixel_fusion` + `whole` 联用） |
| **体素判决** | 同体素内 `mean(conf_base)` vs `mean(conf_plus)`；plus 赢需同时满足：`mean(conf_plus) > mean(conf_base) + margin` 且（若设）`mean(conf_base) < base_thresh` |
| **默认（Stage1 脚本）** | `voxel=0.1`，`gs_margin=0.05`，`conf_agg=mean`，`base_thresh=0.60` |
| **与 pixel 关系** | 单独开 GS：最终指标 = G_gs_fused 渲染；可与 pixel 联合（见 ③″） |

Shell：`stage1/gs_conf_voxel_fusion_pose_injection_single_model.sh`

---

### 方式 ③″：GS + Pixel 联合融合（Stage1 whole）

| 步骤 | 内容 |
|------|------|
| 1 | GS 体素融合 → `G_gs_fused` → 渲染得 `rgb_gs`, `conf_gs` |
| 2 | `G_base` 再渲染得 `rgb_base`, `conf_base` |
| 3 | `fuse_renders_by_conf_pixelwise(rgb_base, rgb_gs, ...)`（**不是** base vs 原始 G_plus） |

CLI 同时传：`--gs_conf_fusion` + `--conf_pixel_level_fusion` + `--conf_fusion_margin`（及 GS 相关参数）。

Shell：`stage1/gs_and_pixel_fusion_pose_injection_single_model.sh`

---

### 消融：Oracle 上界

| 项 | 说明 |
|----|------|
| **CLI** | `--use_gt_view` |
| **模型函数** | `infer_oracle_upper_bound_ablation()` |
| **含义** | **Stage1 推荐**。2-view→G_base；渲染 pseudo→reinject→G_plus；双路渲染后 **用 GT 逐像素选 RGB 误差更小者融合**（非 conf）；主表指标为 `G_fusion` |

---

### 完整对照表（8 种标准推理 + Stage1 融合扩展 + Shell）

| # | Stage | eval_mode | arch | 主 ckpt | 冻结 S1 | Difix | conf fusion | 模型函数（简写） | Shell |
|---|-------|-----------|------|---------|---------|-------|-------------|------------------|-------|
| 1 | 1 | stereosplat | whole | S1 | — | ✗ | ✗ | `infer_stereosplat_two_gt_views_forward` | `stage1/stereosplat_two_gt_views_forward.sh` |
| 2 | 1 | stereosplat_plus | whole | S1 | — | 可选 | ✗ | `infer_stereosplat_plus_pose_injection_single_model` | `stage1/stereosplat_plus_progressive_single_model.sh` |
| 3 | 1 | pixel_fusion | whole | S1 | — | 通常 ✓ | 可选 2D | `infer_pixel_fusion_pose_injection_single_model` | `stage1/pixel_fusion_pose_injection_single_model.sh` |
| 3a | 1 | pixel_fusion + GS | whole | S1 | — | 通常 ✓ | 3D | 同上 + `--gs_conf_fusion` | `stage1/gs_conf_voxel_fusion_pose_injection_single_model.sh` |
| 3b | 1 | pixel_fusion + GS+2D | whole | S1 | — | 通常 ✓ | 3D+2D | 同上 + GS + `--conf_pixel_level_fusion` | `stage1/gs_and_pixel_fusion_pose_injection_single_model.sh` |
| — | 1 | Oracle | whole | S1 | — | 通常 ✓ | GT | `infer_oracle_upper_bound_ablation` | `stage1/oracle_gt_upper_bound_pose_injection.sh` |
| 4 | 2 | stereosplat | whole | S2 | — | ✗ | ✗ | 同 #1 | `stage2/stereosplat_two_gt_views_forward.sh` |
| 5 | 2 | stereosplat_plus | whole | S2 | — | 可选 | ✗ | 同 #2 | `stage2/stereosplat_plus_progressive_single_model.sh` |
| 6 | 2 | stereosplat_plus | separated | S2 | **S1** | 可选 | ✗ | `infer_stereosplat_plus_frozen_stage1_two_models` | `stage2/stereosplat_plus_progressive_frozen_stage1_two_models.sh`（`--pseudo_ratio`） |
| 7 | 2 | pixel_fusion | whole | S2 | — | 通常 ✓ | 可选 | `infer_pixel_fusion_pose_injection_single_model` | `stage2/pixel_fusion_pose_injection_single_model.sh` |
| 8 | 2 | pixel_fusion | separated | S2 | **S1** | 通常 ✓ | 可选 | `infer_pixel_fusion_pose_injection_frozen_stage1_two_models` | `stage2/pixel_fusion_pose_injection_frozen_stage1_two_models.sh` |

Shell 文件名与模型函数名一一对应；路由见 `eval/routes.py`。可视化：shell 末尾取消注释 `# --output_vis`（自动用 `demo.txt`）。

旧函数名在 `stereosplat.py` 类末尾保留为 **Legacy alias**（仍可调用，不推荐），例如：

- `validation_on_the_forward_views` → `infer_stereosplat_two_gt_views_forward`
- `infer_stereosplat_plus_progressive_single_model` / `validation_on_the_forward_views_progressive_iter_once_revised` → `infer_stereosplat_plus_pose_injection_single_model`

---

### 易混淆点

1. **stereosplat 模式：输入恒为 2 GT view**  
   - 评 Stage1 权重 → `stage1/stereosplat_two_gt_views_forward.sh`；评 Stage2 权重 → `stage2/stereosplat_two_gt_views_forward.sh`。

2. **`First_Stage2` config 的真正差别在 S+ / pixel_fusion**  
   - 这些模式会用 pseudo view、双模型、融合等 Stage2 训练配套逻辑；不是 stereosplat 的 2-view 输入变了。

3. **whole 下 S+ 与 pixel_fusion 共用 pose injection + `pseudo_ratio`**  
   - 默认 `0.5/1.0` 与原 progressive（center+last）等价  
   - pixel_fusion 额外支持 2D `--conf_pixel_level_fusion` / A1 `--conf_fusion_margin`，以及 3D `--gs_conf_fusion`（可与 2D 联合，Stage1 已验证）

4. **separated 仅 Stage2 的 S+ / pixel_fusion**  
   - Stage1 训练只有单模型，评估也只有 `whole`。

5. **14D 旧权重**  
   - 本仓库 rasterizer / `gs_dim=15` 不支持直接加载无 conf 旧 ckpt 跑上述流程。

### `pseudo_ratio`：第二 / 第三组 stereo（pose view selection）怎么选

你说的 **pose view selection** 就是这件事：在 **pose injection** 类推理里，除了固定的 **第一组 GT first stereo（2 张）**，还要再选 **两组 pseudo stereo** 作为第二、第三组输入，再 re-inject 做第二次 forward。

| 输入位 | 含义 | 是否可调 |
|--------|------|----------|
| 第 1 组 stereo | GT first view 左右目 | **固定**（始终 2-view GT） |
| 第 2 组 stereo | pseudo view #1 | 由 `pseudo_ratio[0]` 决定 |
| 第 3 组 stereo | pseudo view #2 | 由 `pseudo_ratio[1]` 决定 |

**超参**：`--pseudo_ratio <r2> <r3>`（shell 里常见 `pseudo_ratio="0.50 1.0"`）。

| `pseudo_ratio` | 第二 / 第三组选谁 |
|----------------|-------------------|
| **`0.5 1.0`（默认）** | **center stereo + last stereo**（与 Stage2 训练默认一致；代码里对 `[0.5, 1.0]` 有快速分支） |
| 其他组合 | 沿 bin 内其余渲染轨迹，按比例索引选 stereo pair（见 `prepare_tripleview_by_ratio_index`） |

**哪些 inference 会用 `pseudo_ratio`：**

| 会用 | 不用 |
|------|------|
| `infer_stereosplat_plus_pose_injection_single_model` | `infer_stereosplat_two_gt_views_forward`（纯 2-view） |
| `infer_stereosplat_plus_frozen_stage1_two_models` | |
| `infer_pixel_fusion_pose_injection_single_model` | |
| `infer_pixel_fusion_pose_injection_frozen_stage1_two_models` | |

实现入口：`stereosplat.py` 的 `prepare_tripleview_by_ratio_index()`；pose injection 推理函数里 Difix 增强分支也会按同一套 ratio 取 second/third stereo。

更细的调用链、CLI、FAQ → **[eval/README.md](eval/README.md)**

---

## 评估速查（Shell 与 CLI）

评估逻辑在 **`eval/`**，主入口 **`eval/run.py`**。  
Shell 在 `scripts/evaluation/stage{1,2}/`，**文件名 = 评什么 + 用什么结构**（目录 `stage1/` / `stage2/` 表示用哪套 checkpoint）。  
文件名中仍可见 `progressive` 为历史命名；S+ whole / separated 均已统一 **pose injection + `--pseudo_ratio`**（默认 `0.5 1.0`）。

| 文件名 | 在评什么 |
|--------|----------|
| `stereosplat_two_gt_views_forward.sh` | 最基础：2 张 GT 前向视角直接 forward |
| `stereosplat_plus_progressive_single_model.sh` | S+ pose injection（`pseudo_ratio`），**一个**模型端到端 |
| `stereosplat_plus_progressive_frozen_stage1_two_models.sh` | S+ pose injection（`pseudo_ratio`），**冻结 Stage1 + Stage2** |
| `pixel_fusion_pose_injection_single_model.sh` | pixel_fusion pose injection，**一个**模型（含 A1 margin 函数） |
| `gs_conf_voxel_fusion_pose_injection_single_model.sh` | **仅 GS 体素融合**（Stage1） |
| `gs_and_pixel_fusion_pose_injection_single_model.sh` | **GS + Pixel 联合**（Stage1） |
| `oracle_gt_upper_bound_pose_injection.sh` | Oracle GT 融合上界（Stage1） |
| `pixel_fusion_pose_injection_frozen_stage1_two_models.sh` | pixel_fusion，**冻结 Stage1 + Stage2** 两个模型 |

函数名（脚本**最底部**注释切换要跑哪一个）：

| 函数名 | 什么时候用 |
|--------|------------|
| `run_metric_eval` | 只有这一种跑法（2-view 基础评估） |
| `run_without_difix3d` | S+ / pixel_fusion，不用 Difix3D 修 pseudo |
| `run_with_difix3d` | S+ / pixel_fusion，用 Difix3D 修 pseudo |
| `run_without_conf_pixel_level_fusion` | pixel_fusion，不做逐像素 conf 融合 |
| `run_with_conf_pixel_level_fusion` | pixel_fusion，legacy `--conf_pixel_level_fusion` |
| `run_with_conf_pixel_level_fusion_margin_a1` | pixel_fusion，A1 `--conf_fusion_margin` |
| `run_gs_conf_voxel_fusion` | 仅 GS 3D 融合（`gs_conf_voxel_fusion_*.sh`） |
| `run_gs_and_pixel_fusion` | GS + Pixel 联合（`gs_and_pixel_fusion_*.sh`） |

每个函数内自包含：`output_folder`、`pretrained_model_path`、`stage_1_model_path`、GPU yaml 等；**改路径直接编辑对应 `.sh`**。

### Stage1 / Stage2 完整文件列表

```
scripts/evaluation/
├── stage1/                                          # 加载 Stage1 checkpoint
│   ├── stereosplat_two_gt_views_forward.sh
│   ├── stereosplat_plus_progressive_single_model.sh
│   ├── pixel_fusion_pose_injection_single_model.sh
│   ├── gs_conf_voxel_fusion_pose_injection_single_model.sh
│   ├── gs_and_pixel_fusion_pose_injection_single_model.sh
│   └── oracle_gt_upper_bound_pose_injection.sh
└── stage2/                                          # 加载 Stage2 checkpoint
    ├── stereosplat_two_gt_views_forward.sh
    ├── stereosplat_plus_progressive_single_model.sh
    ├── stereosplat_plus_progressive_frozen_stage1_two_models.sh   # 额外需要 stage_1_model_path
    ├── pixel_fusion_pose_injection_single_model.sh
    └── pixel_fusion_pose_injection_frozen_stage1_two_models.sh    # 额外需要 stage_1_model_path
```

旧实验输出路径（`stereosplat_plus_two_stage/...`）见 `scripts/evaluation/stereosplat_plus/conf_fusion/pixel_level/legacy_*.sh`，语义与上表对应项相同。

### 实验矩阵 → Shell（同上表 #1–#8）

| Stage | eval_mode | architecture | 推荐 Shell |
|-------|-----------|--------------|------------|
| 1 | stereosplat | whole | `stage1/stereosplat_two_gt_views_forward.sh` |
| 1 | stereosplat_plus | whole | `stage1/stereosplat_plus_progressive_single_model.sh` |
| 1 | pixel_fusion | whole | `stage1/pixel_fusion_pose_injection_single_model.sh` |
| 1 | pixel_fusion + GS | whole | `stage1/gs_conf_voxel_fusion_pose_injection_single_model.sh` |
| 1 | pixel_fusion + GS+2D | whole | `stage1/gs_and_pixel_fusion_pose_injection_single_model.sh` |
| 1 | Oracle | whole | `stage1/oracle_gt_upper_bound_pose_injection.sh` |
| 2 | stereosplat | whole | `stage2/stereosplat_two_gt_views_forward.sh` |
| 2 | stereosplat_plus | whole | `stage2/stereosplat_plus_progressive_single_model.sh` |
| 2 | stereosplat_plus | separated | `stage2/stereosplat_plus_progressive_frozen_stage1_two_models.sh` |
| 2 | pixel_fusion | whole | `stage2/pixel_fusion_pose_injection_single_model.sh` |
| 2 | pixel_fusion | separated | `stage2/pixel_fusion_pose_injection_frozen_stage1_two_models.sh` |

### 快速运行

```bash
cd stereosplat

# Stage2 基础 2-view
bash scripts/evaluation/stage2/stereosplat_two_gt_views_forward.sh

# Stage2 S+，冻结 Stage1
bash scripts/evaluation/stage2/stereosplat_plus_progressive_frozen_stage1_two_models.sh

# Stage2 pixel fusion，冻结 Stage1
bash scripts/evaluation/stage2/pixel_fusion_pose_injection_frozen_stage1_two_models.sh

# Stage1 GS 体素融合 / GS+Pixel 联合 / Oracle 上界
bash scripts/evaluation/stage1/gs_conf_voxel_fusion_pose_injection_single_model.sh
bash scripts/evaluation/stage1/gs_and_pixel_fusion_pose_injection_single_model.sh
bash scripts/evaluation/stage1/oracle_gt_upper_bound_pose_injection.sh
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
  --pseudo_ratio 0.5 1.0 \
  --use_diffix3d --use_ref --conf_pixel_level_fusion
```

### 常用参数

| Flag | 说明 |
|------|------|
| `--pseudo_ratio` | 第二/第三组 pseudo stereo 比例，如 `0.5 1.0`（`stereosplat_plus` / `pixel_fusion`；未传时默认 `0.5 1.0`） |
| `--output_vis` | 可视化模式（见下节） |
| `--use_diffix3d` / `--use_ref` | 启用 Difix3D + stereo ref |
| `--conf_pixel_level_fusion` | 2D 逐像素 conf 融合 |
| `--conf_fusion_margin` | A1：选 B 需 `conf_b > conf_a + margin`；需与 `--conf_pixel_level_fusion` 同开 |
| `--gs_conf_fusion` | 3D GS 体素 conf 融合 G_base/G_plus |
| `--gs_fusion_voxel_size` | 体素边长（米），默认 `0.1` |
| `--gs_fusion_margin` | 体素内 plus 赢所需 conf 优势，默认 `0.05` |
| `--gs_fusion_conf_agg` | `mean`（默认）或 `max` |
| `--gs_fusion_base_conf_thresh` | base 优先：仅当体素 `mean(conf_base)` 低于此值才允许 plus 赢 |
| `--use_gt_view` | Oracle 上界（Stage1） |

### 旧路径（仍可用，flat 自包含，与 stage shell 写法一致）

| 旧 Shell / Validator | 等效语义 |
|----------------------|----------|
| `stereosplat/render_inside_bin.sh` | stage2 stereosplat whole |
| `stereosplat_plus/render_inside_bin_whole_model.sh` | stage2 stereosplat_plus whole |
| `conf_fusion/pixel_level/legacy_pixel_fusion_pose_injection_frozen_stage1_two_models_stage2.sh` | 同 stage2 双模型 pixel_fusion（legacy 输出） |
| `conf_fusion/pixel_level/legacy_pixel_fusion_pose_injection_single_model_stage2.sh` | 同 stage2 单模型 pixel_fusion（legacy 输出） |
| `conf_fusion/pixel_level/legacy_pixel_fusion_pose_injection_single_model_stage1.sh` | 同 stage1 单模型 pixel_fusion（legacy 输出） |
| `validator/*.py` | 薄 wrapper → `eval/run.py` |

`conf_fusion/pixel_level/` 输出目录仍用 `stereosplat_plus_two_stage/...` 旧路径；新实验推荐 `stage{1,2}/`。

---

## 可视化

**已实现**：`eval/run.py` 支持 `--output_vis`，模型侧 `vis=True` 保存图像。

| 项 | 说明 |
|----|------|
| 数据列表 | 默认 `filenames/kitti360/train_complete/demo.txt`（9 bins） |
| 扩展列表 | `demo_more.txt`（27 bins），`--demo_filelist` 手动指定 |
| 输出 | `output_folder/<bin_token>/rendered_images/` 等 |
| 指标 | 可视化模式**不写** `metric.json` |

在对应 shell 的 `pixi run accelerate launch ...` 末尾取消注释 `# --output_vis` 即可；或直接 CLI 加 `--output_vis`。

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
│       ├── stage1/          # 7 个 shell（含 gs / gs+pixel / oracle）
│       ├── stage2/          # 5 个（含 frozen_stage1_two_models 双模型）
│       └── stereosplat_plus/conf_fusion/pixel_level/  # legacy_*.sh
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
| **本文件** | 安装、训练、**推理方式详解（8 种 + GS/联合融合 + Oracle）**、Shell 速查、可视化 |
| **[eval/README.md](eval/README.md)** | 评估深入：调用链 mermaid、CLI 全参数、legacy 映射、FAQ |
| 各 `stage{1,2}/*.sh` 函数内 | checkpoint、`output_folder`、GPU、Difix 等（主配置位置） |

---

## Quick Reference

```bash
pixi install -e cu118 && pixi run -e cu118 setup

bash scripts/train/stereosplat/train.sh
bash scripts/train/stereosplat/train_stereosplat_stage2.sh

bash scripts/evaluation/stage2/stereosplat_two_gt_views_forward.sh
bash scripts/evaluation/stage2/pixel_fusion_pose_injection_frozen_stage1_two_models.sh

# 可视化：shell 里取消注释 # --output_vis
```
