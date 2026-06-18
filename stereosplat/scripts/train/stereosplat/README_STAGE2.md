# Stage2 训练说明（Self-Pseudo + Conf + Difix3D）

本文档描述**当前代码**下的 Stage2 训练逻辑、损失项与权重，以及训练收敛后各指标应呈现的趋势。  
对应脚本：`train_stereosplat_stage2.sh`；Trainer：`trainer/train_kitti360_stereosplat_stage2_with_difix3d.py`；Config：`input_invariant_stereosplat_stage2.py`。

---

## 1. 训练目标（一句话）

在 **2-view 质量不劣于 Stage1** 的前提下，让 **multiview 渲染** 和 **pixel-wise conf 融合** 在 pseudo 输入场景下持续变好；最终 validation 上期望：

```
val/pseudo_fused/psnr_mean  ≥  val/2view/psnr_mean + 0.5 dB
val/pseudo_fused/psnr_mean  ≥  val/pseudo_multiview/psnr_mean
```

---

## 2. 两种训练变体

| | 双模型（Two-Model） | 自举（Self-Pseudo，当前激活） |
|---|---|---|
| Shell 函数 | `Train_StereoSplat_Stage2_With_Conf_And_Difix3D` | `Train_StereoSplat_Stage2_Self_Pseudo` |
| 开关 | 无 | `--self_pseudo` |
| Pseudo 来源 | 冻结的 `frozen_stage_1_model` | **当前模型自己**（`no_grad` + 临时 `eval`） |
| `stage_1_model_path` | 仅加载到冻结 Stage1 | **两处**：student 初始化 + `frozen_2v_ref_model` |
| 可训练模型起点 | 随机或 resume checkpoint | Stage1 权重（`resume_from=""` 时） |

### `stage_1_model_path` 的两处用途（Self-Pseudo）

1. **首次初始化**：`my_model.load_state_dict(stage_1)`（仅 `resume_from=""` 时生效；`resume_from="latest"` 会被 accelerate checkpoint 覆盖）。
2. **2v 锚定 teacher**：另建 `frozen_2v_ref_model`，全程冻结，仅用于 `view_num=2` 的 `2v_floor` / `2v_ceiling`。

> `frozen_2v_ref_model` **不**参与 multiview / fusion 里的 2v 参考；那些路径用的是**当前模型**自己的 2v 渲染（`eval` + `no_grad`）。

---

## 3. 每个 iteration 在做什么

### 3.1 随机采样 `view_num`

| `view_num` | 采样概率 | `matching_nums` | 实际输入图数 `stereo_pairs_nums = (view_num+1)//2` |
|------------|----------|-----------------|--------------------------------------------------|
| 2 | 10% | 2 | 1 对 stereo（2 图） |
| 3 | 22.5% | 3 | 2 对（4 图） |
| 4 | 22.5% | 3 | 2 对（4 图） |
| 5 | 22.5% | 5 | 3 对（6 图） |
| 6 | 22.5% | 5 | 3 对（6 图） |

- `view_num=2`：纯锚定步，**不算** multiview GT 重建损失。
- `view_num≥3`：multiview 训练步（占 90%）。

输入索引映射（`prepare_input_multiview_stage2`）：

```
stereo_pairs_nums=1 → [0,3]           # first L/R
stereo_pairs_nums=2 → [0,3,1,4]      # first + center L/R
stereo_pairs_nums=3 → [0,3,1,4,2,5]  # first + center + last L/R
```

### 3.2 Pseudo view 注入

以 `mix_psuedo_views_ratio`（脚本默认 **0.9**）概率：

1. 用 pseudo 源模型（Self-Pseudo 为**当前模型**）从 2-view 渲染 novel views。
2. 再以 `mix_difix3d_ratio`（默认 **0.9**）概率过 Difix3D 增强。
3. 将 `input_batch_dict['imgs'][:, 2:, ...]` 替换为 pseudo 图。

**Case 判定**（`_did_mix_pseudo`）：

| Case | 条件 | 含义 |
|------|------|------|
| **A** | `view_num≥3` 但未注入 pseudo（约 10% 概率） | 全 GT 输入；**跳过** B/C 专用 fusion 损失 |
| **B/C** | `view_num≥3` 且成功注入 pseudo | 开启 `recon_pixel_fused`、margin、`conf_comparative` 等 |

---

## 4. 损失总结构

总损失按分支聚合：

```
total = fusion_branch_loss × branch_weight_fusion(1.0)
      + volume_branch_loss   × branch_weight_volume(1.0)    # 中间项，不打印
      + cv_branch_loss       × branch_weight_cv(1.0)        # 当前 use_cv=False
      + depth_est_loss       × branch_weight_depth_est(0.05)
```

训练步按 `view_num` 分为两条路径：

```
┌─────────────────────────────────────────────────────────────┐
│  view_num = 2  （10%）                                         │
│  仅 2v 锚定：2v_floor + 2v_ceiling                            │
│  对比对象：当前 fusion 渲染 vs frozen Stage1 渲染              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  view_num ≥ 3  （90%）                                        │
│  GT 监督 + conf 自监督 +（B/C 时）pixel-fusion 与 margin 三角  │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 损失项明细与权重

权重来源：`input_invariant_stereosplat_stage2.py` → `loss_args.fusion_sup_dict` 等。

### 5.1 Multiview 步 — GT 监督（`view_num≥3`，每步都有）

| 日志名 | 含义 | 权重 | 备注 |
|--------|------|------|------|
| `loss_recon_gs` | MSE(multiview fusion 渲染 RGB, GT) | `weight_recon = 1.0` | 主 RGB 重建 |
| `loss_perceptual_gs` | LPIPS(multiview fusion, GT) | `weight_perceptual = 0.05` | 感知质量 |
| `loss_depth_abs_gs` | L1(rendered depth, sparse+pseudo depth GT) | `weight_depth_abs = 0.01` | 深度一致性 |
| `loss_depth_est_loss` | 输入侧深度估计监督 | `branch_weight = 0.05`（depth_est_sup_dict） | Custom_Depth_Loss |
| `loss_conf_gs` | MSE(rendered_conf, soft photometric label) | `weight_conf = 0.1` | `conf_gt = exp(-λ·L1)`，λ=`conf_lambda=10` |

Volume 分支（`recon_vol` / `perceptual_vol` / `depth_abs_volume`）同样计算，但 **WandB 与终端不打印**（中间监督）。

### 5.2 Multiview 步 — B/C 专用（`_did_mix_pseudo=True` 时）

| 日志名 | 含义 | 权重 | 约束目标 |
|--------|------|------|----------|
| `loss_recon_pixel_fused` | MSE(pixel_fused_rgb, GT) | `weight_fusion_sup = 1.5` | 融合图贴近 GT |
| `loss_perceptual_pixel_fused` | LPIPS(pixel_fused, GT) | `weight_fusion_sup_percep = 0.3` | 融合感知质量 |
| `loss_conf_comparative` | `ReLU((conf_mv - conf_2v) × (err_mv - err_2v))` | `weight_conf_comparative = 0.3` | mv 更差处 conf 应更低 |
| `loss_fusion_2v_margin` | `ReLU(PSNR_2v + 0.5 - PSNR_fused)` | `weight_fusion_2v_margin = 0.8` | fused 平均 PSNR 比 2v 高 **≥0.5 dB** |
| `loss_fusion_mv_margin` | `ReLU(PSNR_mv - PSNR_fused)` | `weight_fusion_mv_margin = 0.5` | fused ≥ multiview（平均 PSNR） |
| `loss_mv_margin` | `ReLU(PSNR_2v - PSNR_mv)` | `weight_mv_margin = 0.5` | multiview ≥ 当前 2v（平均 PSNR） |

说明：

- **2v 参考**（fusion 块内）：当前模型的 2v 渲染，`detach`；梯度只进 multiview / conf 分支。
- **margin 类**均为 `ReLU` hinge：满足不等式时 loss = 0。
- **PSNR** 在 margin 里按**所有渲染视角**的 per-view MSE 取平均后再算 dB（非仅 center/last）。
- `fusion_2v_psnr_margin = 0.5`；`fusion_mv_psnr_margin` / `mv_psnr_margin` 当前均为 **0**（只需 beat，不需额外 dB 间隙）。

### 5.3 2v 锚定步（`view_num=2`，仅 train）

| 日志名 | 含义 | 权重 | 约束目标 |
|--------|------|------|----------|
| `loss_2v_floor` | `ReLU(MSE_cur - MSE_stage1)` | `weight_2v_floor = 0.5` | 当前 2v **不能比 Stage1 差** |
| `loss_2v_ceiling` | `ReLU(MSE_stage1 - MSE_cur)` | `weight_2v_ceiling = 0.3` | 当前 2v **不能明显好于 Stage1**（带宽锚定） |

对比对象：`rendered_color_fuse`（当前模型 2v 路径）vs `frozen_2v_ref_model` 渲染（Stage1 冻结）。

此步**关闭**：`recon_gs`、`perceptual_gs`、`depth_abs_gs`、`depth_est_loss`、`conf_gs` 及全部 B/C fusion 项。

### 5.4 当前关闭的项

| 项 | 权重 | 说明 |
|----|------|------|
| `2v_floor_mv` | `weight_2v_floor_mv = 0.0` | multiview 步额外 2v forward 锚定，未启用 |
| CV 分支 | `use_cv = False` | 不参与 |
| Case A 的 fusion 专用项 | — | `_did_mix_pseudo=False` 时自动跳过 |

---

## 6. 训练收敛后，各 loss 应如何变化

以下为 **well-trained** 时的期望趋势（WandB / 终端 `train/2v/*` 与 `train/mv/*`）。

### 6.1 `train/2v/*`（约 10% 步）

| 指标 | 健康趋势 | 含义 |
|------|----------|------|
| `loss_2v_floor` | → **0** 且长期贴近 0 | 当前 2v 未跌破 Stage1 |
| `loss_2v_ceiling` | → **0** 或很小 | 未大幅超越 Stage1（防止 2v 漂移过快） |
| `loss_total` | 低且稳定 | 锚定带内，无持续惩罚 |

若 `2v_floor` 长期 > 0：2v 质量在下滑，pseudo 源会变差，需检查学习率或增大锚定权重。

### 6.2 `train/mv/gt/*`（GT 监督）

| 指标 | 健康趋势 |
|------|----------|
| `loss_recon_gs` | 稳步下降后平台 |
| `loss_perceptual_gs` | 下降，略滞后于 recon |
| `loss_depth_abs_gs` | 下降或维持低位 |
| `loss_depth_est_loss` | 下降后低位 |

### 6.3 `train/mv/conf/*`

| 指标 | 健康趋势 |
|------|----------|
| `loss_conf_gs` | 下降；`conf_gs_mean` 与 `conf_gt_mean` 逐渐靠近 |
| `loss_conf_comparative` | → **0**（B/C 步） | conf 排序与 err 差一致 |

### 6.4 `train/mv/margin/*`（B/C 步，关键验收）

| 指标 | 健康趋势 | 对应能力 |
|------|----------|----------|
| `loss_mv_margin` | → **0** | multiview 不比当前 2v 差 |
| `loss_fusion_mv_margin` | → **0** | fused ≥ multiview |
| `loss_fusion_2v_margin` | → **0** | fused 比 2v 高 ≥0.5 dB（平均 PSNR） |

三者同时趋零 ≈ 推理时 pixel fusion 的三角关系成立。

### 6.5 `train/mv/gt/*` 中的融合重建（B/C）

| 指标 | 健康趋势 |
|------|----------|
| `loss_recon_pixel_fused` | 下降 |
| `loss_perceptual_pixel_fused` | 下降 |

### 6.6 Validation（每 `val_freq=5000` step）

路径：`{output_dir}/{exp_name}/validation/step-{N}/fusion_metric.json`

| Section | 健康趋势 |
|---------|----------|
| `2view.psnr_mean` | 接近 Stage1 水平，稳定 |
| `pseudo_multiview.psnr_mean` | 高于 `2view` |
| `pseudo_fused.psnr_mean` | **最高**；比 `2view` 高约 0.5 dB，≥ `pseudo_multiview` |

WandB：`val/2view/*`、`val/pseudo_multiview/*`、`val/pseudo_fused/*`。

---

## 7. 日志与保存

### 7.1 WandB 分组

```
train/2v/loss_total, loss_2v_floor, loss_2v_ceiling     # view_num=2
train/mv/loss_total
train/mv/gt/loss_recon_gs, loss_perceptual_gs, ...        # GT 监督
train/mv/conf/loss_conf_gs, loss_conf_comparative, ...    # conf
train/mv/margin/loss_fusion_2v_margin, ...                # margin 三角
```

终端打印为扁平 `train/loss_*`（过滤 volume/cv 中间项）。

### 7.2 Checkpoint vs Validation

| 类型 | 路径 | 内容 |
|------|------|------|
| Checkpoint | `work_dir/checkpoint-{step}/` | 模型 + optimizer + scheduler + iter（`save_freq=5000`） |
| Validation | `output_dir/.../validation/step-{N}/` | 仅 `fusion_metric.json`（默认不写 PNG） |

---

## 8. 当前脚本默认超参（Self-Pseudo）

```bash
mix_psuedo_views_ratio=0.9
mix_difix3d_ratio=0.9
resume_from=""                    # 首次从 Stage1 初始化
stage_1_model_path=.../checkpoint-145000/
lr=8e-5, warmup_steps=1000, max_train_steps=100000
val_freq=5000, save_freq=5000
resolution=[112, 544]
```

---

## 9. 快速启动

```bash
cd stereosplat
# 编辑 scripts/train/stereosplat/train_stereosplat_stage2.sh 底部，选择函数：
bash scripts/train/stereosplat/train_stereosplat_stage2.sh
```

---

## 10. 相关源码索引

| 文件 | 内容 |
|------|------|
| `stereosplat.py` → `forward_stage2_with_difix3d` | 前向、pseudo 注入、全部 loss |
| `stereosplat.py` → `prepare_input_multiview_stage2` | view_num → 输入图选取 |
| `train_kitti360_stereosplat_stage2_with_difix3d.py` | 采样、frozen 模型、validation、WandB 分组 |
| `input_invariant_stereosplat_stage2.py` | 全部 loss 权重 |

---

## 11. 已知注意点

1. **Validation** 的 progressive pass 固定 `view_num=6, matching_nums=3`，与训练 `view_num=5/6` 时 `matching_nums=5` 不完全一致；看 val 趋势即可，勿与单步训练配置一一对应。
2. **Self-Pseudo** 的 pseudo 来自当前模型，存在 moving target；`2v` 锚定 + `2v_floor/ceiling` 用于抑制漂移。
3. **Case A**（约 10% 的 multiview 步）无 B/C fusion 损失，属正常；margin 指标仅在 B/C 步有值。
4. 打开 validation 可视化需改 trainer 中 `save_visuals=True` 且 `validation_vis_progress=True`。
