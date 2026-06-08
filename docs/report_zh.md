# StereoSplat+ 研究进展报告

**项目**: StereoSplat+: Feed-Forward Gaussian Splatting with Diffusion-Assisted Progressive Inference  
**报告人**: Liu Zihua  
**日期**: 2026-06-08  
**评估数据集**: KITTI-360 val set (5485 bins)

---

## 1. 研究背景与问题定义

### 1.1 StereoSplat 基线系统

StereoSplat 是一个 feed-forward 3D Gaussian Splatting 方法，输入一对立体图像（stereo pair），通过单次前向推理直接预测 3D Gaussians，用于自动驾驶场景（KITTI-360）的新视角合成。模型输入为 first-frame stereo pair（2 views），输出覆盖整个轨迹的 3D Gaussian 场景表示。

### 1.2 StereoSplat+ 的核心思想

StereoSplat+ 通过 **渐进式推理（Progressive Inference）** 扩展 StereoSplat：
1. 第一步：用 first-frame stereo 生成初始 3DGS（G_base）
2. 第二步：从 G_base 渲染轨迹中间/末尾的 pseudo stereo pairs
3. 第三步：将 pseudo pairs 注入模型重新推理，得到 G_plus

### 1.3 核心问题：Pseudo View 注入性能退化

| 方法 | PSNR | SSIM | 说明 |
|------|------|------|------|
| StereoSplat (baseline) | 20.42 | 0.67 | 仅 2-view GT |
| StereoSplat+ (有 GT leakage bug) | 21.03 | 0.73 | 中心 GT 泄漏 |
| StereoSplat+ (修正后, with Difix3D) | 19.89 | 0.66 | 退化 -0.53 |
| StereoSplat+ (修正后, without Difix3D) | 19.78 | 0.65 | 退化 -0.64 |

**结论**：修正 bug 后，pseudo view 注入反而**降低**了性能。原因是 pseudo views 与 GT views 存在 distribution shift——模型训练时只见过 GT 输入，推理时接收到质量不同的渲染图。

---

## 2. 三条改进路径

| 路径 | 策略 | 目标 |
|------|------|------|
| **Path 1** | 训练更强的图像增强器（Difix3D） | 缩小 pseudo/GT 分布差异 |
| **Path 2** | 两阶段 Pseudo-GT 混合训练 | 使模型对 pseudo 输入鲁棒 |
| **Path 3** | 设计更有效的 G_base / G_plus 融合策略 | 鲁棒推理 |

**本报告聚焦 Path 3**：基于 per-Gaussian confidence 的融合策略。

---

## 3. 技术实现：Per-Gaussian Confidence 渲染

### 3.1 系统组件概览

为支持 confidence-based 融合，本工作对 3DGS 管线做了以下修改：

- **15D Gaussians**：在原始 14D（xyz, rotation, scale, opacity, SH）基础上增加 1D confidence
- **自定义 Rasterizer**：修改 diff-gaussian-rasterization CUDA kernel，支持 `rendered_conf` 输出通道
- **Confidence 监督**：在训练中加入自监督 conf loss
- **融合模块**：实现 pixel-level fusion、GS voxel fusion、联合融合、Oracle 上界

### 3.2 Gaussian Prediction Head 架构

模型输出 15D Gaussian 参数：

| 参数 | 维度 | 激活函数 | 含义 |
|------|------|----------|------|
| xyz offset | 0:3 | exp (delta) | 3D 位置偏移 |
| opacity | 3:4 | sigmoid | 不透明度 |
| scale | 4:7 | exp(x) * 0.01 | 3D 尺度 |
| rotation | 7:11 | normalize | 四元数旋转 |
| RGB | 11:14 | sigmoid | 颜色 |
| **confidence** | **14:15** | **sigmoid** | **质量确信度 ∈ (0,1)** |

网络结构（`custom_gs_head.py`）：

```
Input: concat(image[3], depth[1], match_prob[1], upsampled_features[C])
  → gaussian_regressor: Conv2d(in, 64) → GELU → Conv2d(64, 64)
  → concat(regressor_out[64], image[3], features[C], match_prob[1])
  → gaussain_aggregator: Conv2d(in, 128) → GELU → Conv2d(128, 128)
  → gaussian_head: Conv2d(128, 15) → GELU → Conv2d(15, 15)
  → 分别对各参数施加激活函数
```

关键点：**conf 与其他 Gaussian 参数共享同一个预测 head**，最后一维通过 sigmoid 映射到 (0,1)。

### 3.3 Confidence 渲染方式

Confidence 的渲染方式与 RGB 颜色完全相同，采用 **alpha-blending**（体积渲染）：

```
rendered_conf(pixel) = Σ_i conf_i × α_i × T_i
```

其中：
- `conf_i`：第 i 个 Gaussian 的 per-Gaussian confidence 值
- `α_i`：第 i 个 Gaussian 在该像素的 opacity（经 2D 高斯衰减后）
- `T_i = Π_{j<i} (1 - α_j)`：累计透射率（前面 Gaussians 的遮挡）

CUDA 实现（`forward.cu`）：
```cuda
float CONF = 0;
// 对每个 Gaussian 前到后累加:
CONF += geom_confs[collected_id[j]] * alpha * T;
```

输出：`rendered_conf [B, V, 1, H, W]`，与 rendered_image 同分辨率。

### 3.4 Confidence 监督方式

#### 监督信号：自监督光度 soft label

```python
# stereosplat.py, lines 1358-1371
with torch.no_grad():
    l1_err = torch.abs(rendered_image.detach() - gt_image)  # [B,V,3,H,W]
    l1_err = l1_err.mean(dim=2, keepdim=True)               # [B,V,1,H,W] 通道均值
    conf_gt = torch.exp(-conf_lambda * l1_err)              # (0, 1]
```

**语义**：
- 渲染图与 GT 完全一致的区域 → `l1_err ≈ 0` → `conf_gt ≈ 1`（高 confidence）
- 渲染质量差的区域 → `l1_err` 大 → `conf_gt → 0`（低 confidence）
- 梯度停止（`torch.no_grad()`）：conf_gt 不参与反向传播，仅作为伪标签

#### 损失函数

```python
conf_loss = MSE(rendered_conf, conf_gt)
fusion_branch_loss += weight_conf * conf_loss
```

#### 超参数

| 超参数 | 值 | 含义 |
|--------|------|------|
| `use_conf_loss` | `True` | 启用 conf 监督 |
| `conf_lambda` | `10.0` | 指数衰减锐度：越大 conf 对 error 越敏感 |
| `weight_conf` | `0.1` | Conf loss 在 fusion branch 中的权重 |
| `fusion_branch_weight` | `1.0` | Fusion branch 在总 loss 中的权重 |

**Loss 层级结构**：
```
total_loss = cv_branch_loss × 1.0
           + volume_branch_loss × 1.0
           + fusion_branch_loss × 1.0        ← conf_loss 在这里
           + depth_est_loss × 0.05

其中 fusion_branch_loss = recon_loss × 1.0
                        + perceptual_loss × 0.05
                        + depth_abs_loss × 0.01
                        + conf_loss × 0.1       ← 最终有效权重 = 0.1
```

---

## 4. 实验设计

### 4.1 模型定义

| 模型 | 训练方式 | G_base 来源 | G_plus 来源 | 备注 |
|------|----------|-------------|-------------|------|
| **Stage1 (S1)** | 全 GT views + conf 监督 | S1 处理 2-view GT | S1 处理 pseudo views | 单模型两用 |
| **Stage2 whole** | Pseudo-GT 混合训练 | S2 处理 2-view GT | S2 处理 pseudo views | 单模型两用 |
| **Stage2 separated (S2 sep)** | 冻结 S1 + S2 专用 | **S1 处理 2-view GT** | **S2 处理 pseudo views** | 各司其职 |

Stage2 的训练方式：冻结 S1 产 pseudo views，S2 接收 pseudo+GT 混合输入训练。因此 **S2 sep 的推理方式与训练方式完全一致**（S1 产 pseudo，S2 用），这是它效果最好的根本原因。

### 4.2 评估指标

- **主要指标**：`all_view_psnr_average`（first + center + last 三组 stereo 的平均 PSNR）
- 辅助指标：SSIM, LPIPS, Abs_Rel (depth)
- 逐视角分析：first_view / center_view / last_view 分开统计

### 4.3 融合方法

| 方法 | 逻辑 | 粒度 |
|------|------|------|
| **Progressive only** | 直接用 G_plus 渲染（不做 base/plus 选择） | — |
| **Pixel-level fusion** | 逐像素按 conf 选 G_base 或 G_plus 的渲染 | 像素级 |
| **Pixel fusion + margin** | `conf_plus > conf_base + margin` 才选 plus（平局偏 base） | 像素级 |
| **GS voxel fusion** | 3D 体素内按平均 conf 选 base 或 plus 的 Gaussians | 体素级 (0.1m) |
| **GS + Pixel** | 先 GS 融合再 pixel 融合 | 联合 |
| **Oracle** | 用 GT 逐像素选 L1 error 更小的一路（理论上界） | 像素级 |

---

## 5. 实验结果

### 5.1 Confidence 训练对基线性能的影响

**动机**：验证增加 confidence head 是否损害原始性能。

| 模型 | PSNR | SSIM | LPIPS | Abs_Rel |
|------|------|------|-------|---------|
| S1 (without conf, 14D) | 20.424 | 0.674 | 0.195 | 0.071 |
| S1 (with conf, 15D) | 20.436 | 0.668 | 0.199 | 0.075 |
| S2 (with conf, 15D) | 20.425 | 0.673 | 0.205 | 0.076 |

**结论**：Confidence head 对 PSNR 几乎无影响（+0.01），15D 模型是 14D 的合理扩展。

---

### 5.2 Oracle 上界分析

**动机**：确定 confidence-based 融合理论上能达到的最佳性能。

**设计**：用 GT 图像逐像素比较 G_base 和 G_plus 渲染结果，选择更接近 GT 的像素。

| 组件 | all_view PSNR | all_view SSIM |
|------|---------------|---------------|
| G_base（2-view only） | 20.408 | 0.665 |
| G_plus（pseudo multiview） | 19.689 | 0.637 |
| **Oracle Fusion（GT 选择）** | **21.096** | **0.706** |

**逐视角 Oracle 分析**：

| 视角 | G_base | G_plus | Oracle | Oracle 增益 |
|------|--------|--------|--------|-------------|
| First | 25.24 | 23.61 | 26.24 | +1.00 |
| Center | 19.73 | 19.30 | 20.22 | +0.49 |
| Last | 17.59 | 17.27 | 18.11 | +0.52 |
| **All** | **20.41** | **19.69** | **21.10** | **+0.66** |

**关键结论**：
1. G_plus 整体不如 G_base（20.41 vs 19.69），但在**局部区域** G_plus 更好
2. Oracle 可达 21.10，比 baseline **高 +0.66 PSNR**——证明融合策略有巨大潜力
3. First frame 收益最大 (+1.00)：说明即使 G_base 在 first frame 很好（25.24），G_plus 仍有少量 pixel 能贡献

---

### 5.3 三种模型设置的 Progressive 性能对比

**动机**：比较 S1、S2 whole、S2 sep 三种设置下，直接使用 G_plus（不做 fusion）的性能。

| 视角 | S1 (prog+difix) | S2 whole (prog) | S2 sep (prog) | Baseline |
|------|-----------------|-----------------|---------------|----------|
| First | 23.61 | 24.28 | 24.65 | 25.24 |
| Center | 19.30 | 19.47 | 19.79 | 19.73 |
| Last | 17.27 | 17.53 | **17.79** | 17.59 |
| **All** | **19.76** | **20.07** | **20.37** | **20.44** |

**结论**：
1. S2 sep progressive (20.37) 远好于 S1 (19.76)——Stage2 的 pseudo-GT 混合训练有效
2. S2 sep 的 center/last 视角已接近甚至超过 baseline——说明 pseudo views 在远视角确实有用
3. 所有设置的 first frame 都明显低于 baseline——pseudo 输入质量差对 first frame 伤害最大

---

### 5.4 三种模型设置的 Pixel Fusion 性能对比

**动机**：比较不同模型设置下 confidence-based pixel fusion 的效果。

| 视角 | S1 pixel fusion | S2 whole pixel fusion | S2 sep pixel fusion | Baseline |
|------|-----------------|----------------------|---------------------|----------|
| First | 24.07 | 24.80 | **25.36** | 25.24 |
| Center | 19.38 | 19.67 | **19.84** | 19.73 |
| Last | 17.31 | 17.63 | 17.72 | 17.59 |
| **All** | **19.93** | **20.31** | **20.53** | **20.44** |

**与 Baseline 的对比**：

| 设置 | Progressive vs BL | Pixel Fusion vs BL | Fusion 是否有效 |
|------|-------------------|--------------------|--------------------|
| S1 | -0.68 | -0.51 | 无效（仍低于 baseline） |
| S2 whole | -0.37 | -0.13 | 部分有效（接近但未超） |
| **S2 sep** | -0.07 | **+0.10** | **有效（唯一超 baseline）** |

**结论**：只有 S2 separated + pixel fusion 能超过 baseline，且三个视角均有提升。

---

### 5.5 Confidence 偏差的核心发现

**动机**：分析为什么 S1 的 fusion 失败而 S2 sep 成功。

#### Confidence 统计对比

| 设置 | G_base conf (all) | G_plus conf (all) | 偏差 (plus - base) |
|------|--------------------|--------------------|---------------------|
| **S1** | 0.660 | **0.744** | **+0.084** |
| **S2 whole** | 0.663 | 0.668 | +0.005 |
| **S2 sep** | 0.660 | 0.668 | +0.008 |

#### 逐视角 Confidence 对比

| 设置 | G_base: first / center / last | G_plus: first / center / last |
|------|-------------------------------|-------------------------------|
| S1 | 0.743 / 0.649 / 0.597 | 0.752 / **0.758** / **0.749** |
| S2 whole | 0.749 / 0.650 / 0.597 | 0.742 / 0.657 / 0.614 |
| S2 sep | 0.743 / 0.649 / 0.597 | 0.743 / 0.657 / 0.614 |

#### 根因分析

**S1 conf 偏差大的原因**：

S1 训练时只见过 GT 输入。推理时 G_plus 使用 pseudo 输入（GT + pseudo 混合），产生了更多 Gaussians 覆盖同一区域。Alpha-blending 后 conf 天然偏高——模型把 "看到更多输入覆盖" 误解为 "更 confident"。

**S2 sep conf 偏差小的原因**：

S2 训练时就是 separated 架构——冻结 S1 产 pseudo，S2 接收 pseudo 输入。所以 **S2 在训练时已经见过 pseudo views 的 distribution**，它对 pseudo 输入的 conf 预测天然校准过。推理时 S2 处理 pseudo views 的方式和训练时一致，不存在 distribution shift。

**S2 whole 不如 S2 sep 的原因**：

S2 whole 用同一个模型既处理纯 GT（产 G_base）又处理 pseudo（产 G_plus）。但 S2 训练时输入是 pseudo-GT mix，纯 GT 输入不是它的最优工作点。S2 sep 让每个模型处理自己擅长的 distribution：S1 处理 GT → G_base，S2 处理 pseudo → G_plus。

---

### 5.6 S1 Pixel Fusion 逐视角损失分析

**动机**：量化 S1 pixel fusion 在各视角的具体损失和原因。

| 视角 | Baseline | S1 Pixel Fusion | 损失 | Oracle | Oracle 增益 | 原因 |
|------|----------|-----------------|------|--------|-------------|------|
| First | 25.24 | 24.07 | **-1.17** | 26.24 | +1.00 | G_plus conf 0.752 > G_base 0.743，大量 first pixel 被错误替换 |
| Center | 19.73 | 19.38 | -0.35 | 20.22 | +0.49 | G_plus conf 0.758 >> G_base 0.649，偏差最大的视角 |
| Last | 17.59 | 17.31 | -0.28 | 18.11 | +0.52 | G_plus conf 0.749 >> G_base 0.597，同上 |
| **All** | **20.44** | **19.93** | **-0.51** | **21.10** | **+0.66** | |

**核心问题**：S1 的 G_plus conf 在**所有视角**都系统性高于 G_base，导致大量像素被错误替换为质量更差的 G_plus 渲染。First frame 损失最严重（-1.17），是全部损失 (-0.51) 的主要来源。

---

### 5.7 S2 Separated Pixel Fusion 详细分析

**动机**：理解 S2 sep 为什么有效，以及剩余提升空间在哪。

| 视角 | Baseline | S2 sep Progressive | S2 sep Pixel Fusion | Fusion vs Prog | Fusion vs BL |
|------|----------|--------------------|--------------------|----------------|--------------|
| First | 25.24 | 24.65 | **25.36** | +0.71 | **+0.12** |
| Center | 19.73 | 19.79 | **19.84** | +0.05 | **+0.11** |
| Last | 17.59 | **17.79** | 17.72 | -0.07 | **+0.13** |
| **All** | **20.44** | **20.37** | **20.53** | **+0.16** | **+0.10** |

**关键观察**：

1. **First frame: fusion 大幅改善 progressive (+0.71)**。说明 conf 在 first frame 正确选择了 G_base（因为 G_base 在 first frame 明显更好 25.24 vs 24.65）
2. **Center frame: fusion 小幅改善 progressive (+0.05)**。conf 能部分区分 base 和 plus
3. **Last frame: fusion 反而比 progressive 差 (-0.07)**。说明 last frame 有些区域 G_plus 更好（progressive 17.79 > baseline 17.59），但 conf 没有正确识别，反而把部分好的 plus pixel 替换回了 base

**剩余空间**：与 S1 Oracle (21.10) 相比还有 0.57 PSNR 的空间（需要跑 S2 sep 的 oracle 确认其自身天花板）。

---

### 5.8 S1 融合策略消融

**动机**：在 S1 上对比不同融合粒度和方式的效果。

| 方法 | PSNR | SSIM | vs Baseline |
|------|------|------|-------------|
| Baseline (2-view) | 20.436 | 0.668 | — |
| Progressive + Difix3D | 19.762 | 0.656 | -0.67 |
| Pixel fusion (no margin) | 19.925 | 0.660 | -0.51 |
| Pixel fusion + margin 0.05 | 20.007 | 0.662 | -0.43 |
| GS voxel fusion | 19.842 | 0.656 | -0.59 |
| GS + Pixel 联合 | 20.071 | 0.661 | -0.36 |
| **Oracle** | **21.096** | **0.706** | **+0.66** |

**结论**：
1. 所有 S1 融合方法都低于 baseline——根本原因是 conf 系统性偏差
2. Margin (+0.08 over no-margin) 和 GS+Pixel (-0.36 vs GS-only -0.59) 有增量帮助
3. GS voxel fusion 最差——体素粒度太粗，受 conf 偏差影响更严重
4. Oracle 说明 pixel-level 选择有 +0.66 空间，问题纯粹在于选择的**准确性**

---

### 5.9 Margin 策略的效果

| 配置 | Margin | PSNR | 相对 no-margin |
|------|--------|------|----------------|
| S1 pixel fusion | 0 | 19.925 | — |
| S1 pixel fusion | 0.05 | 20.007 | +0.08 |

**原理**：`conf_plus > conf_base + 0.05` 才选 plus，平局偏向 base。因为 G_base 平均质量更好，让系统在不确定时选 base 是合理的保守策略。

---

## 6. 核心结论总结

### 6.1 方法有效性

| 结论 | 证据 |
|------|------|
| Oracle 证明 conf fusion 有 +0.66 PSNR 潜力 | Oracle = 21.10 vs Baseline = 20.44 |
| S2 sep + pixel fusion 是唯一超 baseline 的方案 | 20.53 vs 20.44 (+0.10) |
| Conf 偏差是 S1 fusion 失败的根本原因 | S1 偏差 +0.084 vs S2 sep 偏差 +0.008 |
| 训练-推理 distribution 对齐是关键 | S2 sep 训练时用 separated 方式 → 推理时 conf 天然校准 |

### 6.2 各模型设置的优劣

| 设置 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| S1 whole | 实现简单；Oracle 天花板最高 | Conf 偏差大，fusion 无法正常工作 | 需要偏差校正方案 |
| S2 whole | Conf 偏差小 | G_base 质量不如 S1（S2 不擅长纯 GT 输入） | 不推荐 |
| **S2 sep** | **Conf 偏差小 + G_base 质量高（S1产）** | 需要两个模型 | **推荐的最终方案** |

### 6.3 Confidence 监督的分析

| 方面 | 分析 |
|------|------|
| 监督方式 | 自监督 `exp(-10 × L1_error)`，合理但不完美 |
| 问题 1 | 模型把"多视角覆盖"误解为"高质量"，导致 G_plus conf 虚高（S1） |
| 问题 2 | S2 通过训练时见过 pseudo distribution 天然解决了此问题 |
| 问题 3 | `conf_lambda=10` 在常见 error 范围内区分度有限 |
| 问题 4 | `weight_conf=0.1` 较低，模型学习 conf 的动力有限 |

---

## 7. 进度总结

### 7.1 已完成

| 内容 | 状态 |
|------|------|
| 15D Gaussian + conf 自定义 Rasterizer | Done |
| Stage1 conf 模型训练 | Done |
| Stage2 pseudo-GT 混合训练（whole + separated） | Done |
| Pixel-level conf fusion 实现 & 评估 | Done |
| GS voxel conf fusion 实现 & 评估 | Done |
| GS + Pixel 联合融合 | Done |
| Oracle 上界分析 (S1) | Done |
| 完整 ablation（13 组实验，3 种模型设置） | Done |
| 评估系统统一化（eval/run.py） | Done |
| 根因分析：conf 偏差与 train-test distribution 对齐 | Done |

### 7.2 当前最佳结果

```
方法: Stage2 Separated + Pixel-level Confidence Fusion
PSNR: 20.53 (vs baseline 20.44, +0.10)
SSIM: 0.670
LPIPS: 0.194
```

---

## 8. 未来计划（2 周）

### 8.1 核心思路

基于分析，对 S1 和 S2 sep 采用不同优化策略：

| 线 | 瓶颈 | 策略 | 目标 |
|----|------|------|------|
| S1 whole | Conf 偏差 +0.084 → 大量像素选错 | Per-view margin + conf 校准 | 19.93 → ~20.5-20.7 |
| S2 sep | Conf 偏差小 +0.008 → 选择精度不够 | Soft blending + 精细 margin | 20.53 → ~20.6-20.7 |

### 8.2 具体方案

**方案 A：Per-View Adaptive Fusion（主要针对 S1）**
- First frame 强制使用 base（因为 S1 first frame 被严重伤害：25.24→24.07）
- Center/Last frame 使用 conf fusion with 校准后的 conf
- 预期：S1 all_view 从 19.93 回升到 ~20.5

**方案 B：Conf Calibration（主要针对 S1）**
- Per-image z-score 标准化：消除 G_base/G_plus 的全局 conf 偏差
- 让比较基于相对 conf 排名而非绝对值

**方案 C：Soft Blending（主要针对 S2 sep）**
- `weight_plus = sigmoid((conf_plus - conf_base) × temperature)` 做连续加权
- 减少 hard selection 在 conf 接近时的错误选择代价
- 预期：S2 sep 从 20.53 提升到 ~20.6

### 8.3 需要确认

- S2 separated 的 Oracle 上界（确认天花板）
- Per-view fusion 的 V 维度索引布局（确认代码中 first/center/last 的位置）

---

*报告完*
