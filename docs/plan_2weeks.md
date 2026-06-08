# StereoSplat+ 两周提升计划（详细实施版）

**目标**: 同时在 Stage1 whole 和 Stage2 separated 两条线上优化融合策略  
**时间**: 2 周  
**核心策略**: 不动模型训练（太慢），只改推理阶段的融合逻辑

**两条线并行**：
- **Stage1 whole**: 当前 pixel fusion = 20.01，空间大（oracle = 21.10），验证快（单模型）
- **Stage2 separated**: 当前 pixel fusion = 20.53，需要先跑 oracle 确认天花板，再做精细优化

---

## 1. 现状诊断

### 1.1 数字总览

| 指标 | 值 |
|------|-----|
| Baseline (2-view, S1) | 20.44 |
| 当前最佳 (S2 sep pixel fusion) | 20.53 (+0.10) |
| S1 Oracle 上界 | 21.10 (+0.66) |
| S2 separated Oracle | **未跑，需确认** |

### 1.2 S1 pixel fusion 的逐视角分析——问题出在哪

| 视角 | Baseline (G_base) | S1 pixel fusion | Oracle | fusion 造成的损失 | Oracle 可获得的增益 |
|------|-------------------|-----------------|--------|-------------------|---------------------|
| First | 25.24 | 24.07 | 26.24 | **-1.17** | +1.00 |
| Center | 19.73 | 19.38 | 20.22 | -0.35 | +0.49 |
| Last | 17.59 | 17.31 | 18.11 | -0.28 | +0.52 |
| **All** | **20.44** | **19.93** | **21.10** | **-0.51** | **+0.66** |

**关键发现**：first frame 损失了 1.17 PSNR，占了全部 fusion 损失的大头。

### 1.3 S2 separated pixel fusion 的逐视角分析

| 视角 | Baseline (S2 base) | S2 sep pixel fusion | 损失 |
|------|---------------------|---------------------|------|
| First | 25.24 (S1) | 25.36 | **+0.12** (不受影响) |
| Center | 19.73 | 19.84 | +0.11 |
| Last | 17.59 | 17.72 | +0.13 |
| **All** | **20.44** | **20.53** | **+0.10** |

S2 separated 三个视角都比 baseline 好。但距离 Oracle (21.10) 还有 0.57 的空间。

### 1.4 三种设置全面对比分析

#### (a) Progressive Only (G_plus only, 不做 fusion)

| 视角 | S1 (prog+difix) | S2 whole (prog) | S2 sep (prog) | Baseline |
|------|-----------------|-----------------|---------------|----------|
| First | 23.61 | 24.28 | 24.65 | 25.24 |
| Center | 19.30 | 19.47 | 19.79 | 19.73 |
| Last | 17.27 | 17.53 | **17.79** | 17.59 |
| **All** | **19.76** | **20.07** | **20.37** | **20.44** |

#### (b) Pixel Fusion (G_base vs G_plus by conf)

| 视角 | S1 pixel fusion | S2 whole pixel fusion | S2 sep pixel fusion | Baseline |
|------|-----------------|----------------------|---------------------|----------|
| First | 24.07 | 24.80 | **25.36** | 25.24 |
| Center | 19.38 | 19.67 | **19.84** | 19.73 |
| Last | 17.31 | 17.63 | 17.72 | 17.59 |
| **All** | **19.93** | **20.31** | **20.53** | **20.44** |

#### (c) Confidence 偏差对比（核心数据）

| 设置 | G_base conf (all) | G_plus conf (all) | 偏差 (plus - base) |
|------|--------------------|--------------------|---------------------|
| **S1** | 0.660 | **0.744** | **+0.084** |
| **S2 whole** | 0.663 | 0.668 | +0.005 |
| **S2 sep** | 0.660 | 0.668 | +0.008 |

逐视角 conf 对比：

| 设置 | G_base first/center/last | G_plus first/center/last |
|------|--------------------------|--------------------------|
| S1 | 0.743 / 0.649 / 0.597 | 0.752 / 0.758 / 0.749 |
| S2 whole | 0.749 / 0.650 / 0.597 | 0.742 / 0.657 / 0.614 |
| S2 sep | 0.743 / 0.649 / 0.597 | 0.743 / 0.657 / 0.614 |

#### (d) Fusion vs Baseline 总结

| 设置 | Progressive vs BL | Pixel Fusion vs BL | Fusion 是否有效 |
|------|-------------------|--------------------|--------------------|
| S1 | -0.68 | -0.51 | 无效（conf 偏差太大） |
| S2 whole | -0.37 | -0.13 | 部分有效（接近但未超） |
| S2 sep | -0.07 | **+0.10** | **有效（唯一超 baseline）** |

### 1.5 根因分析：为什么三种设置差异巨大

#### 结论 1：Conf 偏差的根源是"S1 模型面对 pseudo 输入时 conf 虚高"

- S1 progressive：S1 自己产 pseudo → S1 自己处理 pseudo → G_plus conf 大幅虚高 (+0.084)
- S2 progressive：冻结 S1 产 pseudo → S2 处理 pseudo → G_plus conf 偏差极小 (+0.005)

**原因**：S2 训练时就是 separated 方式——冻结 S1 产 pseudo，S2 接收 pseudo 输入。所以 **S2 已经见过 pseudo 的 distribution**，它对 pseudo 输入的 conf 预测是校准过的。而 S1 只见过 GT，面对 pseudo 时 conf 就失真了。

#### 结论 2：S2 whole vs S2 sep 差异的原因

S2 whole pixel fusion (20.31) < S2 sep pixel fusion (20.53)。

- S2 whole：**同一个 S2 模型**既产 G_base（2-view GT）又产 G_plus（pseudo 输入）
- S2 sep：**S1 产 G_base**（它擅长 GT 输入），**S2 产 G_plus**（它擅长 pseudo 输入）

S2 sep 好因为每个模型处理自己训练时见过的 distribution。S2 whole 的 G_base 反而差——因为 S2 训练时输入是 pseudo-GT mix，纯 GT 输入不是它的最优工作点。

#### 结论 3：对 S1 和 S2 sep 需要不同的融合策略

| 设置 | 核心瓶颈 | 需要的策略 |
|------|----------|------------|
| **S1** | Conf 系统性偏差 (+0.084) 导致选错 | 偏差校正 + per-view margin |
| **S2 sep** | Conf 偏差小 (+0.008) 但精度不够 | 提升选择精度 / soft blending |

S2 sep 的 pixel fusion 已经在正确工作（三视角均 > baseline），问题是 "选得不够准" 而非 "系统性选错"。

### 1.6 Confidence 系统性偏差详解（S1 的问题）

从 S1 pixel fusion 的 conf 统计可以看到：

| 来源 | First conf | Center conf | Last conf | All conf |
|------|------------|-------------|-----------|----------|
| G_base (2-view) | 0.743 | 0.649 | 0.597 | 0.660 |
| G_plus (pseudo mv) | 0.752 | 0.758 | 0.749 | 0.744 |

**G_plus 的 conf 在所有视角都高于 G_base**，但它的实际 PSNR 更低（19.69 vs 20.41）。

**原因**：模型训练时 `conf_gt = exp(-10 × L1_error)`，学到的是"这个区域渲染得好不好"。但 G_plus 输入了更多视角 → 更多 Gaussians 叠加到同一区域 → alpha-blending 后 conf 天然偏高，不代表实际质量更好。

**结论：当前 fusion 的核心瓶颈不是模型能力，而是 fusion decision logic 对所有视角"一刀切"且被 conf 偏差误导。**

---

## 2. 整体方案设计

### 2.1 核心思路

基于 1.5 的分析，对 S1 和 S2 sep 采用**不同策略**：

| 线 | 瓶颈 | 核心策略 | 目标 |
|----|------|----------|------|
| **S1 whole** | Conf 偏差 +0.084 导致大量 pixel 被错误替换 | 偏差校正 + per-view 保护 | 从 19.93 提到 ~20.5-20.7 |
| **S2 sep** | Conf 偏差小，但选择精度不够 | Soft blending + 精细调参 | 从 20.53 提到 ~20.6-20.7 |

**为什么两线策略不同？**

- S1 的 first frame 被严重伤害（25.24→24.07），因为 G_plus conf 虚高 0.084。修复偏差就能回收大量分数。
- S2 sep 的 first frame 反而更好（25.24→25.36），conf 偏差只有 0.008。对它做偏差校正没意义——需要的是让 "选对的比例" 更高。

### 2.2 方案分层

| 层级 | 方案 | 主要目标线 | 核心机制 |
|------|------|------------|----------|
| **L1: Per-view strategy** | S1 为主 | 不同视角不同 margin（保护 S1 first frame） |
| **L2: Conf calibration** | S1 为主 | zscore/minmax 消除系统性偏差 |
| **L3: Soft blending** | S2 sep 为主 | 连续权重替代 hard 选择，减少"选错"的代价 |

对 S1：最终方案 = L1 + L2（+ 可选 L3）  
对 S2 sep：最终方案 = L3（+ 可选微调 margin）

---

## 3. 方案详细设计

### 3.1 方案 L1: Per-View Adaptive Fusion

#### 动机

数据证据：
- S1 pixel fusion: first frame 从 25.24 掉到 24.07（-1.17），因为 G_plus conf (0.752) > G_base conf (0.743)，大量 first pixels 被错误替换
- Oracle: first frame 可达 26.24，但几乎全选 base——说明 first frame 就不该做 conf fusion
- Center/Last: Oracle 确实能从 fusion 获益（+0.49/+0.52），说明这里值得做 fusion

**结论**：对 first frame 强制或近乎强制用 base，对 center/last 用校准后的 conf fusion。

#### 实现位置

当前代码结构（`stereosplat.py` line 10417-10443）：

```python
# 现状：对所有 V 个视角一起 fusion
elif pixel_level_conf_fusion:
    rendered_color_fuse, rendered_depth_fuse, rendered_conf_fuse = fuse_renders_by_conf_pixelwise(
        rendered_color_2view_final,       # [B, V, 3, H, W] — 全部视角
        rendered_color_pseudo_multiview,   # [B, V, 3, H, W] — 全部视角
        rendered_depth_2view_final,
        rendered_depth_pseudo_multiview,
        rendered_conf_2view_final,
        rendered_conf_pseudo_multiview,
        conf_fusion_margin=conf_fusion_margin,  # 一个标量，所有视角共享
    )
```

fusion 后的 tensor 再经过 `interleave_left_right` 变成 interleaved stereo format，然后：
- `[:, -2:]` = first stereo
- `[:, -4:-2]` = last stereo
- `[:, -6:-4]` = center stereo

**但 fusion 发生在 interleave 之前**，此时 V 维度的布局是 output views（6 个视角：center_L, center_R, last_L, last_R, first_L, first_R）。

需要确认：在 fusion 前，V 维度的排列是 `[center_L, center_R, last_L, last_R, first_L, first_R]`——对应 interleave 后 `[-6:-4]`=center, `[-4:-2]`=last, `[-2:]`=first。

#### 具体代码改动

**新增函数**（在 `fuse_renders_by_conf_pixelwise` 附近，line ~422 之后）：

```python
def fuse_renders_per_view_adaptive(
    rgb_a,          # G_base renders [B, V, 3, H, W]
    rgb_b,          # G_plus renders [B, V, 3, H, W]
    depth_a,        # [B, V, H, W]
    depth_b,        # [B, V, H, W]
    conf_a,         # [B, V, H, W]
    conf_b,         # [B, V, H, W]
    first_margin=999.0,    # 极大 → first frame 永远选 base
    center_margin=0.0,     # center frame 的 margin
    last_margin=0.0,       # last frame 的 margin
    calibration="none",    # "none" | "zscore" | "minmax"
    temperature=None,      # None=hard selection, float=soft blending
):
    """
    Per-view adaptive fusion.
    
    V 维度布局（fusion 前）: [center_L, center_R, last_L, last_R, first_L, first_R]
    即 V indices: 0,1=center; 2,3=last; 4,5=first
    """
    B, V, H, W = conf_a.shape
    
    # Step 1: 构建 per-view margin tensor
    margins = torch.zeros(1, V, 1, 1, device=conf_a.device)
    margins[0, 0:2, 0, 0] = center_margin   # center stereo
    margins[0, 2:4, 0, 0] = last_margin     # last stereo
    margins[0, 4:6, 0, 0] = first_margin    # first stereo (极大=强制 base)
    
    # Step 2: 可选 conf 校准
    if calibration == "zscore":
        # per-image z-score（每张图独立标准化）
        conf_a_cal = (conf_a - conf_a.mean(dim=(-2,-1), keepdim=True)) / \
                     (conf_a.std(dim=(-2,-1), keepdim=True) + 1e-8)
        conf_b_cal = (conf_b - conf_b.mean(dim=(-2,-1), keepdim=True)) / \
                     (conf_b.std(dim=(-2,-1), keepdim=True) + 1e-8)
    elif calibration == "minmax":
        # per-image min-max normalization
        a_min = conf_a.amin(dim=(-2,-1), keepdim=True)
        a_max = conf_a.amax(dim=(-2,-1), keepdim=True)
        b_min = conf_b.amin(dim=(-2,-1), keepdim=True)
        b_max = conf_b.amax(dim=(-2,-1), keepdim=True)
        conf_a_cal = (conf_a - a_min) / (a_max - a_min + 1e-8)
        conf_b_cal = (conf_b - b_min) / (b_max - b_min + 1e-8)
    else:
        conf_a_cal = conf_a
        conf_b_cal = conf_b
    
    # Step 3: Fusion decision
    if temperature is None:
        # Hard selection
        pick_b = conf_b_cal > conf_a_cal + margins
        pick_b_rgb = pick_b.unsqueeze(2)
        fused_rgb = torch.where(pick_b_rgb, rgb_b, rgb_a)
        fused_depth = torch.where(pick_b, depth_b, depth_a)
        fused_conf = torch.where(pick_b, conf_b, conf_a)
    else:
        # Soft blending
        conf_diff = conf_b_cal - conf_a_cal - margins  # [B, V, H, W]
        weight_b = torch.sigmoid(conf_diff * temperature)
        weight_a = 1.0 - weight_b
        fused_rgb = weight_a.unsqueeze(2) * rgb_a + weight_b.unsqueeze(2) * rgb_b
        fused_depth = weight_a * depth_a + weight_b * depth_b
        fused_conf = weight_a * conf_a + weight_b * conf_b
    
    return fused_rgb, fused_depth, fused_conf
```

**修改调用处**（line ~10417）：

```python
elif pixel_level_conf_fusion:
    render_pkg_2view_final = self.renderer.render(...)
    rendered_color_2view_final = ...
    rendered_depth_2view_final = ...
    rendered_conf_2view_final = ...
    
    if fusion_mode == "per_view_adaptive":
        rendered_color_fuse, rendered_depth_fuse, rendered_conf_fuse = fuse_renders_per_view_adaptive(
            rendered_color_2view_final,
            rendered_color_pseudo_multiview,
            rendered_depth_2view_final,
            rendered_depth_pseudo_multiview,
            rendered_conf_2view_final,
            rendered_conf_pseudo_multiview,
            first_margin=fusion_first_margin,
            center_margin=fusion_center_margin,
            last_margin=fusion_last_margin,
            calibration=fusion_calibration,
            temperature=fusion_temperature,
        )
    else:
        # legacy path，不影响现有实验
        rendered_color_fuse, rendered_depth_fuse, rendered_conf_fuse = fuse_renders_by_conf_pixelwise(
            ...,
            conf_fusion_margin=conf_fusion_margin,
        )
```

#### 为什么 per-image z-score 而不是全局

全局 z-score（所有视角一起算 mean/std）仍然无法消除 G_base vs G_plus 之间的系统偏差。**Per-image** 意味着每张渲染图单独标准化——这样 G_base 的 "相对高 conf 区域" 和 G_plus 的 "相对高 conf 区域" 才能公平比较。

直觉：如果 G_base 某区域的 conf 在 G_base 自己内部排名 top 10%，G_plus 同区域也在自己内部 top 10%，那它们是平局——选 base（因为 base 平均更好）。只有 G_plus 在自己内部明显高于 G_base 在自己内部时，才选 plus。

---

### 3.2 方案 L2: Conf Calibration（嵌入在 L1 内）

已经在上面的 `fuse_renders_per_view_adaptive` 中实现，作为 `calibration` 参数。

**三种校准方式及原理**：

| 方式 | 做什么 | 为什么 |
|------|--------|--------|
| `none` | 不校准，原始 conf 比较 | 对照组 |
| `zscore` | 每张图 (conf - mean) / std | 消除 G_plus 系统性高均值(0.744 vs 0.660)的影响，让比较基于相对排名 |
| `minmax` | 每张图 (conf - min) / (max - min) | 把两路 conf 都映射到 [0,1]，消除绝对值差异 |

**预期**：`zscore` 应该比 `minmax` 好，因为 zscore 保留了分布形状信息，而 minmax 容易被极值扭曲。

---

### 3.3 方案 L3: Soft Blending（嵌入在 L1 内）

已经在上面的 `fuse_renders_per_view_adaptive` 中实现，作为 `temperature` 参数。

**原理**：

当前 hard selection 的问题：
```
pixel A: conf_base=0.70, conf_plus=0.71 → 选 plus
pixel B: conf_base=0.70, conf_plus=0.69 → 选 base
```
A 和 B 的 conf 差异只有 0.01，但结果完全相反。这在空间上会造成噪点/不连续。

Soft blending：
```
weight_plus = sigmoid((conf_plus - conf_base - margin) × temperature)
fused = weight_plus × rgb_plus + (1 - weight_plus) × rgb_base
```

- temperature=∞ → 退化为 hard selection
- temperature=5~20 → conf 差小时 soft blend，差大时接近 hard
- 好处：在 conf 不确定的区域取平均，减少错误选择的损失

---

## 4. CLI 参数设计

### 4.1 新增参数（`eval/run.py`）

```python
# Per-view adaptive fusion
parser.add_argument('--fusion_mode', type=str, default='legacy',
                    choices=['legacy', 'per_view_adaptive'],
                    help='legacy=原有逻辑; per_view_adaptive=新方案')
parser.add_argument('--fusion_first_margin', type=float, default=999.0,
                    help='first frame margin, 999=强制base')
parser.add_argument('--fusion_center_margin', type=float, default=0.0,
                    help='center frame margin')
parser.add_argument('--fusion_last_margin', type=float, default=0.0,
                    help='last frame margin')
parser.add_argument('--fusion_calibration', type=str, default='none',
                    choices=['none', 'zscore', 'minmax'],
                    help='conf 校准方式')
parser.add_argument('--fusion_temperature', type=float, default=None,
                    help='None=hard; float=soft blending temperature')
```

### 4.2 为什么这样设计 CLI

1. `--fusion_mode legacy` 保证所有现有实验结果完全不受影响（向后兼容）
2. 每个参数独立可调，方便 grid search
3. 参数命名直观，看参数就知道在干什么

---

## 5. 需要改动的文件及位置

| # | 文件 | 位置 | 改什么 | 原因 |
|---|------|------|--------|------|
| 1 | `stereosplat.py` | line ~422 (在 `fuse_renders_by_conf_pixelwise` 后) | 新增 `fuse_renders_per_view_adaptive` 函数 (~60行) | 核心融合逻辑 |
| 2 | `stereosplat.py` | line ~10417 (`infer_pixel_fusion_pose_injection_single_model` 内) | 加 `if fusion_mode == "per_view_adaptive"` 分支 (~10行) | S1 whole 的调用处 |
| 3 | `stereosplat.py` | line ~9580 (`infer_pixel_fusion_pose_injection_frozen_stage1_two_models` 内) | 同上 (~10行) | S2 separated 的调用处 |
| 4 | `stereosplat.py` | line ~9940 (函数签名) | 加新参数到签名 | 透传 |
| 5 | `eval/run.py` | argparse 部分 | 新增 6 个 CLI 参数 (~15行) | 用户接口 |
| 6 | `eval/routes.py` | `run_batch_inference` 调用处 | 透传新参数 (~5行) | 连接层 |

**总改动量：~100 行新增代码，~20 行修改**

---

## 6. 需要确认的前置问题

### 6.1 Fusion 前 V 维度的布局

**问题**：在 `fuse_renders_by_conf_pixelwise` 被调用时，`rendered_color_2view_final` 的 V 维度排列是什么？

**证据链**：
- `output_batch_dict["output_c2ws"]` 的布局由 `prepare_tripleview_by_ratio_index` 决定
- `render_pkg_fuse` 是用 `output_c2ws` 渲染的
- 后面 `interleave_left_right` 后，`[:, -2:]`=first, `[:, -4:-2]`=last, `[:, -6:-4]`=center

**关键**：`interleave_left_right` 的输入格式是 `[B, V, C, H, W]`，它把 (left_i, right_i) 对 interleave。输出的排列中 first 在最后 (`-2:`)，说明输入中 first pair 也在最后的位置。

**推断**：fusion 前 V 维度 = `[center_L, center_R, last_L, last_R, first_L, first_R]`（6 views），即：
- V indices 0,1 = center stereo
- V indices 2,3 = last stereo  
- V indices 4,5 = first stereo

**需要代码验证**：在 Day 1 实现前，先 print 一下 shape 和比对确认。如果搞错了 view 索引，效果会完全反转。

### 6.2 S2 separated 的代码路径

S2 separated 用的是 `infer_pixel_fusion_pose_injection_frozen_stage1_two_models`，其内部 fusion 逻辑和 single_model 版本类似，但有两个模型。需要确认：
- 它的 V 维度布局是否一样
- fusion 调用处的代码结构是否相同

**处理方式**：先在 single_model 上实现验证，确认有效后再迁移到 frozen_stage1 版本。

---

## 7. 消融实验矩阵

### 7.1 Stage1 whole 消融（demo 9 bins 快速筛选）

| 实验 ID | fusion_mode | first_margin | center_margin | last_margin | calibration | temperature | 预期 |
|---------|-------------|--------------|---------------|-------------|-------------|-------------|------|
| S1-0 | legacy | — | — | — | — | — | 19.93 (已有) |
| S1-1 | per_view_adaptive | 999 | 0.0 | 0.0 | none | None | ~20.4 (first recovery) |
| S1-2 | per_view_adaptive | 999 | 0.0 | 0.0 | zscore | None | ~20.5 |
| S1-3 | per_view_adaptive | 999 | 0.0 | 0.0 | minmax | None | ~20.4 |
| S1-4 | per_view_adaptive | 999 | 0.05 | 0.0 | zscore | None | ~20.5 |
| S1-5 | per_view_adaptive | 999 | 0.05 | -0.05 | zscore | None | ~20.5 |
| S1-6 | per_view_adaptive | 999 | 0.0 | 0.0 | zscore | 10.0 | ~20.5 |
| S1-7 | per_view_adaptive | 999 | 0.0 | 0.0 | zscore | 5.0 | ~20.5 |
| S1-8 | per_view_adaptive | 0.1 | 0.0 | 0.0 | zscore | None | ~20.4 (允许少量 first fusion) |

**消融逻辑**：
1. S1-1 vs S1-0：验证 "first frame 强制 base" 的核心收益
2. S1-2 vs S1-1：加 zscore 后 center/last 是否改善
3. S1-3 vs S1-2：zscore vs minmax 哪个好
4. S1-4, S1-5：对 center/last 微调 margin
5. S1-6, S1-7：soft blending 是否进一步帮助
6. S1-8：first frame 用小 margin（而非完全禁止）是否更好

### 7.2 Stage2 separated 消融

S2 sep 的策略与 S1 不同——重点是 soft blending + 精细 margin，而非偏差校正。

**原因**：
- S2 sep conf 偏差仅 +0.008，zscore 校准基本没意义
- S2 sep pixel fusion 已经三视角都 > baseline，不存在"选错了"的系统问题
- 需要解决的是：在 conf 差距小时避免 hard 选错（→ soft blend），以及 last view progressive (17.79) > fusion (17.72) 的退化（→ last margin 调负让更多 plus 进来）

| 实验 ID | fusion_mode | first_margin | center_margin | last_margin | calibration | temperature | 动机 |
|---------|-------------|--------------|---------------|-------------|-------------|-------------|------|
| S2-0 | legacy | — | — | — | — | — | 20.53 (已有) |
| S2-1 | per_view_adaptive | 0.0 | 0.0 | 0.0 | none | 5.0 | 纯 soft blend，减少选错代价 |
| S2-2 | per_view_adaptive | 0.0 | 0.0 | 0.0 | none | 10.0 | 更 hard 的 soft blend |
| S2-3 | per_view_adaptive | 0.0 | 0.0 | -0.02 | none | 5.0 | last 稍偏 plus（因为 prog > fusion 在 last） |
| S2-4 | per_view_adaptive | 0.02 | 0.0 | -0.02 | none | 5.0 | first 微偏 base，last 微偏 plus |
| S2-5 | per_view_adaptive | 0.0 | 0.0 | 0.0 | zscore | 5.0 | 验证 zscore 对 S2 是否有害 |

**关键对比逻辑**：
- S2-1 vs S2-0：soft blend 本身是否有收益（预期 +0.05~0.1，通过减少边界错误选择）
- S2-3 vs S2-1：last margin=-0.02 是否回收 last view 的退化（17.72→接近17.79）
- S2-5 vs S2-1：验证 zscore 对低偏差设置是否中性/有害（预期无显著差异）

---

## 8. 每日计划

### Week 1：方案验证（Stage1 快速迭代 + S2 oracle）

| 天 | 具体任务 | 产出 | 判断标准 |
|----|----------|------|----------|
| **D1 上午** | 1) 确认 V 维度布局（print shape）; 2) 实现 `fuse_renders_per_view_adaptive`; 3) 加 CLI 参数 | 代码 ready | 能 import 不报错 |
| **D1 下午** | 4) 跑 S1-0 (legacy) 确认 9 bins 结果和已有一致（sanity check）; 5) 跑 S1-1（first=999, 无校准）; 6) 同时启动 S2 oracle 完整 val | S1-1 的 first/center/last PSNR; S2 oracle 开始跑 | S1-1 的 first_psnr > 25.0 |
| **D2** | 7) 跑 S1-2, S1-3 (zscore/minmax); 8) 分析 S1-1 的 per-view 结果，确认 first recovery | 最佳 calibration | S1-2 all_view > S1-1 |
| **D3** | 9) 跑 S1-4, S1-5 (margin 调参); 10) S2 oracle 结果出来，分析天花板 | margin 最佳值; S2 上界 | 确认 S2 方向 |
| **D4** | 11) 跑 S1-6, S1-7 (soft blending); 12) 确定 S1 最优配置，跑完整 val (5485 bins) | S1 完整指标 | all_view > 20.4 |
| **D5** | 13) 把 S1 最优迁移到 S2 sep（改 `infer_pixel_fusion_pose_injection_frozen_stage1_two_models`）; 14) 跑 S2-1 demo | S2 趋势确认 | S2-1 > 20.53 |

### Week 2：精调 + 出数字

| 天 | 具体任务 | 产出 |
|----|----------|------|
| **D6** | 15) S2 margin 调参（S2 的 first 不需要强制 base，调小）; 16) 跑 S2-2, S2-3 demo | S2 最优配置 |
| **D7** | 17) S2 最优配置跑完整 val (5485 bins) | S2 正式数字 |
| **D8** | 18) 如果还有空间：尝试更细的 margin grid search（步长 0.01）on demo | 精调 |
| **D9** | 19) 整理所有消融数据写表; 20) 跑 demo_more (27 bins) 出可视化 | 汇报材料 |
| **D10** | 21) 更新 report; 22) 整理代码确保向后兼容 | 完成 |

---

## 9. 风险评估与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| V 维度布局搞错 | 中 | 结果反转 | D1 第一件事就 print 确认 |
| zscore 校准后 conf 仍无区分度（center/last 不涨） | 中 | S1 只回收 first，center/last 不涨 | 仍有 first recovery 保底；尝试 minmax / spatial block |
| S2 separated oracle 只有 ~20.7 | 低-中 | S2 剩余空间仅 0.17 | 调低 S2 预期，focus on S1 出数字 |
| Soft blending 引入整体模糊，SSIM 下降 | 低 | 换 PSNR 掉 SSIM | 用高 temperature (20+) 或放弃 soft blend |
| S2 sep 路径的代码结构不同，迁移困难 | 低 | 多花半天 | 先读代码再改 |
| 完整 val 跑太慢（>24h） | 高 | 延迟 | 只跑 top-2 配置；去掉 CUDA_LAUNCH_BLOCKING=1 加速 |

---

## 10. 成功标准

| 级别 | S1 whole | S2 separated | 整体判断 |
|------|----------|--------------|----------|
| **最低可接受** | > 20.44 (超过 baseline) | > 20.53 (超过当前最佳) | 证明方向对 |
| **正常预期** | ~20.5-20.6 | ~20.6-20.65 | 有明确提升 |
| **乐观** | ~20.7 | ~20.7-20.8 | 接近 Oracle 的一半 |

**如果 D2 结束时 S1-2 的 all_view_psnr < 20.3**（即 first recovery + calibration 加起来都不够），则需要重新审视方向——可能 conf 的 per-pixel 信息本身就没用，需要转向其他信号（如 rendering error propagation 或 learned fusion network）。

---

## 11. 预期论文可呈现的 ablation 表

最终报告中的核心表格：

| Method | First PSNR | Center PSNR | Last PSNR | All PSNR | SSIM |
|--------|-----------|-------------|-----------|----------|------|
| Baseline (2-view) | 25.24 | 19.73 | 17.59 | 20.44 | 0.674 |
| StereoSplat+ (no fusion) | 23.61 | 19.30 | 17.27 | 19.76 | 0.656 |
| + Pixel fusion (legacy) | 24.07 | 19.38 | 17.31 | 19.93 | 0.660 |
| + Per-view adaptive | ? | ? | ? | ? | ? |
| + Per-view + zscore | ? | ? | ? | ? | ? |
| + Per-view + zscore + soft | ? | ? | ? | ? | ? |
| Oracle (GT selection) | 26.24 | 20.22 | 18.11 | 21.10 | 0.706 |

这张表能清晰展示每个改进的贡献。
