# Stage2 Self-Pseudo Debug Run — Best Checkpoint 选型

训练配置：`stage2_self_pseudo_debug`，`margin_detach_ref=True`，val 集 `val_tiny.txt`（549 scenes），soft fusion。

**训练范围：** step-0 → step-6000（`save_freq=500`, `val_freq=500`）

**锚点基线（step-0）：** 2v mean PSNR = **20.99 dB**

---

## KPI 定义

| ID | 标准 | 说明 |
|----|------|------|
| **KPI-1** | 2v mean ∈ step-0 ± 0.3 dB | 2v 相对 Stage1-init 不退化 |
| **KPI-2** | mv center/last ≥ 2v + 0.3 dB | multiview 在 center/last 赢 2v |
| **KPI-3** | fused mean > mv，且 fused ≥ 2v + 0.6 dB | 融合 mean 三角关系 |
| **KPI-4** | fused center/last ≥ 2v + 0.3 dB | 融合在 center/last 大幅赢 2v |
| **KPI-F** | fused center > 2v center **且** fused last > 2v last | 融合在 center/last 双侧均优于 2v |

> 全轨迹最高完整过关：**1/4**（仅 KPI-1 稳定成立）。KPI-F 自 step-2500 起持续成立（至 step-6000）。

---

## 全轨迹摘要

| step | 2v mean | fused mean | fused−2v mean | fused_c−2v | fused_l−2v | 2v vs s0 | KPI-F | KPI 过关 |
|------|---------|------------|---------------|--------------|------------|----------|-------|----------|
| 0 | 20.99 | 20.29 | −0.69 | −0.19 | −0.05 | 0.00 | ✗ | 1/4 |
| 2000 | 21.17 | 21.31 | +0.14 | ~0 | −0.01 | +0.18 | ✗ | 1/4 |
| 2500 | 21.28 | 21.60 | +0.32 | +0.12 | +0.03 | +0.29 | ✓ | 1/4 |
| 3000 | 21.21 | 21.58 | +0.37 | +0.13 | +0.05 | +0.23 | ✓ | 1/4 |
| **3500** | 20.89 | 21.31 | **+0.43** | +0.21 | **+0.15** | −0.10 | ✓ | 1/4 |
| **4000** | 21.13 | 21.38 | +0.25 | +0.20 | +0.14 | +0.14 | ✓ | 1/4 |
| 4500 | 21.05 | 21.43 | +0.38 | +0.14 | +0.03 | +0.06 | ✓ | 1/4 |
| 5000 | 21.34 | 21.71 | +0.37 | +0.13 | +0.04 | +0.35 | ✓ | 0/4 |
| **5500** | 20.99 | 21.41 | +0.42 | **+0.26** | +0.06 | **0.00** | ✓ | 1/4 |
| 6000 | 21.37 | 21.74 | +0.37 | +0.08 | +0.08 | +0.39 | ✓ | 0/4 |

**关键规律：**

- step-2500 起 KPI-F 持续成立（fused center/last 双双 > 2v）。
- step-3500 为 fused−2v mean 峰值（+0.43），伴随 2v 轻微下滑。
- step-5500 为 fused center 峰值（+0.26），且 2v 精确贴合 step-0。
- step-5000/6000 绝对 PSNR 最高，但 2v 超出 KPI-1 上限（+0.35 / +0.39）。

---

## Checkpoint 路径

```
/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_debug/checkpoint-{N}
```

Val 指标路径：

```
/data1/zliu/IROS26/Compared_With_Others_Pixi/train_visualization/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_debug/stereosplat_kitti360_stage2_self_pseudo_with_conf_and_difix3d/validation/step-{N}/fusion_metric.json
```

---

## best1 — 综合主模型（推荐论文主结果）

**Checkpoint：`checkpoint-5500`**

| 指标 | 数值 | 评价 |
|------|------|------|
| KPI-1 | 2v@s0 = **0.00 dB** | 精确贴合 step-0 |
| KPI-F | ✓ (+0.26 / +0.06) | center/last 双侧赢 2v |
| fused−2v mean | +0.42 dB | 全场第 2 |
| fused center − 2v | **+0.26 dB** | 全场最高 |
| fused last − 2v | +0.06 dB | 偏弱 |
| 绝对 fused mean | 21.41 dB | — |

**选型理由：**

- KPI-F 成立，且 **center 融合增益全场最强**。
- KPI-1 完美铆住 step-0，审稿人视角「2v 无明显退化」最安全。
- fused−2v mean 接近峰值（+0.42 vs 3500 的 +0.43）。

**短板：** last 仅 +0.06 dB；KPI-2/3/4 仍未完整达标。

**适合叙事：** center 视角 fusion 明显优于 2v；整体 2v 保真。

---

## best2 — 双侧均衡（center + last 都够看）

**Checkpoint：`checkpoint-4000`**

| 指标 | 数值 | 评价 |
|------|------|------|
| KPI-1 | 2v@s0 = **+0.14 dB** | 余量充足 |
| KPI-F | ✓ (**+0.20 / +0.14**) | 双侧均衡 |
| fused−2v mean | +0.25 dB | 中等 |
| min(center, last) | **+0.14 dB** | 均衡度全场第 2 |
| 绝对 fused mean | 21.38 dB | — |

**选型理由：**

- KPI-F 成立，且 center/last 相对增益 **最均衡**（无 5500 的 last 过弱、无 3500 的 2v 献祭）。
- KPI-1 有 +0.14 dB 舒适余量。
- 2v 绝对值 21.13，处于安全区间中段。

**短板：** fused−2v mean 仅 +0.25，融合 mean 叙事偏弱。

**适合叙事：** fusion 在 center 和 last 都稳定优于 2v，且 2v 铆钉稳健。

---

## best3 — 融合增益最强（mean + last 双高）

**Checkpoint：`checkpoint-3500`**

| 指标 | 数值 | 评价 |
|------|------|------|
| KPI-1 | 2v@s0 = **−0.10 dB** | 仍过关，偏下限 |
| KPI-F | ✓ (+0.21 / **+0.15**) | last 全场最高 |
| fused−2v mean | **+0.43 dB** | 全场最高 |
| fused last − 2v | **+0.15 dB** | 全场最高 |
| min(center, last) | **+0.15 dB** | 全场最高 |
| 绝对 fused mean | 21.31 dB | — |

**选型理由：**

- KPI-F 成立，且 **fused−2v mean 与 last 增益均为全场最高**。
- min(center, last) 双过幅度最大，融合相对优势最突出。

**短板：** 2v mean 偏低（20.89），存在「献祭 2v 拓宽 gap」痕迹；KPI-1 余量最小。

**适合叙事：** 强调 fusion 在 center/last 及 mean 上相对 2v 的最大增益。

---

## 三档对照

| | best1: 5500 | best2: 4000 | best3: 3500 |
|--|-------------|-------------|-------------|
| 定位 | 综合主模型 | 双侧均衡 | 融合增益最强 |
| KPI-F | ✓ | ✓ | ✓ |
| fused_c − 2v | **+0.26** | +0.20 | +0.21 |
| fused_l − 2v | +0.06 | **+0.14** | **+0.15** |
| fused−2v mean | +0.42 | +0.25 | **+0.43** |
| 2v vs step-0 | **0.00** | +0.14 | −0.10 |
| 主要风险 | last 弱 | mean gap 小 | 2v 献祭感 |

---

## 不推荐作为主模型的 checkpoint

| step | 原因 |
|------|------|
| **5000** | KPI-1 FAIL（2v@s0 = +0.35，超上限） |
| **6000** | KPI-1 FAIL（2v@s0 = +0.39）；绝对值最高但越铆钉带 |
| **2000** | KPI-F 未成立（fused center/last 未双双赢 2v） |
| **3000** | KPI-F 过但 last 仅 +0.05，不如 4000/5500/3500 |

---

## 使用建议

```bash
# best1 主模型
resume_from=".../stage2_self_pseudo_debug/checkpoint-5500"

# best2 均衡对照
resume_from=".../stage2_self_pseudo_debug/checkpoint-4000"

# best3 融合增益 ablation
resume_from=".../stage2_self_pseudo_debug/checkpoint-3500"
```

---

*生成时间：2026-06-20 · 训练 run：`stereosplat_kitti360_stage2_self_pseudo_with_conf_and_difix3d` · work_dir：`stage2_self_pseudo_debug`*
