# StereoSplat Plus (Pixi)

Feed-forward 3D Gaussian Splatting for autonomous driving scenes (KITTI-360).

本仓库为 **带 per-Gaussian confidence 的 StereoSplat+ 专用版**（15D Gaussians + 自定义 rasterizer），训练与评估均围绕 conf 模型设计。

---

## 组件概览

| 组件 | 说明 |
|------|------|
| **StereoSplat（conf）** | Stage1：全 GT view + conf 监督 |
| **StereoSplat Stage2** | Pseudo-GT mix + 可选 Difix3D 修复 |
| **StereoSplat+ 推理** | `stereosplat_plus`：pose injection + `pseudo_ratio` |
| **Pixel-level fusion** | `pixel_fusion`：在 S+ 上按 conf 逐像素融合两路渲染（可选 A1 `--conf_fusion_margin`） |
| **GS voxel fusion** | `--gs_conf_fusion`：3D 体素内融合 G_base 与 G_plus，再单次渲染 |
| **GS + Pixel 联合** | `--gs_conf_fusion` + `--conf_pixel_level_fusion`：先 GS 融合，再 base 渲染 vs GS 融合渲染做 pixel 融合 |
| **Oracle 上界** | `--use_gt_view`：GT 逐像素选 G_base / G_plus（Stage1 消融） |
| **Difix3D** | 参考图引导的 pseudo view 修复（`difix3d/` 独立 Pixi 子项目） |

---

## Quick Start

```bash
cd stereosplat
pixi install -e cu118          # Python 依赖（accelerate, torch, wandb, ...）
pixi run -e cu118 setup        # 编译含 conf 的 diff-gaussian-rasterization（必须）
bash scripts/train/stereosplat/train.sh
```

评估示例（Stage2 基础 2-view）：

```bash
bash scripts/evaluation/stage2/stereosplat_two_gt_views_forward.sh
```

---

## 评估三种 mode（递进）

| `eval_mode` | 含义 | 典型 Shell |
|-------------|------|------------|
| `stereosplat` | 2-view GT → 一次 forward → 渲染评估 | `stage{1,2}/stereosplat_two_gt_views_forward.sh` |
| `stereosplat_plus` | 在 ① 上：全轨迹渲染 → `pseudo_ratio` 选 pseudo → 可选 Difix → reinject → 再 forward | `stage{1,2}/stereosplat_plus_progressive_single_model.sh` |
| `pixel_fusion` | 在 ② 上：两路 render 按 conf 逐像素融合 | `stage{1,2}/pixel_fusion_pose_injection_*.sh` |
| GS 融合（Stage1） | 3D 体素 conf 融合 G_base/G_plus | `stage1/gs_conf_voxel_fusion_pose_injection_single_model.sh` |
| GS+Pixel（Stage1） | 先 GS 融合，再 base vs G_gs_fused 的 pixel 融合 | `stage1/gs_and_pixel_fusion_pose_injection_single_model.sh` |

统一入口：`stereosplat/eval/run.py` → `eval/routes.py` → `stereosplat.py` 中 `infer_*` 函数。

**融合扩展（均在 `pixel_fusion` + `whole` + Stage1 权重上验证）**：

- **Pixel A1**：`--conf_pixel_level_fusion --conf_fusion_margin 0.05`（plus 需明显更高才选，平局偏 base）
- **GS**：`--gs_conf_fusion` + `--gs_fusion_voxel_size` / `--gs_fusion_margin` / `--gs_fusion_conf_agg mean` / `--gs_fusion_base_conf_thresh`
- **联合**：上述 GS 与 pixel 参数同时开启

```bash
cd stereosplat
bash scripts/evaluation/stage1/gs_conf_voxel_fusion_pose_injection_single_model.sh      # 仅 GS
bash scripts/evaluation/stage1/gs_and_pixel_fusion_pose_injection_single_model.sh       # GS + Pixel
bash scripts/evaluation/stage1/oracle_gt_upper_bound_pose_injection.sh                   # Oracle 上界
```

**`pseudo_ratio`**（S+ / pixel_fusion 共用）：`--pseudo_ratio 0.5 1.0` 表示第二组 = center stereo、第三组 = last stereo（默认，与原 progressive 行为等价）。未传时 `eval/run.py` 对 `stereosplat_plus` 与 `pixel_fusion` 自动填 `[0.5, 1.0]`。

**Shell 文件名说明**：部分脚本仍带 `progressive` 历史命名（如 `stereosplat_plus_progressive_single_model.sh`），实际已统一为 **pose injection + `pseudo_ratio`**；模型函数为 `infer_stereosplat_plus_pose_injection_single_model()`。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| **[stereosplat/README.md](stereosplat/README.md)** | 安装、训练、推理对照表（含 GS/联合融合）、`pseudo_ratio`、Shell 速查、可视化 |
| **[stereosplat/eval/README.md](stereosplat/eval/README.md)** | 调用链、CLI 全参数、legacy 映射、FAQ |
| **[docker/README.md](docker/README.md)** | 容器内 Pixi 环境 |
| **[difix3d/README.md](difix3d/README.md)** | Difix3D 独立训练/评估（Pixi） |

---

## 目录结构（简）

```
StereoSplat_Plus/
├── README.md                 # 本文件
├── stereosplat/              # ★ 主工程（训练 + 评估 + 模型）
│   ├── eval/                 # 评估逻辑
│   ├── trainer/
│   ├── scripts/evaluation/stage{1,2}/   # 含 gs_conf / gs_and_pixel 融合脚本（Stage1）
│   └── src/stereosplat/
├── difix3d/                  # Difix3D 独立子项目
└── docker/
```
