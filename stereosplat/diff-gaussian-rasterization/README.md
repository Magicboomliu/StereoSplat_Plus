# Differential Gaussian Rasterization

本库是 [原始仓库](https://github.com/graphdeco-inria/diff-gaussian-rasterization) 的修改版，在原有基础上增加了以下功能：

- **深度图（depth）渲染**：前向与反向传播均支持
- **不透明度累积图（alpha）渲染**：前向与反向传播均支持
- **置信度图（conf）渲染**：为每个 3D Gaussian 添加可学习的标量置信度属性，可渲染为每个视角的 confidence map，用于多视角一致性分析

## 接口说明

```python
rendered_image, radii, rendered_depth, rendered_alpha, rendered_conf = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=colors_precomp,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=cov3D_precomp,
    confs=confs,          # 新增，可选，shape: [P]，每个 Gaussian 的置信度
)
```

`confs` 参数为可选项，不传时 `rendered_conf` 输出全零图，不影响已有代码。

## 置信度渲染原理

`conf` 的渲染方式与 `depth` 完全一致，采用 alpha-weighted splatting：

```
rendered_conf[pixel] = Σ conf_i * α_i * T_i
```

其中 `α_i` 为第 `i` 个 Gaussian 在该像素的混合权重，`T_i` 为透射率。整个过程对 `confs` 可微，支持通过梯度下降优化置信度。

## 典型用法

给每个 3D Gaussian 赋予一个可学习的置信度，渲染出每个训练视角的 confidence map，通过跨视角平均来评估当前 Gaussian 的可靠性：

```python
import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

# confs 为可学习参数，建议用 sigmoid 约束到 (0, 1)
raw_confs = torch.nn.Parameter(torch.zeros(num_gaussians, device="cuda"))
confs = torch.sigmoid(raw_confs)  # shape: [P]

# 渲染
color, radii, depth, alpha, conf_map = rasterizer(
    means3D=means3D,
    means2D=means2D,
    opacities=opacity,
    shs=shs,
    scales=scales,
    rotations=rotations,
    confs=confs,
)
# conf_map: shape [1, H, W]
# 跨多个视角平均 conf_map，投影回每个 Gaussian，可得到该 Gaussian 的多视角置信度估计
```

## 安装

本项目使用 [pixi](https://pixi.sh) 管理环境，需要 CUDA 驱动 >= 12.0。

```bash
# 1. 初始化环境（首次执行，会下载 PyTorch + CUDA 工具链，约 2GB）
pixi install

# 2. 编译安装
pixi run build

# 3. 运行测试
pixi run test
```

若不使用 pixi，也可在已有 PyTorch 环境中手动安装：

```bash
pip install -e . --no-build-isolation
```

> **注意**：编译时需要系统已安装与 PyTorch 版本匹配的 CUDA 工具链（nvcc）。

## 引用

本库基于以下论文的官方实现：

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

---

本仓库 StereoSplat+ 使用 **15D Gaussian**（含 `conf` 维）与 `rendered_conf` 输出；编译入口 `pixi run -e cu118 setup`。训练 / 评估 / pixel_fusion 见 **[../README.md](../README.md)**。
