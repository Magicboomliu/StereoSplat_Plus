# Differential Gaussian Rasterization (with confidence)

This package is a fork of the [original repository](https://github.com/graphdeco-inria/diff-gaussian-rasterization) with the following extensions:

- **Depth rendering**: forward and backward passes
- **Opacity accumulation (alpha) rendering**: forward and backward passes
- **Confidence (conf) rendering**: an optional learnable scalar confidence per 3D Gaussian, rasterized into a per-view confidence map for multi-view consistency

## API

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
    confs=confs,          # optional, shape [P], per-Gaussian confidence
)
```

`confs` is optional. If omitted, `rendered_conf` is an all-zero map and existing call sites behave as before.

## How confidence is rasterized

`conf` uses the same alpha-weighted splatting as `depth`:

```
rendered_conf[pixel] = Σ conf_i * α_i * T_i
```

Here `α_i` is the blending weight of Gaussian `i` at the pixel and `T_i` is transmittance. The operation is differentiable w.r.t. `confs`, so confidence can be optimized with gradient descent.

## Typical usage

Assign a learnable confidence to each 3D Gaussian, render a confidence map per training view, and aggregate across views to estimate reliability:

```python
import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

# Learnable confs; sigmoid is a common way to keep values in (0, 1)
raw_confs = torch.nn.Parameter(torch.zeros(num_gaussians, device="cuda"))
confs = torch.sigmoid(raw_confs)  # shape: [P]

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
# Average conf_map over views and map back to Gaussians for a multi-view confidence estimate
```

## Installation

This subproject can be managed with [pixi](https://pixi.sh) (CUDA driver >= 12.0 recommended):

```bash
# 1. Create the environment (first run downloads PyTorch + CUDA toolchain, ~2GB)
pixi install

# 2. Build and install
pixi run build

# 3. Run tests
pixi run test
```

Without pixi, install in an existing PyTorch environment:

```bash
pip install -e . --no-build-isolation
```

> **Note:** Building requires a CUDA toolkit (nvcc) compatible with your PyTorch build.

## Citation

This code extends the official implementation of:

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

In the parent **StereoSplat+ conf** repo, Gaussians are **15D** (including `conf`) and training uses `rendered_conf`. Build via the parent project: `pixi run -e cu118 setup`. See **[../README.md](../README.md)** for training, evaluation, and pixel fusion.
