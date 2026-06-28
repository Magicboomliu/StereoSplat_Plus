#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Collect extra include paths: pixi puts cub/cuda-std/nv headers under
# targets/x86_64-linux/include/ and targets/.../include/cccl/
extra_include_dirs = [
    os.path.join(ROOT, "third_party/glm/"),
]

# Auto-detect targets include dir used by conda/pixi CUDA packages
conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
targets_include = os.path.join(conda_prefix, "targets", "x86_64-linux", "include")
if os.path.isdir(targets_include):
    extra_include_dirs.append(targets_include)
    cccl_include = os.path.join(targets_include, "cccl")
    if os.path.isdir(cccl_include):
        extra_include_dirs.append(cccl_include)

nvcc_flags = ["-I" + p for p in extra_include_dirs]

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": nvcc_flags},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
