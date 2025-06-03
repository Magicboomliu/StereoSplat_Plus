import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import numpy as np


from dataclasses import dataclass
from typing import Literal
import torch

from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
        
@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    depth: Float[Tensor, "batch view height width"] | None



# Decoder Splatting CUDA Decoder
class DecoderSplattingCUDA(nn.Module):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        dataset_cfg
    ) -> None:
        super().__init__()
        
        self.dataset_cfg = dataset_cfg
        
        self.register_buffer(
            "background_color",
            torch.tensor(self.dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )
        
        
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ):  
        
        b, v, _, _ = extrinsics.shape
        
        color, depth, alpha = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        )
        

        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v)
        alpha = rearrange(alpha,"(b v) c h w -> b v c h w", b=b, v=v)


        return {
            
            "color": color,
            "depth": None if depth_mode is None else depth,
            "alpha": None if depth_mode is None else alpha
        }
