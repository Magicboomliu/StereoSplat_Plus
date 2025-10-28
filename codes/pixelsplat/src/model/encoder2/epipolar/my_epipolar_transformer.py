from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .geometry.epipolar_lines import get_depth
from .encodings.positional_encoding import PositionalEncoding
from .transformer.transformer import Transformer
from .conversions import depth_to_relative_disparity
from .epipolar_sampler import EpipolarSampler, EpipolarSampling
from .image_self_attention import ImageSelfAttention, ImageSelfAttentionCfg


class EpipolarTransformer(nn.Module):
    def __init__(self,
                 d_in=128,
                 num_context_views=2,
                 num_samples=32,
                 num_octaves=10
                 ):
        super().__init__()
        
        self.num_context_views = num_context_views        
        self.epipolar_sampler = EpipolarSampler(num_context_views, num_samples)

        if num_octaves > 0:
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(num_octaves)),
                nn.Linear(pe.d_out(1), d_in),
            )

        self_attention_cfg = ImageSelfAttentionCfg(
            patch_size=4,
            num_octaves=10,
            num_layers=2,
            num_heads=4,
            d_token=128,
            d_dot=128,
            d_mlp=256,
        )

        num_layers = 2
        num_heads = 4
        d_dot = 128
        d_mlp = 256
        downscale = 4
        
        feed_forward_layer = partial(ImageSelfAttentionWrapper, self_attention_cfg)

        self.transformer = Transformer(
            d_in,
            num_layers,
            num_heads,
            d_dot,
            d_mlp,
            selfatt=False,
            kv_dim=d_in,
            feed_forward_layer=feed_forward_layer,
        )
        
        self.downscale = downscale
        
        # perform the downscaling here
        if downscale:
            self.downscaler = nn.Conv2d(d_in, d_in, downscale, downscale)
            self.upscaler = nn.ConvTranspose2d(d_in, d_in, downscale, downscale)
            self.upscale_refinement = nn.Sequential(
                nn.Conv2d(d_in, d_in * 2, 7, 1, 3),
                nn.GELU(),
                nn.Conv2d(d_in * 2, d_in, 7, 1, 3),
            )

        if num_context_views > 2:
            self.view_embeddings = nn.Embedding(num_context_views, d_in)
        
        
    def forward(self,
        features,
        extrinsics,
        intrinsics,
        near,
        far):
        
        b, v, _, h, w = features.shape
        
        # If needed, apply downscaling.
        if self.downscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.downscaler(features)
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        # Get the samples used for epipolar attention.
        sampling = self.epipolar_sampler.forward(
            features, extrinsics, intrinsics, near, far
        )

        # Compute positionally encoded depths for the features.
        collect = self.epipolar_sampler.collect
        depths = get_depth(
            rearrange(sampling.origins, "b v r xyz -> b v () r () xyz"),
            rearrange(sampling.directions, "b v r xyz -> b v () r () xyz"),
            sampling.xy_sample,
            rearrange(collect(extrinsics), "b v ov i j -> b v ov () () i j"),
            rearrange(collect(intrinsics), "b v ov i j -> b v ov () () i j"),
        )

        # Clip the depths. This is necessary for edge cases where the context views
        # are extremely close together (or possibly oriented the same way).
        depths = depths.maximum(near[..., None, None, None])
        depths = depths.minimum(far[..., None, None, None])
        depths = depth_to_relative_disparity(
            depths,
            rearrange(near, "b v -> b v () () ()"),
            rearrange(far, "b v -> b v () () ()"),
        )
        depths = self.depth_encoding(depths[..., None])
        kv = sampling.features + depths
        
        
        # Add randomly permuted per-view embeddings to the other views.
        if v > 2:
            shuffle = torch.randperm(v - 1, device=kv.device)
            view_embeddings = rearrange(
                self.view_embeddings(shuffle), "ov c -> () () ov () () c"
            )
            kv = kv + view_embeddings

        # Run the transformer.
        q = rearrange(features, "b v c h w -> (b v h w) () c")
        features = self.transformer.forward(
            q,
            rearrange(kv, "b v ov r s c -> (b v r) (s ov) c"),
            b=b,
            v=v,
            h=h // self.downscale,
            w=w // self.downscale,
        )
        features = rearrange(
            features,
            "(b v h w) () c -> b v c h w",
            b=b,
            v=v,
            h=h // self.downscale,
            w=w // self.downscale,
        )

        # If needed, apply upscaling.
        if self.upscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.upscaler(features)
            features = self.upscale_refinement(features) + features
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        return features, sampling
        

    
    

class ImageSelfAttentionWrapper(nn.Module):
    def __init__(
        self,
        self_attention_cfg: ImageSelfAttentionCfg,
        d_in: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = ImageSelfAttention(self_attention_cfg, d_in, d_in)

    def forward(
        self,
        x: Float[Tensor, "batch token dim"],
        b: int,
        v: int,
        h: int,
        w: int,
    ) -> Float[Tensor, "batch token dim"]:
        x = rearrange(x, "(b v h w) () c -> (b v) c h w", b=b, v=v, h=h, w=w)
        x = self.self_attention(x) + x
        return rearrange(x, "(b v) c h w -> (b v h w) () c", b=b, v=v, h=h, w=w)