import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import json
import os
import time
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import cv2
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from collections import OrderedDict

from encoder2.backbone.backbone_pyramid import BackbonePyramid
from encoder2.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from encoder2.costvolume.depth_predictor_multiview import DepthPredictorMultiViewPyramid

from geometry.projection import sample_image_grid

from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor


from decoder2.my_decoder_splatting_cuda import DecoderSplattingCUDA


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"] | Float[Tensor, "batch view gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"] | Float[Tensor, "batch view gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"] | Float[Tensor, "batch view gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"] | Float[Tensor, "batch view gaussian"]



class HiSplatEncoder(nn.Module):
    def __init__(self,
                 unimatch_weights_path,
                 vit_path
                 ):
        super(HiSplatEncoder, self).__init__()
        
        self.unimatch_weights_path = unimatch_weights_path
        self.vit_path = vit_path

        d_feature = 128
        downscale_factor = 4

        # here is the backbone of feature extraction.
        self.backbone = BackbonePyramid(feature_channels=d_feature,
                downscale_factor=downscale_factor,
                vit_path=vit_path)    

        ckpt_path = self.unimatch_weights_path
        unimatch_pretrained_ckpt_weights = torch.load(ckpt_path)["model"]
        updated_state_dict = {}
        for k, v in unimatch_pretrained_ckpt_weights.items():
            if k in self.backbone.state_dict():
                updated_state_dict[k] = v
            else:
                possible_k = "backbone.encoder." + ".".join(k.split(".")[1:])
                if possible_k in self.backbone.state_dict():
                    updated_state_dict["backbone.encoder." + possible_k] = v
        updated_state_dict = OrderedDict(updated_state_dict)
        self.backbone.load_state_dict(updated_state_dict, strict=False)


        gaussian_adapter_cfg =GaussianAdapterCfg(
            gaussian_scale_min=0.5,
            gaussian_scale_max=15.0,
            sh_degree=4
        )
        self.gaussian_adapter = GaussianAdapter(gaussian_adapter_cfg)


        num_depth_candidates=192
        multiview_trans_attn_split=2
        costvolume_unet_feat_dim=128
        costvolume_unet_channel_mult=[1,1,1]
        costvolume_unet_attn_res=[]
        depth_unet_feat_dim=64
        depth_unet_attn_res=[]
        depth_unet_channel_mult=[1, 1, 1]
        downscale_factor=4
        num_surfaces = 1
        gaussians_per_pixel = 1
        num_context_views = 2
        
        
        self.multiview_trans_attn_split = multiview_trans_attn_split
        self.gaussians_per_pixel = gaussians_per_pixel
        self.num_surfaces = num_surfaces


        self.depth_predictor = DepthPredictorMultiViewPyramid(
                    feature_channels=d_feature,
                    upscale_factor=downscale_factor,
                    num_depth_candidates=num_depth_candidates,
                    costvolume_unet_feat_dim=costvolume_unet_feat_dim,
                    costvolume_unet_channel_mult=tuple(costvolume_unet_channel_mult),
                    costvolume_unet_attn_res=tuple(costvolume_unet_attn_res),
                    gaussian_raw_channels=num_surfaces * (self.gaussian_adapter.d_in + 2),
                    gaussians_per_pixel=gaussians_per_pixel,
                    num_views=num_context_views,
                    depth_unet_feat_dim=depth_unet_feat_dim,
                    depth_unet_attn_res=depth_unet_attn_res,
                    depth_unet_channel_mult=depth_unet_channel_mult,
                )



    def map_pdf_to_opacity(
        self,
        pdf,
        global_step,
    ):
        # https://www.desmos.com/calculator/opvwti3ba9

        initial=0.0
        final=0.0
        warm_up=1
        x = initial + min(global_step / warm_up, 1) * (final - initial)
        exponent = 2**x
        # Map the probability density to an opacity. default is pdf
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))



    def convert_to_gaussians(self, result_dict, context, features_list, global_step, visualization_dump):
        stage_num = len(result_dict)
        gaussian_dict = {k: {} for k in result_dict.keys()}
        device = context["image"].device
        for i in range(stage_num):
            raw_gaussians = result_dict[f"stage{i}"]["raw_gaussians"]
            densities = result_dict[f"stage{i}"]["densities"]
            depths = result_dict[f"stage{i}"]["depths"]
            h, w = features_list[0][i].shape[-2:]
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            offset_xy = gaussians[..., :2].sigmoid()  # [offset: 2, scales: 3, rotation: 4, sh: 3*25 ]
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size  # maximum change 0.5 pixel, normed xy ray
            gpp = self.cfg.gaussians_per_pixel
            gaussians, scales = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),  # 1 2 4096 1 2
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
            )


            # Optionally apply a per-pixel opacity.
            opacity_multiplier = 1
            scales = rearrange(scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
            rotations = rearrange(gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw")
            gaussian_dict[f"stage{i}"]["gaussians"] = Gaussians(
                rearrange(
                    gaussians.means.float(),
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances.float(),
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics.float(),
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    (opacity_multiplier * gaussians.opacities).float(),
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
            gaussian_dict[f"stage{i}"]["depths"] = depths
            gaussian_dict[f"stage{i}"]["scales"] = scales
            gaussian_dict[f"stage{i}"]["rotations"] = rotations
        return gaussian_dict

    def convert_to_gaussians_single_stge(
        self,
        raw_gaussians,
        densities,
        depths,
        image_size,
        extrinsics,
        intrinsics,
        global_step,
        opacity_multiplier=1.0,
        stage_id=0,
    ):
        device = raw_gaussians.device
        h, w = image_size[0], image_size[1]
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()  # [offset: 2, scales: 3, rotation: 4, sh: 3*25 ]
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size  # maximum change 0.5 pixel, normed xy ray
        gpp = self.gaussians_per_pixel
        gaussians, scales = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),  # 1 2 4096 1 2
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
            stage_id=stage_id,
        )

        # Optionally apply a per-pixel opacity.
        scales = rearrange(scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
        rotations = rearrange(gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw")
        return_gaussians = Gaussians(
            rearrange(
                gaussians.means.float(),
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances.float(),
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics.float(),
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                (opacity_multiplier * gaussians.opacities).float(),
                "b v r srf spp -> b (v r srf spp)",
            ),
        )
        return return_gaussians, scales, rotations


        
    def forward(self,images,intrinsics,extrinsics,near,far,scene_names):

        device = images.device
        b, v, _, h, w = images.shape
        # Encode the context images.
        epipolar_kwargs = None
        
        deterministic = False
    
        
        context = dict()
        
        context["image"] = images
        context["intrinsics"] = intrinsics
        context["extrinsics"] = extrinsics
        context["near"] = near
        context["far"] = far

        features_list = self.backbone(
            context,
            attn_splits=self.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        # Sample depths from the resulting features.
        in_feats = features_list
        extra_info = {}
        extra_info["images"] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        extra_info["global_step"] = global_step
        gpp = self.gaussians_per_pixel
        
        gaussian_dict, result_dict = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            encoder=self,
        )
        
        
        # rendereing here
        # Debug here
        for i in range(len(gaussian_dict)):
            gaussians = gaussian_dict[f"stage{i}"]["gaussians"]
            rendered_color, rendered_depth = self.decoder.forward(
                gaussians,
                context["extrinsics"],
                context["intrinsics"],
                context["near"],
                context["far"],
                (h, w),
                depth_mode="depth",
            )
            

        
        return gaussian_dict, result_dict
        

        





if __name__ == "__main__":
    


    device = "cuda:0"
    global_step = 0
    
    # networks input configruations.
    from model_input_demo import create_mvsplat_demo_input
    batch_example, context, target = create_mvsplat_demo_input(device="cuda:0",
                              dtype=torch.float32,
                              image_height=112,
                              image_width=544,
                              batch_size=1,
                              num_context_views=2,
                              num_target_views=6)
    
    
    scene_name = batch_example["scene"]
    
    
    unimatch_weights_path = "/home/zliu/HiSplat/checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth"
    vit_path ="/home/zliu/HiSplat/checkpoints/dinov2_vitb14_pretrain.pth"
    
    
    
    # declare a decoder here for rendering.
    
    
    
    
    hisplat_encoder = HiSplatEncoder(unimatch_weights_path, vit_path)
    hisplat_encoder.to(device)
    
    # Create decoder for rendering
    hisplat_decoder = DecoderSplattingCUDA()
    hisplat_decoder.to(device)
    
    # Assign decoder to encoder (following original design pattern)
    hisplat_encoder.decoder = hisplat_decoder
    
    hisplat_encoder(batch_example["context"]["image"],
                    batch_example["context"]["intrinsics"],
                    batch_example["context"]["extrinsics"],
                    batch_example["context"]["near"],
                    batch_example["context"]["far"],
                    scene_name)
    
    print("currently is OK SO FAR")
    
    
    
    
    
    