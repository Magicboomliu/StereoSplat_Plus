import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import os
import numpy as np
from jaxtyping import Float

import sys
from ..unimatch.dpt_head import DPTHead
from ..common.gaussian_adapter import GaussianAdapter
from ..common.gaussian_adapter import GaussianAdapterCfg
from ..geometry.projection import sample_image_grid
from einops import rearrange
from dataclasses import dataclass

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]


# Gaussain Estimation Head
class Gaussains_Estimator_Head(nn.Module):
    def __init__(self,
                 monodepth_vit_type,
                 upsample_factor,
                 num_scales,
                 gaussian_head_settings_dict,
                 gaussians_color_branch_dict
                 
                 ):
        super().__init__()
        
        self.num_scales = num_scales
        self.gaussians_color_branch_dict = gaussians_color_branch_dict
        
        self.monodepth_vit_type = monodepth_vit_type
        self.upsample_factor = upsample_factor
        
        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},}
        
        self.feature_upsampler = DPTHead(**model_configs[monodepth_vit_type],
                                        downsample_factor=upsample_factor,
                                        return_feature=True,
                                        num_scales=num_scales,)
        
        feature_upsampler_channels = model_configs[monodepth_vit_type]["features"]
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(**gaussian_head_settings_dict)

        ''' 
        First encode the 
        - "image", - "depth", - "match_prob", - "features" 
        
        Decode the
        - raw feature is 64
        '''
        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = gaussians_color_branch_dict['gaussian_regressor_channels']
        
        # conv regressor
        modules = [ nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),]

        self.gaussian_regressor = nn.Sequential(*modules)

        # scale is 3, q is 4, sh is (3+sh)^2, 2,1
        # predict gaussian parameters: scale, q, sh, offset, opacity
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1 # 7+3x9+3
        


        ''' Feature 2: What is this used for? '''
        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1
        self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                          num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if gaussians_color_branch_dict['init_sh_input_img']:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:]) # here the weight is 10?
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])


        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])



    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self,imgs,
                extrinsics,
                intrinsics,
                results_dict,
                return_depth=True):
        
        depth_preds = results_dict['depth_preds']
        # [B, V, H, W]
        depth = depth_preds[-1]
        b,v = depth.shape[:2]
        
        device = depth.device
        h,w = depth.shape[-2:]
        


        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )


        ''' Using: Images + Depths + Matching Probalities '''
        match_prob = results_dict['match_probs'][-1] #[BV,D,H,W]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]------> Only Get the Max
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')
        concat = torch.cat((
            rearrange(imgs, "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)
        
        out = self.gaussian_regressor(concat) #[2,64,H,W]
        

        concat = [out,rearrange(imgs,"b v c h w -> (b v) c h w"),features,
                match_prob]
        out = torch.cat(concat, dim=1) # torch.Size([2, 132, 224, 832])

        gaussians = self.gaussian_head(out)  # [BV, C, H, W]
        
        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        depths = rearrange(depth, "b v h w -> b v (h w) () ()")
        
        

        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)  # [B, V, H*W, 1, 1]

        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c") #[B,V,HW,37]
        

        '''Opacity Estmation'''
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1) # index=0 is the opacity
        
        
        raw_gaussians = raw_gaussians[..., 1:] #(B,V,HW,C)

        # have been normalized
        xy_ray, _ = sample_image_grid((h, w), device) #(H,W,2)
        

        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy") #(H*W,1,2)
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.gaussians_color_branch_dict['num_surfaces'],
        ) #(B,V,HW,1,C)
        offset_xy = gaussians[..., :2].sigmoid() # offset
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = imgs #[B,V,3,H,W]
        


        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics,
                        "b v i j -> b v () () () i j"),
            rearrange(intrinsics,
                        "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            opacities,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
            input_images=sh_input_images if self.gaussians_color_branch_dict['init_sh_input_img'] else None,
        )
        
        # print(gaussians.means.shape) #(1,2,H*W,1,1,3)
        # print(gaussians.covariances.shape) #(B,V,H*W,1,1,3,3)
        # print(gaussians.harmonics.shape) #(B,V,H*W,1,1,3,9)
        # print(gaussians.opacities.shape) #(B,V,H*W,1,1)
        # print("------------------------------------")


        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        
        # print(gaussians.means.shape) #torch.Size([1, 372736, 3])
        # print(gaussians.covariances.shape) #torch.Size([1, 372736, 3, 3])
        # print(gaussians.harmonics.shape) #torch.Size([1, 372736, 3, 9])
        # print(gaussians.opacities.shape) #torch.Size([1, 372736])
        # print("------------------------------------")

        if return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]
 
            return {
                "gaussians": gaussians,
                "depths": depths
            }
            
        else:
            return {
                "gaussians": gaussians
            }
            

        # print(gaussians.means.shape) #(1,2,H*W,1,1,3)
        # print(gaussians.covariances.shape) #(B,V,H*W,1,1,3,3)
        # print(gaussians.harmonics.shape) #(B,V,H*W,1,1,3,9)
        # print(gaussians.opacities.shape) #(B,V,H*W,1,1)
        


if __name__=="__main__":
    pass