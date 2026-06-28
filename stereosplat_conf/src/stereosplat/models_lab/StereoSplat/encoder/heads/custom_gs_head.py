import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import os
import numpy as np
from jaxtyping import Float

import sys
from ..unimatch.dpt_head import DPTHead
from einops import rearrange, einsum, repeat


def get_world_points_from_depth(depth, K, T):
    """
    将深度图转换为世界坐标下的点云。

    Args:
        depth: (B, V, H, W) 深度图，单位为米
        K: (B, V, 3, 3) 相机内参
        T: (B, V, 4, 4) cam2world 外参矩阵

    Returns:
        xyz_world: (B, V, 3, H, W) 世界坐标点云
    """
    B, V, H, W = depth.shape
    device = depth.device

    # 像素网格坐标：使用像素中心点 (x+0.5, y+0.5)
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    x = x + 0.5
    y = y + 0.5
    ones = torch.ones_like(x)

    # (3, H, W)
    pix_coords = torch.stack((x, y, ones), dim=0)

    # (B, V, 3, H, W)
    pix_coords = pix_coords.unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1, 1)

    # 内参逆矩阵 (B, V, 3, 3)
    K_inv = torch.inverse(K)

    # 像素坐标 -> 相机坐标单位向量 (B, V, 3, H*W)
    cam_coords = K_inv @ pix_coords.view(B, V, 3, -1)  # (B, V, 3, H*W)
    cam_coords = cam_coords.view(B, V, 3, H, W)        # (B, V, 3, H, W)

    # 相机坐标系下3D点 (B, V, 3, H, W)
    xyz_cam = cam_coords * depth.unsqueeze(2)

    # 齐次坐标 (B, V, 4, H, W)
    ones = torch.ones((B, V, 1, H, W), device=device)
    xyz_homo = torch.cat([xyz_cam, ones], dim=2)

    # 展平为 (B, V, 4, H*W)
    xyz_homo_flat = xyz_homo.view(B, V, 4, -1)

    # cam2world 变换 (B, V, 4, H*W)
    xyz_world_flat = T @ xyz_homo_flat

    # reshape 回 (B, V, 3, H, W)
    xyz_world = xyz_world_flat.view(B, V, 4, H, W)[..., :3, :, :]

    return xyz_world



# Gaussain Estimation Head
class Custom_Gaussain_Head(nn.Module):
    def __init__(self,
                 monodepth_vit_type,
                 upsample_factor,
                 num_scales,
                 gaussian_regressor_channels,
                 ):
        super().__init__()
        
        self.num_scales = num_scales
        
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
        

        ''' 
        First encode the 
        - "image", - "depth", - "match_prob", - "features" 
        
        Decode the
        - raw feature is 64
        '''
        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = gaussian_regressor_channels
        
        # conv regressor
        modules = [ nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),]

        self.gaussian_regressor = nn.Sequential(*modules)

        # 15 = xyz(3) + rgb(3) + opacity(1) + rotation(4) + scale(3) + conf(1)
        self.num_gaussian_parameters = 15
        

        ''' Feature 2: What is this used for? '''
        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1
        
        self.gaussain_aggregator = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(128, 128, 3, 1, 1),
            
        )
        
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(128, self.num_gaussian_parameters,
                      3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(self.num_gaussian_parameters,
                      self.num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
        )

        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: torch.exp(x) * 0.01
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        self.conf_act = torch.sigmoid
        
        self.delta_clamp = lambda x: x.clamp(-10.0, 6.0)
        self.delta_act = torch.exp


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
                return_depth=True,
                cfg=None):
        
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
        features = torch.cat(concat, dim=1) # torch.Size([2, 132, 224, 832])
        
        features = self.gaussain_aggregator(features)
        
        gaussians = self.gaussian_head(features)  # [BV, C, H, W]
        
        
        depths = depth 
        gaussians = rearrange(gaussians, "(b v) c h w -> b (v h w) c",
                              b=b, v=v, c=self.num_gaussian_parameters) #(B,V*H*W,15)
        


        offsets = gaussians[..., :3] # three dimension: offset
        opacities = self.opt_act(gaussians[..., 3:4]) # opcaity
        scales = self.scale_act(gaussians[..., 4:7])  # scale, 3-dimension
        rotations = self.rot_act(gaussians[..., 7:11]) # rotations, 4-dimension, quard
        rgbs = self.rgb_act(gaussians[..., 11:14]) # RGB
        confs = self.conf_act(gaussians[..., 14:15]) # confidence, 1-dimension, in (0,1)
        
        
        if not cfg.used_3D_offset:
            means = get_world_points_from_depth(depth=depths,
                                                K=intrinsics,
                                                T=extrinsics)
            means = rearrange(means, "b v c h w -> b (v h w) c",
                                b=b, v=v, c=3) #(B,V*H*W,14)
        else:
            means = get_world_points_from_depth(depth=depths, K=intrinsics, T=extrinsics)   # [B,V,3,H,W]
            means = rearrange(means, "b v c h w -> b (v h w) c", b=b, v=v, c=3)             # [B, VHW, 3]

            # --- 如果需要 3D offset：将 camera 系 offset 旋到 world 再相加 ---
            # R: [B,V,3,3] -> 重复到 [B,VHW,3,3]
            R = extrinsics[:, :, :3, :3]                                               # [B,V,3,3]
            R_rep = repeat(R, "b v i j -> b (v h w) i j", h=h, w=w)                    # [B, VHW, 3,3]

            # 将 offsets 视为 camera 坐标的偏移（做个温和限幅/尺度，默认 ~5cm，可在 cfg 里覆盖）
            offset_scale = getattr(cfg, "offset_scale", 0.05)                           # 0.05m ≈ 5 cm
            offsets_cam = torch.tanh(offsets) * offset_scale                            # [B, VHW, 3]

            # 旋到 world：delta_world = R * offsets_cam
            delta_world = einsum(R_rep, offsets_cam, "b n i j, b n j -> b n i")         # [B, VHW, 3]
            means = means + delta_world
                
        
        # layout: mean(0:3) + rgb(3:6) + opacity(6:7) + rotation(7:11) + scale(11:14) + conf(14:15)
        gaussians = torch.cat([means, rgbs, opacities, rotations, scales, confs], dim=-1)
 

        features = rearrange(features, "(b v) c h w -> b (v h w) c", b=b, v=v)
        features = features.unsqueeze(2) # b v*h*w n c
        features = rearrange(features, "b r n c -> b (r n) c")

        return gaussians, features,depths

        


    