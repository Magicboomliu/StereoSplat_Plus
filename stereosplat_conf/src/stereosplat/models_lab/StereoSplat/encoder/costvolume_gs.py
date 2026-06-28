import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import math

from .unimatch.mv_unimatch import MultiViewUniMatch
from .heads.custom_gs_head import Custom_Gaussain_Head
from safetensors.torch import load_file


def sanitize_gaussians_tensor(gaussians: torch.Tensor):
    if torch.isnan(gaussians).any() or torch.isinf(gaussians).any():
        print("[Sanitize] Invalid values found → fixing...")

    gaussians = gaussians.clone()  # 避免 in-place 修改原图计算图
    # 0:3 mean3D
    mean3D = torch.nan_to_num(gaussians[..., 0:3], nan=0.0, posinf=0.0, neginf=0.0)
    # 3:6 RGB
    rgb = torch.nan_to_num(gaussians[..., 3:6], nan=0.0, posinf=0.0, neginf=0.0)
    # rgb = torch.clamp(rgb, 0.0, 1.0)
    # 6:7 opacity
    opacity = torch.nan_to_num(gaussians[..., 6:7], nan=0.0, posinf=10.0, neginf=-10.0)
    opacity = torch.clamp(opacity, -10.0, 10.0)
    # 7:11 rotation
    rotation = gaussians[..., 7:11]
    norm = torch.norm(rotation, dim=-1, keepdim=True)
    bad_mask = (
        (norm < 1e-6)
        | torch.isnan(rotation).any(dim=-1, keepdim=True)
        | torch.isinf(rotation).any(dim=-1, keepdim=True)
    )
    # 清理数值 + 归一化
    norm = torch.clamp(norm, min=1e-6)
    rotation = torch.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0)
    rotation = rotation / norm
    # fallback 仅对异常数据赋值
    if bad_mask.any():
        fallback_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rotation.device)
        fallback_expand = fallback_quat.expand(bad_mask.sum(), 4)
        rotation[bad_mask.expand_as(rotation)] = fallback_expand

    # 11:14 scale
    scale = torch.nan_to_num(gaussians[..., 11:14], nan=1.0, posinf=1.0, neginf=1.0)
    scale = torch.clamp(scale, min=1e-6)

    parts = [mean3D, rgb, opacity, rotation, scale]

    # 14:15 conf (optional, only present in 15D layout)
    if gaussians.shape[-1] >= 15:
        conf = torch.nan_to_num(gaussians[..., 14:15], nan=0.5, posinf=1.0, neginf=0.0)
        conf = torch.clamp(conf, 0.0, 1.0)
        parts.append(conf)

    cleaned = torch.cat(parts, dim=-1)
    return cleaned


class CostVolumeGS(nn.Module):
    def __init__(self,
                 depth_estimator_kwargs:dict,
                 gaussain_head_kwargs:dict,
                 **kwargs
                 ):
        super().__init__()
        
        # depth unimatch model
        self.depth_estimator = MultiViewUniMatch(
                num_scales=depth_estimator_kwargs.num_scales, # default is 1
                upsample_factor=depth_estimator_kwargs.upsample_factor, # upsample factor is 4
                lowest_feature_resolution=depth_estimator_kwargs.lowest_feature_resolution, # 4
                vit_type=depth_estimator_kwargs.monodepth_vit_type, # 'vits'
                unet_channels=depth_estimator_kwargs.depth_unet_channels, # 128
                grid_sample_disable_cudnn=depth_estimator_kwargs.grid_sample_disable_cudnn, # False, Grid Sampling 
            )
        
        if depth_estimator_kwargs.unimatch_weights_path=='None':
            self.unimatch_weight = None
        else:
            self.unimatch_weight = depth_estimator_kwargs.unimatch_weights_path
            
            
        if self.unimatch_weight is not None:
            state_dict = load_file(self.unimatch_weight)  # 返回的是一个 PyTorch state_dict 格式的字典
        
            stripped_state_dict = {
                k.replace("depth_estimator.", "", 1): v for k, v in state_dict.items()
            }
            self.depth_estimator.load_state_dict(stripped_state_dict, strict=True)
            print("depth branch initailzation with {}".format(self.unimatch_weight))
        
        # define the 3DGS Head
        
        self.gaussains_estimation_head = Custom_Gaussain_Head(**gaussain_head_kwargs)
        


    
    def forward(self,input_batch_dict=None,
                images_feat=None,
                cfg=None):


        depth_max_value = cfg.max_depth # 100
        depth_min_value = cfg.min_depth # 0.3    
        
        # inputs information
        input_images = input_batch_dict['imgs'] # [B,V,3,H,W]
        intrinsics = input_batch_dict['intrinsics'] # [B,V,3,3]
        input_extrinsics = input_batch_dict['extrinsics'] # [B,V,4,4]
        input_nn_matrix = input_batch_dict['nn_matrix'] #[B,V,K]
        bs = input_images.shape[0]
        input_pseudo_depth = input_batch_dict['pseudo_depths']
        input_sparse_gt_depth = input_batch_dict['sparse_depths']

        mask = input_sparse_gt_depth > 0
        mask = mask.float()
        input_nn_matrix = input_nn_matrix.long()


        num_of_cameras = input_images.shape[1]
        min_depth=1.0 / depth_max_value  # inverse depth range
        max_depth=1.0 / depth_min_value
        
        min_depth = torch.from_numpy(np.array(min_depth)).unsqueeze(0).repeat(bs,num_of_cameras).type_as(input_images)
        max_depth = torch.from_numpy(np.array(max_depth)).unsqueeze(0).repeat(bs,num_of_cameras).type_as(input_images)
        
        
        # debug here
        intrinsics = intrinsics.clone()

        
        results_dict = self.depth_estimator(
            images=input_images,
            images_feat=images_feat,
            attn_splits_list=[2],
            intrinsics=intrinsics,
            min_depth=min_depth,  # inverse depth range
            max_depth=max_depth,
            num_depth_candidates=192, # here I set it to 192
            extrinsics=input_extrinsics,
            nn_matrix=input_nn_matrix
        )
        
        predicted_input_depth = results_dict['depth_preds'][0]
        
        if cfg.train_depth_only:
            return predicted_input_depth
        
        
        else:
            # estimated the gs 
            # change the head here
            gaussians_cv, features,pred_depths = self.gaussains_estimation_head(imgs=input_images,
                                           extrinsics=input_extrinsics,
                                           intrinsics = intrinsics,
                                           results_dict=results_dict,
                                           return_depth=False,
                                           cfg=cfg)
            
        
        
        return gaussians_cv,features,pred_depths

    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

