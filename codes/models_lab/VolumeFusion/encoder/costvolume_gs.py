import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from .unimatch.mv_unimatch import MultiViewUniMatch
from safetensors.torch import load_file






class CostVolumeGS(nn.Module):
    def __init__(self,
                 depth_estimator_kwargs:dict,
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
        
        height, width = input_images.shape[3:]
        
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
        
        print(predicted_input_depth.shape)
        
        quit()
        
        

    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

