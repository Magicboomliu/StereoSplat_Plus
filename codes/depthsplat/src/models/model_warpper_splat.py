from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import moviepy.editor as mpy
import wandb
from einops import pack, rearrange, repeat, einsum
from jaxtyping import Float
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import json
import os
import math
from PIL import Image
import torchvision.transforms as T

#FIXME Here
from encoder.unimatch.mv_unimatch import MultiViewUniMatch
from encoder.unimatch.dpt_head import DPTHead
import numpy as np


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    weight_decay: float



class ModelWarpper(nn.Module):
    def __init__(self, 
                 depth_estimator=None,
                 **kwargs,
                 ):
        super().__init__()
        
        # depth estimation
        self.depth_estimator = depth_estimator
        
        # 3D Gaussains Estimation Head
        
        
    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype


    def forward(self,batch, mode="train", iter=0, cfg=None):
        
        iter_end = cfg.max_train_steps 
        
        depth_max_value = cfg.max_depth # 100
        depth_min_value = cfg.min_depth # 0.3    
        
        images = batch['imgs'] # [B,V,3,H,W]
        bs = images.shape[0]
        intrinsics = batch['intrinsics'] # [B,V,3,3]
        extrinsics = batch['extrinsics'] # [B,V,4,4]
        nn_matrix = batch['nn_matrix'] #[B,V,K]
        
        pseudo_depth = batch['pseudo_depths']
        sparse_gt_depth = batch['sparse_depths']

        mask = sparse_gt_depth>0
        mask = mask.float()

        
        num_of_cameras = images.shape[1]
        
        min_depth=1.0 / depth_max_value  # inverse depth range
        max_depth=1.0 / depth_min_value
        
        
        min_depth = torch.from_numpy(np.array(min_depth)).unsqueeze(0).repeat(bs,num_of_cameras).type_as(images)
        max_depth = torch.from_numpy(np.array(max_depth)).unsqueeze(0).repeat(bs,num_of_cameras).type_as(images)
        
        height, width = images.shape[3:]
        
        intrinsics = intrinsics.clone()
        # Normalized the instrinsics -----> Maybe not neccssary
        intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/width
        intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/height
        

        results_dict = self.depth_estimator(
            images=images,
            attn_splits_list=[2],
            intrinsics=intrinsics,
            min_depth=min_depth,  # inverse depth range
            max_depth=max_depth,
            num_depth_candidates=192, # here I set it to 192
            extrinsics=extrinsics,
            nn_matrix=nn_matrix
        )
        


if __name__=="__main__":
    
    class CFG(object):
        def __init__(self,max_train_steps,max_depth,min_depth,train_depth_only):
            self.max_train_steps= max_train_steps
            self.max_depth = max_depth
            self.min_depth = min_depth
            self.train_depth_only = train_depth_only
            
    
    input_images = torch.randn(1,2,3,224,832).cuda() # batch is 2, 0 is left and the 1 is the right
    
    b, v, _, h, w = input_images.shape
        
    cameras_dist_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=input_images.device)  # [2, 2]
    cameras_dist_index = cameras_dist_index.unsqueeze(0).expand(1, -1, -1)  # [B, 2, 2]
    intrinsics =   torch.Tensor([[552.554261,   0,       682.049453],
                        [  0, 552.554261, 238.769549],
                        [  0, 0,    1]]).unsqueeze(0).unsqueeze(0).repeat(1,2,1,1).type_as(input_images)
    

    T_left = torch.eye(4).type_as(input_images).unsqueeze(0).unsqueeze(0)
    T_right = torch.eye(4)
    T_right[0, 3] = 0.59  # 沿 x 轴右移 0.59 米
    T_right = T_right.type_as(input_images).unsqueeze(0).unsqueeze(0)
    
    extrinsics = torch.cat((T_left,T_right),dim=1)
    extrinsics = extrinsics.repeat(1,1,1,1)
    min_depth=1.0 / 100,  # inverse depth range
    max_depth=1.0 / 0.3,
    
    min_depth = torch.Tensor(min_depth).unsqueeze(0).repeat(1,2)
    max_depth = torch.Tensor(max_depth).unsqueeze(0).repeat(1,2)
    intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/832
    intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/224

    
    depth_estimator_unimatch = MultiViewUniMatch(
            num_scales=1, # default is 1
            upsample_factor=4, # upsample factor is 4
            lowest_feature_resolution=4, # 4
            vit_type="vits", # 'vits'
            unet_channels=192, # 128
            grid_sample_disable_cudnn=False, # False, Grid Sampling 
        )
    
    depth_estimator_unimatch = depth_estimator_unimatch
    
    my_model = ModelWarpper(depth_estimator=depth_estimator_unimatch)
    
    my_model = my_model.cuda()
    
    
    batch = dict()
    batch['imgs'] = input_images.cuda()
    batch['intrinsics']= intrinsics.cuda()
    batch['extrinsics']= extrinsics.cuda()
    batch['nn_matrix'] = cameras_dist_index.cuda()
    batch['pseudo_depths'] = torch.abs(torch.randn(1,2,224,832)*10-10).cuda()
    batch['sparse_depths'] = torch.abs(torch.randn(1,2,224,832)*10-10).cuda()
    
    cfg = CFG(max_train_steps=1000,max_depth=150,min_depth=0.3)
    
    with torch.no_grad():
        my_model(batch, mode="train", iter=0, cfg=cfg)