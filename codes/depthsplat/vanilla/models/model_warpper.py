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
from .encoder.unimatch.mv_unimatch import MultiViewUniMatch
import numpy as np
from .loss import depth_l1_loss,depth_loss


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
        
        self.depth_estimator = depth_estimator


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
        
        # loss here
        psuedo_depth_loss = depth_l1_loss(depth_pred=results_dict['depth_preds'][0],
                                          depth_gt=pseudo_depth)
        
        gt_sparse_lidar_loss = 0
        
        total_loss = psuedo_depth_loss * 0.5 + gt_sparse_lidar_loss*1.0

        loss_terms = {}
        def set_loss(key, split, loss_value, loss_weight=1.0):
            loss_terms[f"{split}/loss_{key}"] = loss_value.item()
            loss_terms[f"{split}/loss_{key}_w"] = loss_value.item() * loss_weight
            
        set_loss("Total_Loss","train",total_loss,loss_weight=1.0)
        
        # dict_keys(['features_cnn_all_scales', 'features_cnn', 'features_mv', 'features_mono_intermediate', 'features_mono', 'depth_preds', 'match_probs'])
        
        if mode=='train':
            return total_loss, loss_terms,results_dict
        elif mode=='val' or mode=='test':
            return results_dict['depth_preds'][0]

