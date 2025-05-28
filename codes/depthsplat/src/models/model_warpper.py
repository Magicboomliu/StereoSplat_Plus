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
        
        num_of_cameras = images.shape[1]
        
        min_depth=1.0 / depth_max_value  # inverse depth range
        max_depth=1.0 / depth_min_value
        
        
        min_depth = torch.from_numpy(np.array(min_depth)).unsqueeze(0).repeat(bs,num_of_cameras).type_as(images)
        max_depth = torch.from_numpy(np.array(max_depth)).unsqueeze(0).repeat(bs,num_of_cameras).type_as(images)
        
        height, width = images.shape[3:]
        
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
        
        
        print(results_dict.keys()) 
        # dict_keys(['features_cnn_all_scales', 'features_cnn', 'features_mv', 'features_mono_intermediate', 'features_mono', 'depth_preds', 'match_probs'])
        quit()
        
        
        pass

