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
        data_dict = self.get_data(batch)
        
        images = batch['imgs'] # [B,V,3,H,W]
        bs = images.shape[0]
        intrinsics = batch['intrinsics'] # [B,V,3,3]
        extrinsics = batch['extrinsics'] # [B,V,4,4]
        nn_matrix = batch['nn_matrix'] #[B,V,K]
        
        print(images.shape)
        print(intrinsics.shape)
        print(extrinsics.shape)
        print(nn_matrix.shape)
        
        quit()
        
        
        pass

