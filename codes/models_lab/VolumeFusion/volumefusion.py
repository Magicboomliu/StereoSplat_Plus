import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange, einsum
from .encoder.costvolume_gs import CostVolumeGS


@MODELS.register_module()
class VolumeFusion(BaseModule):
    def __init__(self,
                 backbone=None, # feature extraction
                 neck=None,      # feature aggregation
                 costvolume_gs=None,
                 camera_args=None, # camera/3D Range
                #  loss_args=None,    # loss args setings
                 dataset_params=None, # dataset params
                 use_checkpoint=False, # using checkpoints or not
                 **kwargs,
                 ):
        super().__init__()
        if backbone:
            self.backbone = MODELS.build(backbone)
        if neck:
            self.neck = MODELS.build(neck)
        self.dataset_params = dataset_params
        self.camera_args = camera_args
        
        # define the depthsplat gs estimation: expected output is the GS and the GS Feature
        self.costvolume_gs = CostVolumeGS(**costvolume_gs)

        
    
    def forward(self,batch,mode='train',iter=0,cfg=None):
        
        
        pass




if __name__=="__main__":
    pass