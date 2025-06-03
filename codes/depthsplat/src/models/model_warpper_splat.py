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
from encoder.heads.gaussains_head import Gaussains_Estimator_Head,GaussianAdapterCfg
from torch import Tensor, nn

# Decoder Here
from decoder.decoder_splatting_head_cuda import DecoderSplattingCUDA


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    weight_decay: float


class ModelWarpper(nn.Module):
    def __init__(self, 
                 depth_estimator=None,
                 gaussain_head = None,
                 
                 decoder_branch = None,
                 **kwargs,
                 ):
        super().__init__()
        # Depth Estimation
        self.depth_estimator = depth_estimator        
        
        # 3D Gaussains Estimation Head
        self.gaussains_estimation_head = gaussain_head

        # decoder branch
        self.decoder_branch = decoder_branch
    
    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self,batch, mode="train", iter=0, cfg=None):
        
        return_depth = cfg.return_depth
        
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

        if cfg.train_depth_only:
            pass
        
        else:
            # estimated the gs 
            estimated_raw_gaussains_dict = self.gaussains_estimation_head(imgs=images,
                                           extrinsics=extrinsics,
                                           intrinsics = intrinsics,
                                           results_dict=results_dict,
                                           return_depth=return_depth)
        
        # return values
        if len(estimated_raw_gaussains_dict.keys())>1:
            pred_depths = estimated_raw_gaussains_dict["depths"]
            gaussians = estimated_raw_gaussains_dict["gaussians"]
        else:
            gaussians = estimated_raw_gaussains_dict["gaussians"]
            pred_depths = None
          


        # 3DGS decoder
        rendered_results = self.decoder_branch(gaussians=estimated_raw_gaussains_dict["gaussians"],
                                               extrinsics= extrinsics,
                                               intrinsics = intrinsics,
                                               near = batch['near'].float(),
                                               far = batch['far'].float(),
                                               image_shape=(height,width),
                                               depth_mode = 'depth'
                                               )
        
        rendered_color = rendered_results['color']
        rendered_depth = rendered_results['depth']
        rendered_alpha = rendered_results['alpha']
        
        print(rendered_color.shape) 
        print(rendered_depth.shape)
        print(rendered_alpha.shape)
        quit()
        
        

            

        
if __name__=="__main__":
    
    class CFG(object):
        def __init__(self,max_train_steps,max_depth,min_depth,train_depth_only,return_depth):
            self.max_train_steps= max_train_steps
            self.max_depth = max_depth
            self.min_depth = min_depth
            self.train_depth_only = train_depth_only
            self.return_depth = return_depth
    
    class DatasetCFG(object):
        def __init__(self,background_color=[0.0, 0.0, 0.0]):
            self.background_color = background_color

            
    #----------------------------------------------------------------------------------------------#
    #---------------------------------Input Images and Inputs--------------------------------------#
    #----------------------------------     And Inputs       --------------------------------------#
    #----------------------------------------------------------------------------------------------#
    
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

    
    z_near = 0.1
    z_far = 1000.0
    
    
    z_near_batch = torch.from_numpy(np.array([z_near])).unsqueeze(0).repeat(1,2)
    z_far_batch = torch.from_numpy(np.array([z_far])).unsqueeze(0).repeat(1,2)
    

    

    '''   Encoder Part of This Model  '''
    # Define the Unimatch Branch
    depth_estimator_unimatch = MultiViewUniMatch(
            num_scales=1, # default is 1
            upsample_factor=4, # upsample factor is 4
            lowest_feature_resolution=4, # 4
            vit_type="vits", # 'vits'
            unet_channels=192, # 128
            grid_sample_disable_cudnn=False, # False, Grid Sampling 
        )
    depth_estimator_unimatch = depth_estimator_unimatch
    
    
    # Define the the gaussain head
    gaussian_adapter_config = {"gaussian_scale_min": 1e-10,
                                "gaussian_scale_max": 3,
                                "sh_degree": 2 }
    
    gaussain_color_branch_config = {
            "large_gaussian_head": False,
            "color_large_unet": False,
            "init_sh_input_img": True,
            "feature_upsampler_channels": 64,
            "gaussian_regressor_channels": 64,
            "num_surfaces":1}
    
    gaussain_head = Gaussains_Estimator_Head(monodepth_vit_type='vits',
                                             upsample_factor=4,
                                             num_scales=1,
                                             gaussian_head_settings_dict=gaussian_adapter_config,
                                             gaussians_color_branch_dict=gaussain_color_branch_config)
    
    
    dataset_cfg = DatasetCFG(background_color=[0.0,0.0,0.0])

    depthsplattercuda_decoder = DecoderSplattingCUDA(dataset_cfg=dataset_cfg)
    
    

    
    
    my_model = ModelWarpper(depth_estimator=depth_estimator_unimatch,
                            gaussain_head=gaussain_head,
                            decoder_branch=depthsplattercuda_decoder
                            )
    
    my_model = my_model.cuda()
    

    batch = dict()
    batch['imgs'] = input_images.cuda()
    batch['intrinsics']= intrinsics.cuda()
    batch['extrinsics']= extrinsics.cuda()
    batch['nn_matrix'] = cameras_dist_index.cuda()
    batch['pseudo_depths'] = torch.abs(torch.randn(1,2,224,832)*10-10).cuda()
    batch['sparse_depths'] = torch.abs(torch.randn(1,2,224,832)*10-10).cuda()
    
    batch['near'] = z_near_batch.cuda()
    batch['far'] = z_far_batch.cuda()
    
    
    cfg = CFG(max_train_steps=1000,max_depth=150,min_depth=0.3,
              train_depth_only=False,return_depth=True)
    
    with torch.no_grad():
        my_model(batch, mode="train", iter=0, cfg=cfg)
    
    quit()