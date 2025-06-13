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

from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor


from .encoder.costvolume_gs import CostVolumeGS
from .volume.TriPlaneVolumetircGS import TriPlaneVolumetircGS

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    

class VolumeFusion(BaseModule):
    def __init__(self,
                 backbone=None, # feature extraction
                 neck=None,      # feature aggregation
                 costvolume_gs=None,
                 volume_gs = None,
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
        self.use_checkpoint = use_checkpoint
        
        # define the depthsplat gs estimation: expected output is the GS and the GS Feature
        self.costvolume_gs = CostVolumeGS(**costvolume_gs)
                
        self.tri_plane_volume_gs = TriPlaneVolumetircGS(encoder=volume_gs.encoder,
                                                        gs_decoder=None,
                                                        use_checkpoint = volume_gs.use_checkpoint
                                                        )
        



    def extract_img_feat(self, img, status="train"):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W) # reasonable

        # training, using the checkpoint check to save the memory
        # using DiNO ResNet50
        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, img, use_reentrant=False) # 设置 use_reentrant=False 表示使用 非递归式的 Autograd 实现（新的引擎），
        else:
            img_feats = self.backbone(img) # return a tuple,multiple resolution, here use 4
        # default ouput is 1/4, 1/8, 1/16, 1/32 (DINO-Like)
        img_feats = self.neck(img_feats) # BV, C, H, W # Neck is a FPN for multi-scale feature aggregation.
        img_feats_reshaped = []
        
        
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        
        return img_feats_reshaped
    
    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def prepare_input_batch_data(self,batch):
        
        device_id = self.device
        input_batch_dict = dict()
        output_batch_dict = dict()
        
        # dict_keys(['ck', 'c2w', 'cx', 'cy', 'fx', 'fy', 'rays_o', 'rays_d', 'depth_m', 'conf_m', 'sparse_gt_depth']
        # dict_keys(['rgb', 'c2w', 'fovx', 'fovy', 'rays_o', 'rays_d', 
                    # 'input_image_path', 'depth', 'depth_m', 'conf_m', 
                                        #'sparse_gt_depth'])
        bin_token_name = batch['bin_token']
        input_cam_batch_data = batch['inputs_pix']                                 
        input_batch_data = batch['inputs']
        
        input_rgb =  input_batch_data['rgb'] # torch.Size([1, 2, 3, 224, 840]) #(B,V,3,H,W)
        input_camera_intrinsics = input_cam_batch_data['ck'] #(B,V,3,3) 
        input_camera_extrinsics = input_cam_batch_data['c2w'] #(B,V,4,4)
        
        input_psuedo_depth = input_cam_batch_data['depth_m'] #(B,V,H,W)
        input_sparse_depth = input_cam_batch_data['sparse_gt_depth'] #(B,V,H,W)
        

        cameras_dist_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # [2, 2] [V,K]
        cameras_dist_index= cameras_dist_index.unsqueeze(0).repeat(input_sparse_depth.shape[0],1,1)
        
        
        # input_dict
        input_batch_dict['imgs'] = input_rgb.to(device_id, dtype=self.dtype)
        input_batch_dict['intrinsics'] = input_camera_intrinsics.to(device_id, dtype=self.dtype)
        input_batch_dict['extrinsics'] = input_camera_extrinsics.to(device_id, dtype=self.dtype)
        input_batch_dict['nn_matrix'] =cameras_dist_index.to(device_id, dtype=self.dtype)
        input_batch_dict['pseudo_depths'] = input_psuedo_depth.to(device_id, dtype=self.dtype)
        input_batch_dict['sparse_depths'] = input_sparse_depth.to(device_id, dtype=self.dtype)
        input_batch_dict['bin_token_name'] = bin_token_name
        
        # for volume-gs
        img_metas = []
        bs, v, c, h, w = batch["inputs"]["rgb"].shape
        for w2i in batch["inputs_vol"]["w2i"]:
            img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
        input_batch_dict["img_metas"] = img_metas
        
        # output dict
        # for render and loss and eval
        output_batch_dict["output_imgs"] = batch["outputs"]["rgb"].to(device_id, dtype=self.dtype)
        output_batch_dict["output_depths"] = batch["outputs"]["depth"].to(device_id, dtype=self.dtype)
        output_batch_dict["output_depths_m"] = batch["outputs"]["depth_m"].to(device_id, dtype=self.dtype)
        output_batch_dict["output_confs_m"] = batch["outputs"]["conf_m"].to(device_id, dtype=self.dtype)        
        output_batch_dict["output_positions"] = (batch["outputs"]["rays_o"] + batch["outputs"]["rays_d"] * \
                            batch["outputs"]["depth_m"].unsqueeze(-1)).to(device_id, dtype=self.dtype)
        output_batch_dict["output_rays_o"] = batch["outputs"]["rays_o"].to(device_id, dtype=self.dtype)
        output_batch_dict["output_rays_d"] = batch["outputs"]["rays_d"].to(device_id, dtype=self.dtype)
        output_batch_dict["output_c2ws"] = batch["outputs"]["c2w"].to(device_id, dtype=self.dtype)
        output_batch_dict["output_fovxs"] = batch["outputs"]["fovx"].to(device_id, dtype=self.dtype)
        output_batch_dict["output_fovys"] = batch["outputs"]["fovy"].to(device_id, dtype=self.dtype)
        output_batch_dict['output_sparse_depth'] = batch['outputs']['sparse_gt_depth'].to(device_id, dtype=self.dtype)
        

    
        return input_batch_dict,output_batch_dict
        
    
    def forward(self,batch,mode='train',iter=0,cfg=None):
        # get inpout_batch_dict
        input_batch_dict,output_batch_dict = self.prepare_input_batch_data(batch=batch)

        img =input_batch_dict["imgs"] #[B,6,3,H,W]
        bs = img.shape[0]
        '''
        - torch.Size([2, 6, 128, 56, 100]) ---> 1/4
        - torch.Size([2, 6, 128, 28, 50])  ---> 1/8
        - torch.Size([2, 6, 128, 14, 25])  ---> 1/16
        - torch.Size([2, 6, 128, 7, 13])   ---> 1/32
        '''
        img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
        
        # perform the cost volume-based 
        estimated_raw_gaussains_dict = self.costvolume_gs(input_batch_dict,cfg=cfg)
        
        
        if 'gs' in cfg.return_types:
            gaussians_cost_volume = estimated_raw_gaussains_dict['gs']
        else:
            gaussians_cost_volume = None
        
        if 'depth' in cfg.return_types:
            pred_depth = estimated_raw_gaussains_dict['depth']
        else:
            pred_depth = None
        
        if "feature" in cfg.return_types:
            gaussians_feat = estimated_raw_gaussains_dict['feature']
        else:
            gaussians_feat = None

        
        # print(gaussians_cost_volume.means.shape) # torch.Size([1, 487424, 3])
        # print(gaussians_cost_volume.covariances.shape) # torch.Size([1, 487424, 3, 3])
        # print(gaussians_cost_volume.harmonics.shape) # torch.Size([1, 487424, 3, 9])
        # print(gaussians_cost_volume.opacities.shape) # torch.Size([1, 487424])

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        
        gaussians_cv_mask, gaussians_feat_mask = [], []
        for b in range(gaussians_feat.shape[0]):
            # here the shape is [VHW]
            mask_cv_i = (gaussians_cost_volume.means[b, :, 0] >= x_start) & (gaussians_cost_volume.means[b, :, 0] <= x_end) & \
                        (gaussians_cost_volume.means[b, :, 1] >= y_start) & (gaussians_cost_volume.means[b, :, 1] <= y_end) & \
                        (gaussians_cost_volume.means[b, :, 2] >= z_start) & (gaussians_cost_volume.means[b, :, 2] <= z_end)
            
            # get the valid GS 
            valid_gs_means = gaussians_cost_volume.means[b][mask_cv_i]
            valid_gs_covariances = gaussians_cost_volume.covariances[b][mask_cv_i]
            valid_gs_harmonics = gaussians_cost_volume.harmonics[b][mask_cv_i]
            valid_gs_opacities = gaussians_cost_volume.opacities[b][mask_cv_i]
            
            valid_gs_cv = Gaussians(
                means= valid_gs_means,
                covariances= valid_gs_covariances,
                harmonics= valid_gs_harmonics,
                opacities= valid_gs_opacities
            )
            
            valid_gs_feature_cv = gaussians_feat[b][mask_cv_i]
            
            gaussians_cv_mask.append(valid_gs_cv)
            gaussians_feat_mask.append(valid_gs_feature_cv)
            
        
        # Perform the Volume GS
        
        print("Anything is OK So Far....")
        quit()
            

        
        



if __name__=="__main__":
    pass