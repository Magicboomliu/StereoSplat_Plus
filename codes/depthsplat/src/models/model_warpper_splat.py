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
from .encoder.unimatch.mv_unimatch import MultiViewUniMatch
from .encoder.unimatch.dpt_head import DPTHead
import numpy as np
from .encoder.heads.gaussains_head import Gaussains_Estimator_Head,GaussianAdapterCfg
from torch import Tensor, nn
# Decoder Here
from .decoder.decoder_splatting_head_cuda import DecoderSplattingCUDA
from safetensors.torch import load_file
# vis here
import matplotlib.pyplot as plt
from .rgb_loss import LPIPS
from .utils import maybe_resize


def compute_depth_mae_mse(depth_pred, depth_gt, valid_min=0.0, valid_max=150.0):
    """
    Computes MAE and MSE between predicted and GT depth maps, with optional valid range filtering.

    Args:
        depth_pred (torch.Tensor): Predicted depth, shape [B, V, H, W]
        depth_gt (torch.Tensor): Ground truth depth, shape [B, V, H, W]
        valid_min (float): minimum valid GT depth
        valid_max (float): maximum valid GT depth

    Returns:
        mae (torch.Tensor): scalar mean absolute error
        mse (torch.Tensor): scalar mean squared error
    """
    assert depth_pred.shape == depth_gt.shape, "Shape mismatch between prediction and GT"

    # Create valid mask (only use pixels with valid GT depth)
    valid_mask = (depth_gt > valid_min) & (depth_gt < valid_max)

    # Compute errors
    abs_error = torch.abs(depth_pred - depth_gt)
    sq_error = (depth_pred - depth_gt) ** 2

    # Apply mask
    abs_error = abs_error[valid_mask]
    sq_error = sq_error[valid_mask]

    # Final metrics
    mae = abs_error.mean()
    mse = sq_error.mean()

    return mae, mse


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
                 unimatch_weight = None,
                 **kwargs,
                 ):
        super().__init__()
        # Depth Estimation
        self.depth_estimator = depth_estimator        
        # 3D Gaussains Estimation Head
        self.gaussains_estimation_head = gaussain_head
        # decoder branch
        self.decoder_branch = decoder_branch
        
        self.unimatch_weight  = unimatch_weight
        
        if self.unimatch_weight is not None:
            state_dict = load_file(self.unimatch_weight)  # 返回的是一个 PyTorch state_dict 格式的字典
        
            stripped_state_dict = {
                k.replace("depth_estimator.", "", 1): v for k, v in state_dict.items()
            }
            self.depth_estimator.load_state_dict(stripped_state_dict, strict=True)
            print("depth branch initailzation with {}".format(self.unimatch_weight))
        
        # preception loss here
        self.perceptual_loss = LPIPS().eval()
        
        
        
         
    
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
    
    
    

    def forward(self,batch, mode="train", iter=0, cfg=None):
        
        input_batch_dict,output_batch_dict = self.prepare_input_batch_data(batch=batch)
        
        return_depth = cfg.return_depth
        iter_end = cfg.max_train_steps 
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
        intrinsics = intrinsics.clone()
        # Normalized the instrinsics -----> Maybe not neccssary
        intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/width
        intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/height
        
        results_dict = self.depth_estimator(
            images=input_images,
            attn_splits_list=[2],
            intrinsics=intrinsics,
            min_depth=min_depth,  # inverse depth range
            max_depth=max_depth,
            num_depth_candidates=192, # here I set it to 192
            extrinsics=input_extrinsics,
            nn_matrix=input_nn_matrix
        )
        

        if cfg.train_depth_only:
            pass
            
        
        else:
            # estimated the gs 
            estimated_raw_gaussains_dict = self.gaussains_estimation_head(imgs=input_images,
                                           extrinsics=input_extrinsics,
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
        
        
        #first 2 dimension is the novel final ,final dimension is the input view
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        z_near_batch = torch.from_numpy(np.array([cfg.near])).unsqueeze(0).repeat(render_c2w.shape[0],render_c2w.shape[1]).type_as(render_c2w)
        z_far_batch = torch.from_numpy(np.array([cfg.far])).unsqueeze(0).repeat(render_c2w.shape[0],render_c2w.shape[1]).type_as(render_c2w)

        
        
        rendered_results = self.decoder_branch(gaussians=estimated_raw_gaussains_dict["gaussians"],
                                               extrinsics= render_c2w,
                                               intrinsics = output_intrinsics,
                                               near = z_near_batch,
                                               far = z_far_batch,
                                               image_shape=(height,width),
                                               depth_mode = 'depth'
                                               )

        rendered_color = rendered_results['color'] # torch.Size([1, V, 3, 224, 832])
        rendered_depth = rendered_results['depth'] # torch.Size([1, V, 1, 224, 832])
        rendered_alpha = rendered_results['alpha'] # torch.Size([1, V, 1, 224, 832])
        
        rendered_depth = rendered_depth.squeeze(2)
        rendered_alpha = rendered_alpha.squeeze(2)
        
        
        if mode=='train':
    
            # loss here
            # dict_keys(['output_imgs', 'output_depths', 'output_depths_m', 'output_confs_m', 
                        # 'output_positions', 'output_rays_o', 
                        # 'output_rays_d', 'output_c2ws', 
                        # 'output_fovxs', 'output_fovys'])

            # Get the Loss Here
            # ======================== losses ======================== #
            loss = 0.0
            loss_terms = {}
            def set_loss(key, split, loss_value, loss_weight=1.0):
                loss_terms[f"{split}/loss_{key}"] = loss_value.item()
                loss_terms[f"{split}/loss_{key}_w"] = loss_value.item() * loss_weight
            
            output_rgb = output_batch_dict['output_imgs']
            pseudo_depth_gt = output_batch_dict['output_depths_m']
            sparse_depth_gt = output_batch_dict['output_sparse_depth']
            
            # RGB Loss Here
            if cfg.loss_settings_dict.rendered_rgb_supervision:
                rgb_loss_total = 0
                if cfg.loss_settings_dict.rendered_rgb_supervison_type=="MSE":
                    rec_loss = ((output_rgb - rendered_color) ** 2).mean()
                    rgb_loss_total = rec_loss *1.0
                elif cfg.loss_settings_dict.rendered_rgb_supervison_type=="MSE_LPIPS":
                    rec_loss = F.mse_loss(output_rgb,rendered_color)
                    # preception loss
                    current_height, current_width = output_rgb.shape[-2:]
                    
                    preception_loss = self.perceptual_loss(output_rgb.reshape(-1,3,current_height,current_width),
                                                        rendered_color.reshape(-1,3,current_height,current_width)
                                                        )
                    preception_loss = preception_loss.mean()
                    
                    lpips_loss_alpha = cfg.loss_settings_dict.lpips_alpha
                    
                    rgb_loss_total = rec_loss + lpips_loss_alpha * preception_loss
                
                
                loss +=rgb_loss_total
                
                set_loss(key='rgb_loss',split=mode,loss_value=rgb_loss_total,
                                                    loss_weight=1.0)
            
            # rendered depth loss here
            if cfg.loss_settings_dict.rendered_depth_supervision:
                if cfg.loss_settings_dict.rendered_depth_supervision_type =='sparse_gt':
                    valid_mask_01 = sparse_depth_gt>0
                    valid_mask_02 = sparse_depth_gt<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    
                    sparse_depth_loss = F.l1_loss(sparse_depth_gt[valid_mask],rendered_depth[valid_mask])
                    sparse_depth_loss = sparse_depth_loss * cfg.loss_settings_dict.rendered_depth_weight

                    depth_loss = sparse_depth_loss
                    
                elif cfg.loss_settings_dict.rendered_depth_supervision_type =='pseudo':
                    
                    valid_mask_01 = pseudo_depth_gt>0
                    valid_mask_02 = pseudo_depth_gt<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    
                    pseudo_depth_loss = F.l1_loss(pseudo_depth_gt[valid_mask],rendered_depth[valid_mask])
                    pseudo_depth_loss = pseudo_depth_loss * cfg.loss_settings_dict.rendered_depth_weight
                    
                    depth_loss = pseudo_depth_loss
                    
                elif cfg.loss_settings_dict.rendered_depth_supervision_type =='sparse_gt_pseudo':
                    valid_mask_01 = sparse_depth_gt>0
                    valid_mask_02 = sparse_depth_gt<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    
                    sparse_depth_loss = F.l1_loss(sparse_depth_gt[valid_mask],rendered_depth[valid_mask])
                    sparse_depth_loss = sparse_depth_loss * cfg.loss_settings_dict.rendered_depth_weight


                    valid_mask_01 = pseudo_depth_gt>0
                    valid_mask_02 = pseudo_depth_gt<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    
                    pseudo_depth_loss = F.l1_loss(pseudo_depth_gt[valid_mask],rendered_depth[valid_mask])
                    pseudo_depth_loss = pseudo_depth_loss * cfg.loss_settings_dict.rendered_depth_weight

                    depth_loss = (sparse_depth_loss + pseudo_depth_loss) *1.0/2.0
                    
                else:
                    raise NotImplementedError

                loss +=depth_loss
                set_loss(key='rendered_depth_loss',split=mode,loss_value=depth_loss,
                                                    loss_weight=cfg.loss_settings_dict.rendered_depth_weight)
            
            
            # depth estimation loss here
            if cfg.loss_settings_dict.depth_estimator_supervision:
                # everything is OK
                if cfg.loss_settings_dict.depth_estimator_suppervision_type =='sparse_gt':
                    pred_depth = results_dict['depth_preds'][0]
                    valid_mask_01 = input_sparse_gt_depth>0
                    valid_mask_02 = input_sparse_gt_depth<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    sparse_depth_estimation_loss = F.l1_loss(input_sparse_gt_depth[valid_mask],pred_depth[valid_mask])            
                    depth_estimation_loss = sparse_depth_estimation_loss * cfg.loss_settings_dict.depth_estimation_weight
                
                elif cfg.loss_settings_dict.depth_estimator_suppervision_type =='pseudo':
                    
                    pred_depth = results_dict['depth_preds'][0]
                    valid_mask_01 = input_pseudo_depth>0
                    valid_mask_02 = input_pseudo_depth<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    pseudo_depth_estimation_loss = F.l1_loss(input_pseudo_depth[valid_mask],pred_depth[valid_mask])            
                    depth_estimation_loss = pseudo_depth_estimation_loss * cfg.loss_settings_dict.depth_estimation_weight

                
                elif cfg.loss_settings_dict.depth_estimator_suppervision_type =='sparse_gt_pseudo':
                    
                    pred_depth = results_dict['depth_preds'][0]
                    valid_mask_01 = input_pseudo_depth>0
                    valid_mask_02 = input_pseudo_depth<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    pseudo_depth_estimation_loss = F.l1_loss(input_pseudo_depth[valid_mask],pred_depth[valid_mask])            
                    

                    valid_mask_01 = input_sparse_gt_depth>0
                    valid_mask_02 = input_sparse_gt_depth<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    sparse_depth_estimation_loss = F.l1_loss(input_sparse_gt_depth[valid_mask],pred_depth[valid_mask])  

                    
                    depth_estimation_loss = 0.5 *(pseudo_depth_estimation_loss+sparse_depth_estimation_loss) * cfg.loss_settings_dict.depth_estimation_weight
                    

                loss +=depth_estimation_loss
                set_loss(key='depth_estimation_loss',split=mode,loss_value=depth_estimation_loss,
                                                    loss_weight=cfg.loss_settings_dict.depth_estimation_weight)

            
            return loss, loss_terms,rendered_color,rendered_depth,rendered_alpha,estimated_raw_gaussains_dict

        elif mode=='val' or mode=='test':
            
            return rendered_color,rendered_depth,rendered_alpha,estimated_raw_gaussains_dict
            
        
        

        

            
        
# if __name__=="__main__":
    
#     class CFG(object):
#         def __init__(self,max_train_steps,max_depth,min_depth,train_depth_only,return_depth):
#             self.max_train_steps= max_train_steps
#             self.max_depth = max_depth
#             self.min_depth = min_depth
#             self.train_depth_only = train_depth_only
#             self.return_depth = return_depth
    
#     class DatasetCFG(object):
#         def __init__(self,background_color=[0.0, 0.0, 0.0]):
#             self.background_color = background_color

            
#     #----------------------------------------------------------------------------------------------#
#     #---------------------------------Input Images and Inputs--------------------------------------#
#     #----------------------------------     And Inputs       --------------------------------------#
#     #----------------------------------------------------------------------------------------------#
    
#     input_images = torch.randn(1,2,3,224,832).cuda() # batch is 2, 0 is left and the 1 is the right
#     b, v, _, h, w = input_images.shape
    
#     cameras_dist_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=input_images.device)  # [2, 2]
#     cameras_dist_index = cameras_dist_index.unsqueeze(0).expand(1, -1, -1)  # [B, 2, 2]
#     intrinsics =   torch.Tensor([[552.554261,   0,       682.049453],
#                         [  0, 552.554261, 238.769549],
#                         [  0, 0,    1]]).unsqueeze(0).unsqueeze(0).repeat(1,2,1,1).type_as(input_images)
    

#     T_left = torch.eye(4).type_as(input_images).unsqueeze(0).unsqueeze(0)
#     T_right = torch.eye(4)
#     T_right[0, 3] = 0.59  # 沿 x 轴右移 0.59 米
#     T_right = T_right.type_as(input_images).unsqueeze(0).unsqueeze(0)
    
#     extrinsics = torch.cat((T_left,T_right),dim=1)
#     extrinsics = extrinsics.repeat(1,1,1,1)
#     min_depth=1.0 / 100,  # inverse depth range
#     max_depth=1.0 / 0.3,
    
#     min_depth = torch.Tensor(min_depth).unsqueeze(0).repeat(1,2)
#     max_depth = torch.Tensor(max_depth).unsqueeze(0).repeat(1,2)
#     intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/832
#     intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/224

    
#     z_near = 0.1
#     z_far = 1000.0
    
    
#     z_near_batch = torch.from_numpy(np.array([z_near])).unsqueeze(0).repeat(1,2)
#     z_far_batch = torch.from_numpy(np.array([z_far])).unsqueeze(0).repeat(1,2)
    

    

#     '''   Encoder Part of This Model  '''
#     # Define the Unimatch Branch
#     depth_estimator_unimatch = MultiViewUniMatch(
#             num_scales=1, # default is 1
#             upsample_factor=4, # upsample factor is 4
#             lowest_feature_resolution=4, # 4
#             vit_type="vits", # 'vits'
#             unet_channels=192, # 128
#             grid_sample_disable_cudnn=False, # False, Grid Sampling 
#         )
#     depth_estimator_unimatch = depth_estimator_unimatch
    
    
#     # Define the the gaussain head
#     gaussian_adapter_config = {"gaussian_scale_min": 1e-10,
#                                 "gaussian_scale_max": 3,
#                                 "sh_degree": 2 }
    
#     gaussain_color_branch_config = {
#             "large_gaussian_head": False,
#             "color_large_unet": False,
#             "init_sh_input_img": True,
#             "feature_upsampler_channels": 64,
#             "gaussian_regressor_channels": 64,
#             "num_surfaces":1}
    
#     gaussain_head = Gaussains_Estimator_Head(monodepth_vit_type='vits',
#                                              upsample_factor=4,
#                                              num_scales=1,
#                                              gaussian_head_settings_dict=gaussian_adapter_config,
#                                              gaussians_color_branch_dict=gaussain_color_branch_config)
    
    
#     dataset_cfg = DatasetCFG(background_color=[0.0,0.0,0.0])

#     depthsplattercuda_decoder = DecoderSplattingCUDA(dataset_cfg=dataset_cfg)
    
    
#     my_model = ModelWarpper(depth_estimator=depth_estimator_unimatch,
#                             gaussain_head=gaussain_head,
#                             decoder_branch=depthsplattercuda_decoder
#                             )
    
#     my_model = my_model.cuda()
    

#     batch = dict()
#     batch['imgs'] = input_images.cuda()
#     batch['intrinsics']= intrinsics.cuda()
#     batch['extrinsics']= extrinsics.cuda()
#     batch['nn_matrix'] = cameras_dist_index.cuda()
#     batch['pseudo_depths'] = torch.abs(torch.randn(1,2,224,832)*10-10).cuda()
#     batch['sparse_depths'] = torch.abs(torch.randn(1,2,224,832)*10-10).cuda()
    
#     batch['near'] = z_near_batch.cuda()
#     batch['far'] = z_far_batch.cuda()
    
    
#     cfg = CFG(max_train_steps=1000,max_depth=150,min_depth=0.3,
#               train_depth_only=False,return_depth=True)
    
#     with torch.no_grad():
#         my_model(batch, mode="train", iter=0, cfg=cfg)
    
#     quit()