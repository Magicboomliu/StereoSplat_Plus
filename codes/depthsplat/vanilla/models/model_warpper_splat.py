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
import skimage.io
from .utils import interpolate_extrinsics
from tqdm import tqdm

from .metrics import compute_stereo_psnr_ssim,compute_all_stereo_psnr_ssim,convert_depth_to_disp,kitti_colormap
import math
import skimage.io
import random
from .depth_error_vis import disp_error_img,depths_to_colors

import moviepy.editor as mpy
import wandb
from PIL import Image
import time
import random
import os.path as osp


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

    # Concatenate all cleaned parts
    cleaned = torch.cat([mean3D, rgb, opacity, rotation, scale], dim=-1)
    return cleaned

def compute_depth_stereo_mae_mse(depth_pred, depth_gt,valid_min=0.0,valid_max=150.0):
    """
    Computes MAE and MSE between predicted and GT depth maps, with optional valid range filtering.
    """
    
    B,V,H,W = depth_pred.shape
    
    left_mae = 0
    right_mae = 0
    left_mse = 0
    right_mse = 0
    
    for i in range(V):
        if i%2 == 0:
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
            left_mae += mae
            left_mse += mse
        else:
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
            right_mae += mae
            right_mse += mse

    left_mae /= V//2
    right_mae /= V//2
    left_mse /= V//2
    right_mse /= V//2

    

    return left_mae, left_mse, right_mae, right_mse

def interleave_left_right(x: torch.Tensor) -> torch.Tensor:

    first_left_right = x[:, -2:, :, :,:]
    
    rest_views = x[:, :-2, :, :,:]

    B, twoN, C, H, W = rest_views.shape
    assert twoN % 2 == 0, "2N 必须是偶数"
    N = twoN // 2

    left  = rest_views[:, :N]   # (B, N, 3, H, W)
    right = rest_views[:, N:]   # (B, N, 3, H, W)


    # 堆叠后交替
    y = torch.empty_like(rest_views)
    y[:, 0::2] = left
    y[:, 1::2] = right
    
    return torch.cat((y,first_left_right),dim=1)
  
def interleave_left_right_depth(x: torch.Tensor) -> torch.Tensor:
    
    first_left_right = x[:, -2:, :, :]
    
    rest_views = x[:, :-2, :, :]

    B, twoN, H, W = rest_views.shape
    assert twoN % 2 == 0, "2N 必须是偶数"
    N = twoN // 2

    left  = rest_views[:, :N]   # (B, N, H, W)
    right = rest_views[:, N:]   # (B, N, H, W)

    y = torch.empty_like(rest_views)
    y[:, 0::2] = left
    y[:, 1::2] = right
    return torch.cat((y,first_left_right),dim=1)

def interleave_left_right_pose(x: torch.Tensor) -> torch.Tensor:
    
    first_left_right = x[:, -2:, :, :]
    
    rest_views = x[:, :-2, :, :]

    B, twoN, H, W = rest_views.shape
    assert twoN % 2 == 0, "2N 必须是偶数"
    assert H == 4 and W == 4, "应该是 4x4 相机矩阵"
    N = twoN // 2

    left  = rest_views[:, :N]   # (B, N, 4, 4)
    right = rest_views[:, N:]   # (B, N, 4, 4)

    y = torch.empty_like(rest_views)
    y[:, 0::2] = left
    y[:, 1::2] = right
    return torch.cat((y,first_left_right),dim=1)

def random_index(N):
    return random.randint(0, N - 1)

def compute_depth_stereo_mae_mse(depth_pred, depth_gt,valid_min=0.0,valid_max=150.0):
    """
    Computes MAE and MSE between predicted and GT depth maps, with optional valid range filtering.
    """
    B,V,H,W = depth_pred.shape
    
    left_mae = 0
    right_mae = 0
    left_mse = 0
    right_mse = 0
    
    for i in range(V):
        if i%2 == 0:
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
            left_mae += mae
            left_mse += mse
        else:
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
            right_mae += mae
            right_mse += mse

    left_mae /= V//2
    right_mae /= V//2
    left_mse /= V//2
    right_mse /= V//2


    return left_mae, left_mse, right_mae, right_mse


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
        
        if self.unimatch_weight=='None':
            self.unimatch_weight=None
        if self.unimatch_weight is not None:
            state_dict = load_file(self.unimatch_weight)  # 返回的是一个 PyTorch state_dict 格式的字典
        
            stripped_state_dict = {
                k.replace("depth_estimator.", "", 1): v for k, v in state_dict.items()
            }
            self.depth_estimator.load_state_dict(stripped_state_dict, strict=True)
            print("depth branch initailzation with {}".format(self.unimatch_weight))
        
        # preception loss here
        self.perceptual_loss = LPIPS().eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
        
        
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
        
        # debug here
        # intrinsics[:,:,0,2] = intrinsics[:,:,0,2] + 13
        # intrinsics[:,:,1,2] = intrinsics[:,:,1,2] - 30
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
        
        predicted_input_depth = results_dict['depth_preds'][0]
        

        if cfg.train_depth_only:
            pass
        
        
        else:
            # estimated the gs 
            # change the head here
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
        
        rendered_color = torch.clamp(rendered_color,min=0,max=1.0)
        rendered_depth = torch.clamp(rendered_depth,min=0,max=150)
        
        
        if mode=='train' or mode=='val':
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
                    valid_mask_01_float = valid_mask_01.float()
                    fusion_pseudo_with_sparse_gt = valid_mask_01_float * sparse_depth_gt + (1-valid_mask_01_float) * pseudo_depth_gt 
                    
                    valid_mask_02 = fusion_pseudo_with_sparse_gt >0
                    valid_mask_03 = fusion_pseudo_with_sparse_gt < 150
                    valid_mask = valid_mask_02 * valid_mask_03
                    valid_mask = valid_mask.bool()
                    
                    sparse_depth_loss = F.l1_loss(fusion_pseudo_with_sparse_gt[valid_mask],rendered_depth[valid_mask])
                    sparse_depth_loss = sparse_depth_loss * cfg.loss_settings_dict.rendered_depth_weight

                    depth_loss = sparse_depth_loss
                    
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
                    valid_mask_01 = input_sparse_gt_depth>0
                    
                    valid_mask_01_float = valid_mask_01.float()
                    input_pseudo_gt_fusion_depth = input_sparse_gt_depth * valid_mask_01_float \
                                    + (1-valid_mask_01_float)*input_pseudo_depth
                    valid_mask_02 = input_pseudo_gt_fusion_depth>0
                    valid_mask_03 = input_pseudo_gt_fusion_depth<150
                    valid_mask = valid_mask_02 * valid_mask_03
                    valid_mask = valid_mask.bool()
                    pseudo_depth_estimation_loss = F.l1_loss(input_pseudo_gt_fusion_depth[valid_mask],pred_depth[valid_mask])            
                    
                    
                    depth_estimation_loss = pseudo_depth_estimation_loss * cfg.loss_settings_dict.depth_estimation_weight
                    

                loss +=depth_estimation_loss
                set_loss(key='depth_estimation_loss',split=mode,loss_value=depth_estimation_loss,
                                                    loss_weight=cfg.loss_settings_dict.depth_estimation_weight)

            
            if mode=='train':
                return loss, loss_terms,rendered_color,rendered_depth,rendered_alpha,estimated_raw_gaussains_dict
            
            elif mode=='val':
                return loss, loss_terms,rendered_color,rendered_depth,rendered_alpha,estimated_raw_gaussains_dict,predicted_input_depth,input_sparse_gt_depth,output_rgb,sparse_depth_gt,input_images
            
        elif mode=='test':
            return rendered_color,rendered_depth,rendered_alpha,estimated_raw_gaussains_dict

        else:
            raise NotImplementedError
    
    def validation_step(self, batch, val_result_savedir,cfg=None):
        
        bin_token_name = batch['bin_token'][0][:-4]
    

        
        # loss and loss terms
        with torch.no_grad():
            loss, loss_terms,rendered_color,\
                rendered_depth,rendered_alpha,estimated_raw_gaussains_dict,\
                    predicted_input_depth,input_sparse_gt_depth,\
                        output_rgb,sparse_depth_gt,input_images = self.forward(batch,mode='val',
                                                            cfg=cfg)
        
        batch_data_for_eval = {
        "output_gt_rgb": output_rgb,
        "output_gt_sparse_depth": sparse_depth_gt,
        "input_images": input_images,
        "input_gt_sparse_gt": input_sparse_gt_depth,
        "predicted_input_depth": predicted_input_depth,
        "rendered_rgb": rendered_color,
        "rendered_depth": rendered_depth,
        "rendered_alpha": rendered_alpha,
        "estimated_raw_gs": estimated_raw_gaussains_dict,
        "bin_token_name": bin_token_name
        }
        
        # saved into the val_result_dir: the visualiation results
        
        # rendered RGBs
        # rendered Depths
        # GT RGBs
        # GT Depths
        # Estimated Depths
        
        output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict = self.save_val_results(batch_data_for_eval,val_result_savedir,cfg=cfg)
        
        return output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict
    
    def save_val_results(self,batch_data_for_eval,saved_dir,cfg):
        
        '''input batch data for evaluation'''
        
        output_rgb_meter_dict = dict()
        # get the psnr and ssim for the output view
        output_rendered_rgb = batch_data_for_eval['rendered_rgb'] #torch.Size([1, 6, 3, 224, 832])
        output_gt_rgb = batch_data_for_eval['output_gt_rgb'] #torch.Size([1, 6, 3, 224, 832])
        
        # rendered center
        center_frame_left_est =  output_rendered_rgb[:,0,:,:,:]
        center_frame_right_est = output_rendered_rgb[:,2,:,:,:]
        last_frame_left_est =  output_rendered_rgb[:,1,:,:,:]
        last_frame_right_est =  output_rendered_rgb[:,3,:,:,:]
        first_frame_left_est = output_rendered_rgb[:,4,:,:,:]
        first_frame_right_est = output_rendered_rgb[:,5,:,:,:]
        
        center_frame_left_gt =  output_gt_rgb[:,0,:,:,:]
        center_frame_right_gt = output_gt_rgb[:,2,:,:,:]
        last_frame_left_gt =  output_gt_rgb[:,1,:,:,:]
        last_frame_right_gt =  output_gt_rgb[:,3,:,:,:]
        first_frame_left_gt = output_gt_rgb[:,4,:,:,:]
        first_frame_right_gt = output_gt_rgb[:,5,:,:,:]
        
        
        # print(first_frame_left_est.shape)
        # print(first_frame_left_gt.shape)
        
        
        # skimage.io.imsave("/data1/zliu/feedforward_outputs/DepthSplat/Temp_Visualization/Debugs/left_est.png",(first_frame_left_est.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        # skimage.io.imsave("/data1/zliu/feedforward_outputs/DepthSplat/Temp_Visualization/Debugs/left_gt.png",(first_frame_left_gt.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        
   
        
        
        cl_psnr,cl_ssim = compute_psnr_ssim(pred=center_frame_left_est,target=center_frame_left_gt)
        cr_psnr,cr_ssim = compute_psnr_ssim(pred=center_frame_right_est,target=center_frame_right_gt)
        ll_psnr,ll_ssim = compute_psnr_ssim(pred=last_frame_left_est,target=last_frame_left_gt)
        lr_psnr,lr_ssim = compute_psnr_ssim(pred=last_frame_right_est,target=last_frame_right_gt)
        fl_psnr,fl_ssim = compute_psnr_ssim(pred=first_frame_left_est,target=first_frame_left_gt)
        fr_psnr,fr_ssim = compute_psnr_ssim(pred=first_frame_right_est,target=first_frame_right_gt)
        
        output_rgb_meter_dict['center_view'] = dict()
        output_rgb_meter_dict['center_view']['left'] = dict()
        output_rgb_meter_dict['center_view']['left']['psnr'] = cl_psnr.data.item()
        output_rgb_meter_dict['center_view']['left']['ssim'] = cl_ssim.data.item()

        output_rgb_meter_dict['center_view']['right'] = dict()
        output_rgb_meter_dict['center_view']['right']['psnr'] = cr_psnr.data.item()
        output_rgb_meter_dict['center_view']['right']['ssim'] = cr_ssim.data.item()
        

        output_rgb_meter_dict['last_view'] = dict()
        output_rgb_meter_dict['last_view']['left'] = dict()
        output_rgb_meter_dict['last_view']['left']['psnr'] = ll_psnr.data.item()
        output_rgb_meter_dict['last_view']['left']['ssim'] = ll_ssim.data.item()

        output_rgb_meter_dict['last_view']['right'] = dict()
        output_rgb_meter_dict['last_view']['right']['psnr'] = lr_psnr.data.item()
        output_rgb_meter_dict['last_view']['right']['ssim'] = lr_ssim.data.item()


        output_rgb_meter_dict['first_view'] = dict()
        output_rgb_meter_dict['first_view']['left'] = dict()
        output_rgb_meter_dict['first_view']['left']['psnr'] = fl_psnr.data.item()
        output_rgb_meter_dict['first_view']['left']['ssim'] = fl_ssim.data.item()

        output_rgb_meter_dict['first_view']['right'] = dict()
        output_rgb_meter_dict['first_view']['right']['psnr'] = fr_psnr.data.item()
        output_rgb_meter_dict['first_view']['right']['ssim'] = fr_ssim.data.item()

        
        # get the MAE and the MSE of the output view
        output_depth_meter_dict = dict()
        output_rendered_depth = batch_data_for_eval['rendered_depth'] #torch.Size([1, 6, 3, 224, 832])
        output_gt_depth = batch_data_for_eval['output_gt_sparse_depth'] #torch.Size([1, 6, 3, 224, 832])
        

        center_frame_left_est_depth =  output_rendered_depth[:,0,:,:]
        center_frame_right_est_depth = output_rendered_depth[:,2,:,:]
        last_frame_left_est_depth =  output_rendered_depth[:,1,:,:]
        last_frame_right_est_depth =  output_rendered_depth[:,3,:,:]
        first_frame_left_est_depth = output_rendered_depth[:,4,:,:]
        first_frame_right_est_depth = output_rendered_depth[:,5,:,:]

        center_frame_left_gt_depth =  output_gt_depth[:,0,:,:]
        center_frame_right_gt_depth = output_gt_depth[:,2,:,:]
        last_frame_left_gt_depth =  output_gt_depth[:,1,:,:]
        last_frame_right_gt_depth =  output_gt_depth[:,3,:,:]
        first_frame_left_gt_depth = output_gt_depth[:,4,:,:]
        first_frame_right_gt_depth = output_gt_depth[:,5,:,:]

        cl_mae,cl_mse = compute_depth_mae_mse(depth_pred=center_frame_left_est_depth,
                              depth_gt=center_frame_left_gt_depth)
        
        cr_mae,cr_mse = compute_depth_mae_mse(depth_pred=center_frame_right_est_depth,
                              depth_gt=center_frame_right_gt_depth)
        
        ll_mae,ll_mse = compute_depth_mae_mse(depth_pred=last_frame_left_est_depth,
                              depth_gt=last_frame_left_gt_depth)
        
        lr_mae,lr_mse = compute_depth_mae_mse(depth_pred=last_frame_right_est_depth,
                              depth_gt=last_frame_right_gt_depth)
        
        fl_mae,fl_mse = compute_depth_mae_mse(depth_pred=first_frame_left_est_depth,
                              depth_gt=first_frame_left_gt_depth)

        fr_mae,fr_mse = compute_depth_mae_mse(depth_pred=first_frame_right_est_depth,
                              depth_gt=first_frame_right_gt_depth)

        output_depth_meter_dict['center_view'] = dict()
        output_depth_meter_dict['center_view']['left'] = dict()
        output_depth_meter_dict['center_view']['left']['mae'] = cl_mae.data.item()
        output_depth_meter_dict['center_view']['left']['mse'] = cl_mse.data.item()

        output_depth_meter_dict['center_view']['right'] = dict()
        output_depth_meter_dict['center_view']['right']['mae'] = cr_mae.data.item()
        output_depth_meter_dict['center_view']['right']['mse'] = cr_mse.data.item()
        

        output_depth_meter_dict['last_view'] = dict()
        output_depth_meter_dict['last_view']['left'] = dict()
        output_depth_meter_dict['last_view']['left']['mae'] = ll_mae.data.item()
        output_depth_meter_dict['last_view']['left']['mse'] = ll_mse.data.item()

        output_depth_meter_dict['last_view']['right'] = dict()
        output_depth_meter_dict['last_view']['right']['mae'] = lr_mae.data.item()
        output_depth_meter_dict['last_view']['right']['mse'] = lr_mse.data.item()


        output_depth_meter_dict['first_view'] = dict()
        output_depth_meter_dict['first_view']['left'] = dict()
        output_depth_meter_dict['first_view']['left']['mae'] = fl_mae.data.item()
        output_depth_meter_dict['first_view']['left']['mse'] = fl_mse.data.item()

        output_depth_meter_dict['first_view']['right'] = dict()
        output_depth_meter_dict['first_view']['right']['mae'] = fr_mae.data.item()
        output_depth_meter_dict['first_view']['right']['mse'] = fr_mse.data.item()

        
        # get the MAE and the MSE of the input view (sterep)
        input_depth_meter_dict = dict()
        input_depth_estimation = batch_data_for_eval['predicted_input_depth'] #torch.Size([1, 2, 224, 832])
        input_gt_depth = batch_data_for_eval['input_gt_sparse_gt'] #torch.Size([1, 2, 224, 832])
        
        input_depth_estimation_left = input_depth_estimation[:,0,:,:]
        input_depth_estimation_right = input_depth_estimation[:,1,:,:]
        
        input_gt_depth_sparse_left = input_gt_depth[:,0,:,:]
        input_gt_depth_sparse_right = input_gt_depth[:,1,:,:]
        
        
        input_l_mae,input_l_mse =  compute_depth_mae_mse(depth_pred=input_depth_estimation_left,
                              depth_gt=input_gt_depth_sparse_left)
        
        input_r_mae, input_r_mse = compute_depth_mae_mse(depth_pred=input_depth_estimation_right,
                              depth_gt=input_gt_depth_sparse_right)
        
        
        input_depth_meter_dict['input_depth'] = dict()
        input_depth_meter_dict['input_depth']['left'] = dict()
        input_depth_meter_dict['input_depth']['left']['mae'] = input_l_mae.data.item()
        input_depth_meter_dict['input_depth']['left']['mse'] = input_l_mse.data.item()
        
        input_depth_meter_dict['input_depth']['right'] = dict()
        input_depth_meter_dict['input_depth']['right']['mae'] = input_r_mae.data.item()
        input_depth_meter_dict['input_depth']['right']['mse'] = input_r_mse.data.item()
        
        
        
        # saved into images.
        os.makedirs(saved_dir,exist_ok=True)

        if cfg.validation_vis_progress:
            saved_bin_token_name = batch_data_for_eval["bin_token_name"]

            # saved the output rendered images and the GT Images
            saved_folder_for_visualization = os.path.join(saved_dir,saved_bin_token_name)
            os.makedirs(saved_folder_for_visualization,exist_ok=True)
            
            center_left_vis = torch.cat([center_frame_left_est,center_frame_left_gt],dim=-2)
            center_right_vis = torch.cat([center_frame_right_est,center_frame_right_gt],dim=-2)
            center_view = torch.cat([center_left_vis,center_right_vis],dim=-1)
            skimage.io.imsave(os.path.join(saved_folder_for_visualization,'center.png'),(center_view.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            

            first_left_vis = torch.cat([first_frame_left_est,first_frame_left_gt],dim=-2)
            first_right_vis = torch.cat([first_frame_right_est,first_frame_right_gt],dim=-2)
            first_view = torch.cat([first_left_vis,first_right_vis],dim=-1)
            skimage.io.imsave(os.path.join(saved_folder_for_visualization,'first.png'),(first_view.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            
            last_left_vis = torch.cat([last_frame_left_est,last_frame_left_gt],dim=-2)
            last_right_vis = torch.cat([last_frame_right_est,last_frame_right_gt],dim=-2)
            last_view = torch.cat([last_left_vis,last_right_vis],dim=-1)
            skimage.io.imsave(os.path.join(saved_folder_for_visualization,'last.png'),(last_view.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            
            # saved  the output rendered depths and the GT Sparse depth    
            center_frame_left_depth_vis = torch.cat([center_frame_left_est_depth,center_frame_left_gt_depth],dim=-2)
            center_frame_right_depth_vis = torch.cat([center_frame_right_est_depth,center_frame_right_gt_depth],dim=-2)
            center_depth_vis = torch.cat([center_frame_left_depth_vis,center_frame_right_depth_vis],dim=-1)
            center_depth_vis = center_depth_vis.squeeze(0).cpu().numpy()
            center_depth_vis = convert_depth_to_disp(depth=center_depth_vis)
            skimage.io.imsave(os.path.join(saved_folder_for_visualization,"center_depth.png"),center_depth_vis)
            
            
            first_frame_left_depth_vis = torch.cat([first_frame_left_est_depth,first_frame_left_gt_depth],dim=-2)
            first_frame_right_depth_vis = torch.cat([first_frame_right_est_depth,first_frame_right_gt_depth],dim=-2)
            first_depth_vis = torch.cat([first_frame_left_depth_vis,first_frame_right_depth_vis],dim=-1)
            first_depth_vis = first_depth_vis.squeeze(0).cpu().numpy()
            first_depth_vis = convert_depth_to_disp(depth=first_depth_vis)
            skimage.io.imsave(os.path.join(saved_folder_for_visualization,"first_depth.png"),first_depth_vis)
            
            
            last_frame_left_depth_vis = torch.cat([last_frame_left_est_depth,last_frame_left_gt_depth],dim=-2)
            last_frame_right_depth_vis = torch.cat([last_frame_right_est_depth,last_frame_right_gt_depth],dim=-2)
            last_depth_vis = torch.cat([last_frame_left_depth_vis,last_frame_right_depth_vis],dim=-1)
            last_depth_vis = last_depth_vis.squeeze(0).cpu().numpy()
            last_depth_vis = convert_depth_to_disp(depth=last_depth_vis)
            skimage.io.imsave(os.path.join(saved_folder_for_visualization,"last_depth.png"),last_depth_vis)


            # saved the input images,estimated depths and the GT Sparse Depth        
            input_depth_estimation_vis = torch.cat([input_depth_estimation_left,input_depth_estimation_right],dim=-2)
            input_depth_estimation_vis = input_depth_estimation_vis.squeeze(0).cpu().numpy()
            input_depth_estimation_vis = convert_depth_to_disp(depth=input_depth_estimation_vis)
            skimage.io.imsave(os.path.join(saved_folder_for_visualization,"input_depth.png"),input_depth_estimation_vis)
        

        return output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict
        

    def validation_step_with_token_names(self, batch, val_result_savedir,cfg=None):
        
        bin_token_name = batch['bin_token'][0][:-4]

          # loss and loss terms
        with torch.no_grad():
            loss, loss_terms,rendered_color,\
                rendered_depth,rendered_alpha,estimated_raw_gaussains_dict,\
                    predicted_input_depth,input_sparse_gt_depth,\
                        output_rgb,sparse_depth_gt,input_images = self.forward(batch,mode='val',
                                                            cfg=cfg)
        
        batch_data_for_eval = {
        "output_gt_rgb": output_rgb,
        "output_gt_sparse_depth": sparse_depth_gt,
        "input_images": input_images,
        "input_gt_sparse_gt": input_sparse_gt_depth,
        "predicted_input_depth": predicted_input_depth,
        "rendered_rgb": rendered_color,
        "rendered_depth": rendered_depth,
        "rendered_alpha": rendered_alpha,
        "estimated_raw_gs": estimated_raw_gaussains_dict,
        "bin_token_name": bin_token_name
        }
        
        # saved into the val_result_dir: the visualiation results
        
        # rendered RGBs
        # rendered Depths
        # GT RGBs
        # GT Depths
        # Estimated Depths
        
        output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict,rendered_images,rendered_depth,gt_images,gt_depths = self.save_val_results_with_token_names(batch_data_for_eval,val_result_savedir,cfg=cfg)
        
        return output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict,rendered_images,rendered_depth,gt_images,gt_depths

    def save_val_results_with_token_names(self,batch_data_for_eval,saved_dir,cfg):
        
        '''input batch data for evaluation'''
        
        output_rgb_meter_dict = dict()
        # get the psnr and ssim for the output view
        output_rendered_rgb = batch_data_for_eval['rendered_rgb'] #torch.Size([1, 6, 3, 224, 832])
        output_gt_rgb = batch_data_for_eval['output_gt_rgb'] #torch.Size([1, 6, 3, 224, 832])
        
        # rendered center
        center_frame_left_est =  output_rendered_rgb[:,0,:,:,:]
        center_frame_right_est = output_rendered_rgb[:,2,:,:,:]
        last_frame_left_est =  output_rendered_rgb[:,1,:,:,:]
        last_frame_right_est =  output_rendered_rgb[:,3,:,:,:]
        first_frame_left_est = output_rendered_rgb[:,4,:,:,:]
        first_frame_right_est = output_rendered_rgb[:,5,:,:,:]
        
        center_frame_left_gt =  output_gt_rgb[:,0,:,:,:]
        center_frame_right_gt = output_gt_rgb[:,2,:,:,:]
        last_frame_left_gt =  output_gt_rgb[:,1,:,:,:]
        last_frame_right_gt =  output_gt_rgb[:,3,:,:,:]
        first_frame_left_gt = output_gt_rgb[:,4,:,:,:]
        first_frame_right_gt = output_gt_rgb[:,5,:,:,:]
        
        
        # print(first_frame_left_est.shape)
        # print(first_frame_left_gt.shape)
        
        
        # skimage.io.imsave("/data1/zliu/feedforward_outputs/DepthSplat/Temp_Visualization/Debugs/left_est.png",(first_frame_left_est.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        # skimage.io.imsave("/data1/zliu/feedforward_outputs/DepthSplat/Temp_Visualization/Debugs/left_gt.png",(first_frame_left_gt.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        
   
        
        
        cl_psnr,cl_ssim = compute_psnr_ssim(pred=center_frame_left_est,target=center_frame_left_gt)
        cr_psnr,cr_ssim = compute_psnr_ssim(pred=center_frame_right_est,target=center_frame_right_gt)
        ll_psnr,ll_ssim = compute_psnr_ssim(pred=last_frame_left_est,target=last_frame_left_gt)
        lr_psnr,lr_ssim = compute_psnr_ssim(pred=last_frame_right_est,target=last_frame_right_gt)
        fl_psnr,fl_ssim = compute_psnr_ssim(pred=first_frame_left_est,target=first_frame_left_gt)
        fr_psnr,fr_ssim = compute_psnr_ssim(pred=first_frame_right_est,target=first_frame_right_gt)
        
        output_rgb_meter_dict['center_view'] = dict()
        output_rgb_meter_dict['center_view']['left'] = dict()
        output_rgb_meter_dict['center_view']['left']['psnr'] = cl_psnr.data.item()
        output_rgb_meter_dict['center_view']['left']['ssim'] = cl_ssim.data.item()

        output_rgb_meter_dict['center_view']['right'] = dict()
        output_rgb_meter_dict['center_view']['right']['psnr'] = cr_psnr.data.item()
        output_rgb_meter_dict['center_view']['right']['ssim'] = cr_ssim.data.item()
        

        output_rgb_meter_dict['last_view'] = dict()
        output_rgb_meter_dict['last_view']['left'] = dict()
        output_rgb_meter_dict['last_view']['left']['psnr'] = ll_psnr.data.item()
        output_rgb_meter_dict['last_view']['left']['ssim'] = ll_ssim.data.item()

        output_rgb_meter_dict['last_view']['right'] = dict()
        output_rgb_meter_dict['last_view']['right']['psnr'] = lr_psnr.data.item()
        output_rgb_meter_dict['last_view']['right']['ssim'] = lr_ssim.data.item()


        output_rgb_meter_dict['first_view'] = dict()
        output_rgb_meter_dict['first_view']['left'] = dict()
        output_rgb_meter_dict['first_view']['left']['psnr'] = fl_psnr.data.item()
        output_rgb_meter_dict['first_view']['left']['ssim'] = fl_ssim.data.item()

        output_rgb_meter_dict['first_view']['right'] = dict()
        output_rgb_meter_dict['first_view']['right']['psnr'] = fr_psnr.data.item()
        output_rgb_meter_dict['first_view']['right']['ssim'] = fr_ssim.data.item()

        
        # get the MAE and the MSE of the output view
        output_depth_meter_dict = dict()
        output_rendered_depth = batch_data_for_eval['rendered_depth'] #torch.Size([1, 6, 3, 224, 832])
        output_gt_depth = batch_data_for_eval['output_gt_sparse_depth'] #torch.Size([1, 6, 3, 224, 832])
        

        center_frame_left_est_depth =  output_rendered_depth[:,0,:,:]
        center_frame_right_est_depth = output_rendered_depth[:,2,:,:]
        last_frame_left_est_depth =  output_rendered_depth[:,1,:,:]
        last_frame_right_est_depth =  output_rendered_depth[:,3,:,:]
        first_frame_left_est_depth = output_rendered_depth[:,4,:,:]
        first_frame_right_est_depth = output_rendered_depth[:,5,:,:]

        center_frame_left_gt_depth =  output_gt_depth[:,0,:,:]
        center_frame_right_gt_depth = output_gt_depth[:,2,:,:]
        last_frame_left_gt_depth =  output_gt_depth[:,1,:,:]
        last_frame_right_gt_depth =  output_gt_depth[:,3,:,:]
        first_frame_left_gt_depth = output_gt_depth[:,4,:,:]
        first_frame_right_gt_depth = output_gt_depth[:,5,:,:]

        cl_mae,cl_mse = compute_depth_mae_mse(depth_pred=center_frame_left_est_depth,
                              depth_gt=center_frame_left_gt_depth)
        
        cr_mae,cr_mse = compute_depth_mae_mse(depth_pred=center_frame_right_est_depth,
                              depth_gt=center_frame_right_gt_depth)
        
        ll_mae,ll_mse = compute_depth_mae_mse(depth_pred=last_frame_left_est_depth,
                              depth_gt=last_frame_left_gt_depth)
        
        lr_mae,lr_mse = compute_depth_mae_mse(depth_pred=last_frame_right_est_depth,
                              depth_gt=last_frame_right_gt_depth)
        
        fl_mae,fl_mse = compute_depth_mae_mse(depth_pred=first_frame_left_est_depth,
                              depth_gt=first_frame_left_gt_depth)

        fr_mae,fr_mse = compute_depth_mae_mse(depth_pred=first_frame_right_est_depth,
                              depth_gt=first_frame_right_gt_depth)

        output_depth_meter_dict['center_view'] = dict()
        output_depth_meter_dict['center_view']['left'] = dict()
        output_depth_meter_dict['center_view']['left']['mae'] = cl_mae.data.item()
        output_depth_meter_dict['center_view']['left']['mse'] = cl_mse.data.item()

        output_depth_meter_dict['center_view']['right'] = dict()
        output_depth_meter_dict['center_view']['right']['mae'] = cr_mae.data.item()
        output_depth_meter_dict['center_view']['right']['mse'] = cr_mse.data.item()
        

        output_depth_meter_dict['last_view'] = dict()
        output_depth_meter_dict['last_view']['left'] = dict()
        output_depth_meter_dict['last_view']['left']['mae'] = ll_mae.data.item()
        output_depth_meter_dict['last_view']['left']['mse'] = ll_mse.data.item()

        output_depth_meter_dict['last_view']['right'] = dict()
        output_depth_meter_dict['last_view']['right']['mae'] = lr_mae.data.item()
        output_depth_meter_dict['last_view']['right']['mse'] = lr_mse.data.item()


        output_depth_meter_dict['first_view'] = dict()
        output_depth_meter_dict['first_view']['left'] = dict()
        output_depth_meter_dict['first_view']['left']['mae'] = fl_mae.data.item()
        output_depth_meter_dict['first_view']['left']['mse'] = fl_mse.data.item()

        output_depth_meter_dict['first_view']['right'] = dict()
        output_depth_meter_dict['first_view']['right']['mae'] = fr_mae.data.item()
        output_depth_meter_dict['first_view']['right']['mse'] = fr_mse.data.item()

        
        # get the MAE and the MSE of the input view (sterep)
        input_depth_meter_dict = dict()
        input_depth_estimation = batch_data_for_eval['predicted_input_depth'] #torch.Size([1, 2, 224, 832])
        input_gt_depth = batch_data_for_eval['input_gt_sparse_gt'] #torch.Size([1, 2, 224, 832])
        
        input_depth_estimation_left = input_depth_estimation[:,0,:,:]
        input_depth_estimation_right = input_depth_estimation[:,1,:,:]
        
        input_gt_depth_sparse_left = input_gt_depth[:,0,:,:]
        input_gt_depth_sparse_right = input_gt_depth[:,1,:,:]
        
        
        input_l_mae,input_l_mse =  compute_depth_mae_mse(depth_pred=input_depth_estimation_left,
                              depth_gt=input_gt_depth_sparse_left)
        
        input_r_mae, input_r_mse = compute_depth_mae_mse(depth_pred=input_depth_estimation_right,
                              depth_gt=input_gt_depth_sparse_right)
        
        
        input_depth_meter_dict['input_depth'] = dict()
        input_depth_meter_dict['input_depth']['left'] = dict()
        input_depth_meter_dict['input_depth']['left']['mae'] = input_l_mae.data.item()
        input_depth_meter_dict['input_depth']['left']['mse'] = input_l_mse.data.item()
        
        input_depth_meter_dict['input_depth']['right'] = dict()
        input_depth_meter_dict['input_depth']['right']['mae'] = input_r_mae.data.item()
        input_depth_meter_dict['input_depth']['right']['mse'] = input_r_mse.data.item()
        
        
        
        # # saved into images.
        # os.makedirs(saved_dir,exist_ok=True)
        
        rendered_images = [first_frame_left_est,first_frame_right_est,
                           center_frame_left_est,center_frame_right_est,
                           last_frame_left_est,last_frame_right_est]
        
        gt_images = [first_frame_left_gt,first_frame_right_gt,
                           center_frame_left_gt,center_frame_right_gt,
                           last_frame_left_gt,last_frame_right_gt
        ]
        
        rendered_depth = [
            first_frame_left_est_depth,first_frame_right_est_depth,
            center_frame_left_est_depth,center_frame_right_est_depth,
            last_frame_left_est_depth,last_frame_right_est_depth
        ]
        
        gt_depths = [first_frame_left_gt_depth,first_frame_right_gt_depth,
                    center_frame_left_gt_depth,center_frame_right_gt_depth,
                    last_frame_left_gt_depth,last_frame_right_gt_depth
        ]
        
        

        # if cfg.validation_vis_progress:
        #     saved_bin_token_name = batch_data_for_eval["bin_token_name"]

        #     # saved the output rendered images and the GT Images
        #     saved_folder_for_visualization = os.path.join(saved_dir,saved_bin_token_name)
        #     os.makedirs(saved_folder_for_visualization,exist_ok=True)
            
        #     center_left_vis = torch.cat([center_frame_left_est,center_frame_left_gt],dim=-2)
        #     center_right_vis = torch.cat([center_frame_right_est,center_frame_right_gt],dim=-2)
        #     center_view = torch.cat([center_left_vis,center_right_vis],dim=-1)
        #     skimage.io.imsave(os.path.join(saved_folder_for_visualization,'center.png'),(center_view.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            

        #     first_left_vis = torch.cat([first_frame_left_est,first_frame_left_gt],dim=-2)
        #     first_right_vis = torch.cat([first_frame_right_est,first_frame_right_gt],dim=-2)
        #     first_view = torch.cat([first_left_vis,first_right_vis],dim=-1)
        #     skimage.io.imsave(os.path.join(saved_folder_for_visualization,'first.png'),(first_view.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            
        #     last_left_vis = torch.cat([last_frame_left_est,last_frame_left_gt],dim=-2)
        #     last_right_vis = torch.cat([last_frame_right_est,last_frame_right_gt],dim=-2)
        #     last_view = torch.cat([last_left_vis,last_right_vis],dim=-1)
        #     skimage.io.imsave(os.path.join(saved_folder_for_visualization,'last.png'),(last_view.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            
        #     # saved  the output rendered depths and the GT Sparse depth    
        #     center_frame_left_depth_vis = torch.cat([center_frame_left_est_depth,center_frame_left_gt_depth],dim=-2)
        #     center_frame_right_depth_vis = torch.cat([center_frame_right_est_depth,center_frame_right_gt_depth],dim=-2)
        #     center_depth_vis = torch.cat([center_frame_left_depth_vis,center_frame_right_depth_vis],dim=-1)
        #     center_depth_vis = center_depth_vis.squeeze(0).cpu().numpy()
        #     center_depth_vis = convert_depth_to_disp(depth=center_depth_vis)
        #     skimage.io.imsave(os.path.join(saved_folder_for_visualization,"center_depth.png"),center_depth_vis)
            
            
        #     first_frame_left_depth_vis = torch.cat([first_frame_left_est_depth,first_frame_left_gt_depth],dim=-2)
        #     first_frame_right_depth_vis = torch.cat([first_frame_right_est_depth,first_frame_right_gt_depth],dim=-2)
        #     first_depth_vis = torch.cat([first_frame_left_depth_vis,first_frame_right_depth_vis],dim=-1)
        #     first_depth_vis = first_depth_vis.squeeze(0).cpu().numpy()
        #     first_depth_vis = convert_depth_to_disp(depth=first_depth_vis)
        #     skimage.io.imsave(os.path.join(saved_folder_for_visualization,"first_depth.png"),first_depth_vis)
            
            
        #     last_frame_left_depth_vis = torch.cat([last_frame_left_est_depth,last_frame_left_gt_depth],dim=-2)
        #     last_frame_right_depth_vis = torch.cat([last_frame_right_est_depth,last_frame_right_gt_depth],dim=-2)
        #     last_depth_vis = torch.cat([last_frame_left_depth_vis,last_frame_right_depth_vis],dim=-1)
        #     last_depth_vis = last_depth_vis.squeeze(0).cpu().numpy()
        #     last_depth_vis = convert_depth_to_disp(depth=last_depth_vis)
        #     skimage.io.imsave(os.path.join(saved_folder_for_visualization,"last_depth.png"),last_depth_vis)


        #     # saved the input images,estimated depths and the GT Sparse Depth        
        #     input_depth_estimation_vis = torch.cat([input_depth_estimation_left,input_depth_estimation_right],dim=-2)
        #     input_depth_estimation_vis = input_depth_estimation_vis.squeeze(0).cpu().numpy()
        #     input_depth_estimation_vis = convert_depth_to_disp(depth=input_depth_estimation_vis)
        #     skimage.io.imsave(os.path.join(saved_folder_for_visualization,"input_depth.png"),input_depth_estimation_vis)
        

        return output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict,rendered_images,rendered_depth,gt_images,gt_depths


    def prepare_input_multiviews(self,batch):
        
        input_image_index_selection = [0,3]
        
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
        
   
        input_batch_dict['imgs'] = input_batch_dict['imgs'][:,input_image_index_selection,:,:,:]
        input_batch_dict['intrinsics'] = input_batch_dict['intrinsics'][:,input_image_index_selection,:,:]
        input_batch_dict['extrinsics'] = input_batch_dict['extrinsics'][:,input_image_index_selection,:,:]
        # input_batch_dict['nn_matrix'] = input_batch_dict['nn_matrix'][:,input_image_index_selection,:]
        input_batch_dict['pseudo_depths'] = input_batch_dict['pseudo_depths'][:,input_image_index_selection,:,:]
        input_batch_dict['sparse_depths'] = input_batch_dict['sparse_depths'][:,input_image_index_selection,:,:]
        
 
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


    # rendered video for KITTI-360
    def forward_video_kitti360(self, batch,cfg=None):
        bin_token_name = batch['bin_token']
        
        # perform the Feed-Forward 3DGS Estimator
        input_batch_dict,output_batch_dict = self.prepare_input_multiviews(batch=batch)
        
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
        
        # debug here
        # intrinsics[:,:,0,2] = intrinsics[:,:,0,2] + 13
        # intrinsics[:,:,1,2] = intrinsics[:,:,1,2] - 30
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
        
        predicted_input_depth = results_dict['depth_preds'][0]
        

        if cfg.train_depth_only:
            pass
        
        
        else:
            # estimated the gs 
            # change the head here
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
        
        
        # rendered for new views
        # last 
        c2w_lf_left = output_batch_dict["output_c2ws"][:, -4]
        c2w_lf_right = output_batch_dict["output_c2ws"][:, -3]
        # first
        c2w_ff_left = output_batch_dict["output_c2ws"][:, -2]
        c2w_ff_right = output_batch_dict["output_c2ws"][:, -1]
        # center
        c2w_cf_left = output_batch_dict["output_c2ws"][:, -6] #(1,2,4,4)
        c2w_cf_right = output_batch_dict["output_c2ws"][:, -5] #(1,2,4,4)
        
        ''' 
            Movement 0:  Center Left Rotation 45 Degree
            Movement 1:  Center Rotation Back
        '''
    
        # left backward 3---------rotation 45 ------rotation back 
        theta = -math.pi / 4  # 
        rot_0 = torch.tensor([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0,                0,               1]
        ], dtype=torch.float32).to(c2w_cf_left.device)

        c2w_cf_left_rot2_right = c2w_cf_left.clone()
        c2w_cf_left_rot2_right[...,:3,:3] = rot_0@c2w_cf_left_rot2_right[...,:3,:3]

        c2w_lf_left_rot2_right = c2w_lf_left.clone()
        c2w_lf_left_rot2_right[...,:3,:3] = rot_0 @ c2w_lf_left_rot2_right[...,:3,:3]
        
        c2w_ff_left_rot2_right = c2w_ff_left.clone()
        c2w_ff_left_rot2_right[...,:3,:3] = rot_0 @ c2w_ff_left_rot2_right[...,:3,:3]

        ''' 
            Movement 2:  Center Left Cam to Center Right
            Movement 3:  Center Right Rot Inside
            Movement 4: Rotation Back
        '''
        
        # right backward 3---------rotation 45 ------rotation back:  short +1
        theta = math.pi / 4  # 
        rot_1 = torch.tensor([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0,                0,               1]
        ], dtype=torch.float32).to(c2w_cf_left.device)
        
        c2w_cf_right_rot2_left = c2w_cf_right.clone()
        c2w_cf_right_rot2_left[...,:3,:3] = rot_1 @ c2w_cf_right_rot2_left[...,:3,:3]
        
        c2w_lf_right_rot2_left = c2w_lf_right.clone()
        c2w_lf_right_rot2_left[...,:3,:3] = rot_1 @ c2w_lf_right_rot2_left[...,:3,:3]
        
        c2w_ff_right_rot2_left = c2w_ff_right.clone() 
        c2w_ff_right_rot2_left[...,:3,:3] = rot_1 @ c2w_ff_right_rot2_left[...,:3,:3]
        
        
        ''' Movement 5: from right to left '''
        '''Movement 6: From Center left to Last Left'''
        num_frames_short = 60
        num_frames_long = 60
        
        t_short = torch.linspace(0, 1, num_frames_short, dtype=torch.float32, device=self.device)
        t_long = torch.linspace(0, 1 - 1 / (num_frames_long + 1), num_frames_long, dtype=torch.float32, device=self.device)
        # center left rot
        movement_0 = interpolate_extrinsics(c2w_cf_left,c2w_cf_left_rot2_right,t_short)
        # center left rot back
        movement_1 = interpolate_extrinsics(c2w_cf_left_rot2_right,c2w_cf_left,t_short)
        # center left to right
        movement_2 = interpolate_extrinsics(c2w_cf_left,c2w_cf_right,t_short)
        # center right rot
        movement_3 = interpolate_extrinsics(c2w_cf_right,c2w_cf_right_rot2_left,t_short)
        # center right rot back
        movement_4 = interpolate_extrinsics(c2w_cf_right_rot2_left,c2w_cf_right,t_short)
        # center right to left
        movement_5 = interpolate_extrinsics(c2w_cf_right,c2w_cf_left,t_short)
        # center left to last left
        movement_6 = interpolate_extrinsics(c2w_cf_left,c2w_lf_left,t_short)
        # last left to rot
        movement_7 = interpolate_extrinsics(c2w_lf_left,c2w_lf_left_rot2_right,t_short)
        # last left rot back
        movement_8 = interpolate_extrinsics(c2w_lf_left_rot2_right,c2w_lf_left,t_short)
        # last left to last right
        movement_9 = interpolate_extrinsics(c2w_lf_left,c2w_lf_right,t_short)
        
        # last right rot
        movement_10 = interpolate_extrinsics(c2w_lf_right,c2w_lf_right_rot2_left,t_short)
        movement_11 = interpolate_extrinsics(c2w_lf_right_rot2_left, c2w_lf_right,t_short)
        movement_12 = interpolate_extrinsics(c2w_lf_right,c2w_lf_left ,t_short)
        movement_13 = interpolate_extrinsics(c2w_lf_left,c2w_ff_left,t_short)
        movement_14 = interpolate_extrinsics(c2w_ff_left,c2w_ff_left_rot2_right,t_short)
        movement_15 = interpolate_extrinsics(c2w_ff_left_rot2_right,c2w_ff_left,t_short)
        movement_16 = interpolate_extrinsics(c2w_ff_left,c2w_ff_right,t_short)
        movement_17 = interpolate_extrinsics(c2w_ff_right,c2w_ff_right_rot2_left,t_short)
        movement_18 = interpolate_extrinsics(c2w_ff_right_rot2_left,c2w_ff_right,t_short)
        
        c2w_interp = torch.cat([movement_0, movement_1, movement_2,
                                movement_3, movement_4,movement_5,
                                movement_6,movement_7,
                                movement_8,movement_9,
                                movement_10,movement_11,
                                movement_12,movement_13,
                                movement_14,movement_15,
                                movement_16,movement_17,
                                movement_18
                                ], dim=1)
        
        
        N_Chunks = 10
        interval = int(c2w_interp.shape[1]//N_Chunks)
        
        rendered_rgb_list = []
        rendered_depth_list = []

        for idx in tqdm(range(N_Chunks)):
            
            current_c2w_interp = c2w_interp[:,idx*interval:(idx+1)*interval,:]
            
            output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,current_c2w_interp.shape[1],1,1)
            z_near_batch = torch.from_numpy(np.array([cfg.near])).unsqueeze(0).repeat(current_c2w_interp.shape[0],current_c2w_interp.shape[1]).type_as(current_c2w_interp)
            z_far_batch = torch.from_numpy(np.array([cfg.far])).unsqueeze(0).repeat(current_c2w_interp.shape[0],current_c2w_interp.shape[1]).type_as(current_c2w_interp)

            rendered_results = self.decoder_branch(gaussians=estimated_raw_gaussains_dict["gaussians"],
                                                extrinsics= current_c2w_interp,
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
            
            rendered_color = torch.clamp(rendered_color,min=0,max=1.0)
            rendered_depth = torch.clamp(rendered_depth,min=0,max=150)

            rendered_rgb_list.append(rendered_color)
            rendered_depth_list.append(rendered_depth)
            

        rendered_rgb_final = torch.cat(rendered_rgb_list,dim=1)
        rendered_depth_final = torch.cat(rendered_depth_list,dim=1)
        
        preds = {"img":rendered_rgb_final,"depth":rendered_depth_final}
        
        return preds,bin_token_name
    

    
    def validation_complete_with_bin_tokens(self, batch, val_result_savedir,bin_token_list,cfg=None,vis=False):
        
        
        bin_token_name = bin_token_list[0][:-4]

        input_batch_dict,output_batch_dict = self.prepare_input_multiviews(batch=batch)
        
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
        
        # debug here
        # intrinsics[:,:,0,2] = intrinsics[:,:,0,2] + 13
        # intrinsics[:,:,1,2] = intrinsics[:,:,1,2] - 30
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
        
        predicted_input_depth = results_dict['depth_preds'][0]
        
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
        
        rendered_color = torch.clamp(rendered_color,min=0,max=1.0)
        rendered_depth = torch.clamp(rendered_depth,min=0,max=150)
        
        
        rendered_images_fusion = rendered_color #(1,6,3,H,W)
        rendered_depth_fusion = rendered_depth  #(1,V,H，W)
        rendered_images_gt = output_batch_dict["output_imgs"]
        sparse_depth_gt = output_batch_dict["output_sparse_depth"]
        
        # # change the ordered.
        rendered_images_fusion = interleave_left_right(rendered_images_fusion)
        rendered_depth_fusion = interleave_left_right_depth(rendered_depth_fusion)
        sparse_depth_gt = interleave_left_right_depth(sparse_depth_gt)
        rendered_images_gt = interleave_left_right(rendered_images_gt)
        
        # first view
        rendered_images_first_stereo = rendered_images_fusion[:,-2:,:,:,:]
        gt_images_first_stereo = rendered_images_gt[:,-2:,:,:,:]
        renderded_depth_first_stereo = rendered_depth_fusion[:,-2:,:,:]
        gt_depth_first_stereo = sparse_depth_gt[:,-2:,:,:]
        
        # last view
        rendered_images_last_stereo = rendered_images_fusion[:,-4:-2,:,:,:]
        gt_images_last_stereo = rendered_images_gt[:,-4:-2,:,:,:]
        renderded_depth_last_stereo = rendered_depth_fusion[:,-4:-2,:,:]
        gt_depth_last_stereo = sparse_depth_gt[:,-4:-2,:,:]
        
        # center view
        rendered_images_center_stereo = rendered_images_fusion[:,-6:-4,:,:,:]
        gt_images_center_stereo = rendered_images_gt[:,-6:-4,:,:,:]
        renderded_depth_center_stereo = rendered_depth_fusion[:,-6:-4,:,:]
        gt_depth_center_stereo = sparse_depth_gt[:,-6:-4,:,:]
        # all view
        rendered_images_all_stereo = rendered_images_fusion
        gt_images_all_stereo =  rendered_images_gt
        renderded_depth_all_stereo = rendered_depth_fusion
        gt_depth_all_stereo = sparse_depth_gt
        
        # RGB Evaluation
        first_psnr_left,first_ssim_left,first_psnr_right,first_ssim_right = compute_stereo_psnr_ssim(pred=rendered_images_first_stereo,target=gt_images_first_stereo)
        last_psnr_left,last_ssim_left,last_psnr_right,last_ssim_right = compute_stereo_psnr_ssim(pred=rendered_images_last_stereo,target=gt_images_last_stereo)
        center_psnr_left,center_ssim_left,center_psnr_right,center_ssim_right = compute_stereo_psnr_ssim(pred=rendered_images_center_stereo,target=gt_images_center_stereo)
        all_psnr_left,all_ssim_left,all_psnr_right,all_ssim_right = compute_all_stereo_psnr_ssim(pred=rendered_images_all_stereo,target=gt_images_all_stereo)
        

        
        evaluation_rgb_results_stat = {
  
            'first_view_psnr_left':first_psnr_left.data.item(),
            'first_view_ssim_left':first_ssim_left.data.item(),
            'first_view_psnr_right':first_psnr_right.data.item(),
            'first_view_ssim_right':first_ssim_right.data.item(),
            
            'last_view_psnr_left':last_psnr_left.data.item(),
            'last_view_ssim_left':last_ssim_left.data.item(),
            'last_view_psnr_right':last_psnr_right.data.item(),
            'last_view_ssim_right':last_ssim_right.data.item(),
            
            'center_view_psnr_left':center_psnr_left.data.item(),
            'center_view_ssim_left':center_ssim_left.data.item(),
            'center_view_psnr_right':center_psnr_right.data.item(),
            'center_view_ssim_right':center_ssim_right.data.item(),
            
            'all_view_psnr_left':all_psnr_left.data.item(),
            'all_view_ssim_left':all_ssim_left.data.item(),
            'all_view_psnr_right':all_psnr_right.data.item(),
            'all_view_ssim_right':all_ssim_right.data.item(),
        }
        
        # Depth Evaluation
        first_left_mae,first_left_mse,first_right_mae,first_right_mse = compute_depth_stereo_mae_mse(depth_pred=renderded_depth_first_stereo,depth_gt=gt_depth_first_stereo)
        last_left_mae,last_left_mse,last_right_mae,last_right_mse = compute_depth_stereo_mae_mse(depth_pred=renderded_depth_last_stereo,depth_gt=gt_depth_last_stereo)
        center_left_mae,center_left_mse,center_right_mae,center_right_mse = compute_depth_stereo_mae_mse(depth_pred=renderded_depth_center_stereo,depth_gt=gt_depth_center_stereo)
        all_left_mae,all_left_mse,all_right_mae,all_right_mse = compute_depth_stereo_mae_mse(depth_pred=renderded_depth_all_stereo,depth_gt=gt_depth_all_stereo)
        
        evaluation_depth_results_stat = {
            'first_view_left_mae':first_left_mae.data.item(),
            'first_view_left_mse':first_left_mse.data.item(),
            'first_view_right_mae':first_right_mae.data.item(),
            'first_view_right_mse':first_right_mse.data.item(),
            'last_view_left_mae':last_left_mae.data.item(),
            'last_view_left_mse':last_left_mse.data.item(),
            'last_view_right_mae':last_right_mae.data.item(),
            'last_view_right_mse':last_right_mse.data.item(),
            'center_view_left_mae':center_left_mae.data.item(),
            'center_view_left_mse':center_left_mse.data.item(),
            'center_view_right_mae':center_right_mae.data.item(),
            'center_view_right_mse':center_right_mse.data.item(),
            'all_view_left_mae':all_left_mae.data.item(),
            'all_view_left_mse':all_left_mse.data.item(),
            'all_view_right_mae':all_right_mae.data.item(),
            'all_view_right_mse':all_right_mse.data.item(),
        }


        evaluation_results_stat = {
            "RGB":evaluation_rgb_results_stat,
            "Depth":evaluation_depth_results_stat,
        }

        if vis:
            saved_folder_for_visualization = os.path.join(val_result_savedir,bin_token_name)
            os.makedirs(saved_folder_for_visualization,exist_ok=True)
            
            rendered_images_folder_path = os.path.join(saved_folder_for_visualization,'rendered_images')
            rendered_depth_folder_path = os.path.join(saved_folder_for_visualization,'rendered_depth')
            
            GT_images_folder_path = os.path.join(saved_folder_for_visualization,'GT Images')
            GT_depth_folder_path = os.path.join(saved_folder_for_visualization,'GT Depth')
            
            Rendered_Depth_Error_Folder_Path = os.path.join(saved_folder_for_visualization,"Rendered_Depth_Error")
            
            os.makedirs(rendered_images_folder_path,exist_ok=True)
            os.makedirs(rendered_depth_folder_path,exist_ok=True)
            os.makedirs(GT_images_folder_path,exist_ok=True)
            os.makedirs(GT_depth_folder_path,exist_ok=True)
            os.makedirs(Rendered_Depth_Error_Folder_Path,exist_ok=True)
            
            rendered_first_stereo = torch.cat((rendered_images_first_stereo[:,0,:,:,:],rendered_images_first_stereo[:,1,:,:,:]),dim=-1)
            skimage.io.imsave(os.path.join(rendered_images_folder_path,'first_stereo.png'),(rendered_first_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            rendered_last_stereo = torch.cat((rendered_images_last_stereo[:,0,:,:,:],rendered_images_last_stereo[:,1,:,:,:]),dim=-1)
            skimage.io.imsave(os.path.join(rendered_images_folder_path,'last_stereo.png'),(rendered_last_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            rendered_center_stereo = torch.cat((rendered_images_center_stereo[:,0,:,:,:],rendered_images_center_stereo[:,1,:,:,:]),dim=-1)
            skimage.io.imsave(os.path.join(rendered_images_folder_path,'center_stereo.png'),(rendered_center_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

            gt_first_stereo = torch.cat((gt_images_first_stereo[:,0,:,:,:],gt_images_first_stereo[:,1,:,:,:]),dim=-1)
            skimage.io.imsave(os.path.join(GT_images_folder_path,'first_stereo.png'),(gt_first_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            gt_last_stereo = torch.cat((gt_images_last_stereo[:,0,:,:,:],gt_images_last_stereo[:,1,:,:,:]),dim=-1)
            skimage.io.imsave(os.path.join(GT_images_folder_path,'last_stereo.png'),(gt_last_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            gt_center_stereo = torch.cat((gt_images_center_stereo[:,0,:,:,:],gt_images_center_stereo[:,1,:,:,:]),dim=-1)
            skimage.io.imsave(os.path.join(GT_images_folder_path,'center_stereo.png'),(gt_center_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            
            # renderded_depth_first_stereo
            rendered_depth_first_stereo =torch.cat((renderded_depth_first_stereo[:,0,:,:],renderded_depth_first_stereo[:,1,:,:]),dim=-1)
            rendered_depth_first_stereo_vis = rendered_depth_first_stereo.squeeze(0).cpu().numpy()
            rendered_depth_first_stereo_vis = convert_depth_to_disp(depth=rendered_depth_first_stereo_vis)
            skimage.io.imsave(os.path.join(rendered_depth_folder_path,'first_stereo_depth.png'),rendered_depth_first_stereo_vis)
            rendered_depth_last_stereo =torch.cat((renderded_depth_last_stereo[:,0,:,:],renderded_depth_last_stereo[:,1,:,:]),dim=-1)
            rendered_depth_last_stereo_vis = rendered_depth_last_stereo.squeeze(0).cpu().numpy()
            rendered_depth_last_stereo_vis = convert_depth_to_disp(depth=rendered_depth_last_stereo_vis)
            skimage.io.imsave(os.path.join(rendered_depth_folder_path,'last_stereo_depth.png'),rendered_depth_last_stereo_vis)
            rendered_depth_center_stereo =torch.cat((renderded_depth_center_stereo[:,0,:,:],renderded_depth_center_stereo[:,1,:,:]),dim=-1)
            rendered_depth_center_stereo_vis = rendered_depth_center_stereo.squeeze(0).cpu().numpy()
            rendered_depth_center_stereo_vis = convert_depth_to_disp(depth=rendered_depth_center_stereo_vis)
            skimage.io.imsave(os.path.join(rendered_depth_folder_path,'center_stereo_depth.png'),rendered_depth_center_stereo_vis)
            

            # # gt_depth_first_stereo
            gt_depth_first_stereo = torch.cat((gt_depth_first_stereo[:,0,:,:],gt_depth_first_stereo[:,1,:,:]),dim=-1)
            gt_depth_first_stereo_vis = gt_depth_first_stereo.squeeze(0).cpu().numpy()
            gt_depth_first_stereo_vis = convert_depth_to_disp(depth=gt_depth_first_stereo_vis)
            skimage.io.imsave(os.path.join(GT_depth_folder_path,'first_stereo_depth.png'),gt_depth_first_stereo_vis)
            gt_depth_last_stereo = torch.cat((gt_depth_last_stereo[:,0,:,:],gt_depth_last_stereo[:,1,:,:]),dim=-1)
            gt_depth_last_stereo_vis = gt_depth_last_stereo.squeeze(0).cpu().numpy()
            gt_depth_last_stereo_vis = convert_depth_to_disp(depth=gt_depth_last_stereo_vis)
            skimage.io.imsave(os.path.join(GT_depth_folder_path,'last_stereo_depth.png'),gt_depth_last_stereo_vis)
            gt_depth_center_stereo = torch.cat((gt_depth_center_stereo[:,0,:,:],gt_depth_center_stereo[:,1,:,:]),dim=-1)
            gt_depth_center_stereo_vis = gt_depth_center_stereo.squeeze(0).cpu().numpy()
            gt_depth_center_stereo_vis = convert_depth_to_disp(depth=gt_depth_center_stereo_vis)
            skimage.io.imsave(os.path.join(GT_depth_folder_path,'center_stereo_depth.png'),gt_depth_center_stereo_vis)

            # rendered depth error map
            disp_error_img_first_stereo = disp_error_img(D_est_tensor=rendered_depth_first_stereo,D_gt_tensor=gt_depth_first_stereo)
            disp_error_img_first_stereo_vis = (disp_error_img_first_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            skimage.io.imsave(os.path.join(Rendered_Depth_Error_Folder_Path,'first_stereo_depth_error.png'),disp_error_img_first_stereo_vis)
            disp_error_img_last_stereo = disp_error_img(D_est_tensor=rendered_depth_last_stereo,D_gt_tensor=gt_depth_last_stereo)
            disp_error_img_last_stereo_vis = (disp_error_img_last_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            skimage.io.imsave(os.path.join(Rendered_Depth_Error_Folder_Path,'last_stereo_depth_error.png'),disp_error_img_last_stereo_vis)
            disp_error_img_center_stereo = disp_error_img(D_est_tensor=rendered_depth_center_stereo,D_gt_tensor=gt_depth_center_stereo)
            disp_error_img_center_stereo_vis = (disp_error_img_center_stereo.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            skimage.io.imsave(os.path.join(Rendered_Depth_Error_Folder_Path,'center_stereo_depth_error.png'),disp_error_img_center_stereo_vis)
            
            
            # rendered videos
            saved_videos_path = os.path.join(saved_folder_for_visualization,'videos')
            os.makedirs(saved_videos_path,exist_ok=True)
            
            preds, saved_video_name = self.forward_video_kitti360(batch=batch,cfg=cfg)
            
            bs = preds["img"].shape[0]  
            pred_imgs = preds["img"] #(4,960,3,224,400)
            pred_depths = preds["depth"] #(4,960,3,224,400)
            
            
            # saved the results with batch
            for b in range(bs):
                bin_token = saved_video_name[b]
                # dump rgb view
                dump_path = osp.join(saved_videos_path, "{}_rgb.mp4".format(bin_token))
                video = (pred_imgs[b].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
                video_rec = wandb.Video(video[None], fps=30, format="mp4")
                video_tensor = video_rec._prepare_video(video_rec.data)
                clip = mpy.ImageSequenceClip(list(video_tensor), fps=30)
                clip.write_videofile(dump_path, codec='libx264', preset='medium', logger=None)
                
                # dump depth view
                dump_path_dpt = osp.join(saved_videos_path, "{}_depth.mp4".format(bin_token))
                pred_depth = pred_depths[b].clamp(0.0, 100.0)
                max_val = float(pred_depth.max())
                video_dpt = depths_to_colors(pred_depths[b], concat="frame", max_val=max_val)
                video_dpt = video_dpt.transpose((0, 3, 1, 2))
                video_rec_dpt = wandb.Video(video_dpt[None], fps=30, format="mp4")
                video_tensor_dpt = video_rec_dpt._prepare_video(video_rec_dpt.data)
                clip_dpt = mpy.ImageSequenceClip(list(video_tensor_dpt), fps=30)
                clip_dpt.write_videofile(dump_path_dpt, codec='libx264', preset='medium', logger=None)
        
        
        return evaluation_results_stat        


