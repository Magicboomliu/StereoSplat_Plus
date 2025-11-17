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
from .volume.TPVGaussainEster import VolumeGaussian
from .losses import LPIPS
# debug here
# import matplotlib.pyplot as plt

import copy 
import skimage.io
from .metrics import convert_depth_to_disp,compute_psnr_ssim,compute_stereo_psnr_ssim,compute_all_stereo_psnr_ssim
import matplotlib.pyplot as plt
import math
from .gaussian import GaussianRenderer
from .losses import Custom_Depth_Loss
from .utils.interpolation import interpolate_extrinsics
from tqdm import tqdm
from .gs_fuse import transform_g2_to_g1
#from .utilsdir.gaussain_fusion import fuse_gaussians_by_voxel_with_depth_batched_vectorized,fuse_gaussians_by_voxel_with_depth_scatter_batched

from .depth_error_vis import disp_error_img,depths_to_colors
import moviepy.editor as mpy
import wandb
from PIL import Image
import time
import math
import skimage.io


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

def get_pointmap_from_depth(depth, intrinsics, c2w):
    """
    depth:      [B, V, H, W]
    intrinsics: [B, V, 3, 3]
    c2w:        [B, V, 4, 4]
    return:     pointmap [B, V, H, W, 3]
    """
    B, V, H, W = depth.shape

    # 创建归一化像素网格 [H, W, 3]
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=depth.device),
        torch.arange(W, dtype=torch.float32, device=depth.device),
        indexing='ij'
    )
    xy1 = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H, W, 3]
    xy1 = xy1[None, None, ...].expand(B, V, H, W, 3)        # [B, V, H, W, 3]

    # 反投影：pixel → camera coordinates
    K_inv = torch.inverse(intrinsics)                      # [B, V, 3, 3]
    K_inv = K_inv[:, :, None, None, :, :]                  # [B, V, 1, 1, 3, 3]
    cam_dirs = torch.matmul(K_inv, xy1.unsqueeze(-1))      # [B, V, H, W, 3, 1]
    cam_dirs = cam_dirs.squeeze(-1)                        # [B, V, H, W, 3]

    # 深度 * 单位方向 = 相机坐标点
    cam_points = cam_dirs * depth.unsqueeze(-1)            # [B, V, H, W, 3]

    # 相机 → 世界坐标
    R = c2w[:, :, :3, :3]                                  # [B, V, 3, 3]
    T = c2w[:, :, :3, 3]                                   # [B, V, 3]
    R = R[:, :, None, None, :, :]                          # [B, V, 1, 1, 3, 3]
    T = T[:, :, None, None, :]                             # [B, V, 1, 1, 3]

    world_points = torch.matmul(R, cam_points.unsqueeze(-1)).squeeze(-1) + T  # [B, V, H, W, 3]
    return world_points

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

def add_local_pitch(c2w: torch.Tensor, deg: float):
    """绕相机自身X轴(右轴)旋转deg度，低头用负角度"""
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    Rloc = torch.tensor([[1, 0, 0, 0],
                         [0, c,-s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]], dtype=c2w.dtype, device=c2w.device)
    return c2w @ Rloc  # 后乘=局部旋转

def add_local_yaw_about_camZ(c2w: torch.Tensor, deg: float):
    """绕相机自身Z轴(前轴)旋转deg度"""
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    Rloc = torch.tensor([[ c,-s, 0, 0],
                         [ s, c, 0, 0],
                         [ 0, 0, 1, 0],
                         [ 0, 0, 0, 1]], dtype=c2w.dtype, device=c2w.device)
    return c2w @ Rloc  # 后乘=局部旋转



class VolumeFusionRevision_CV_Branch_Only(BaseModule):
    
    def __init__(self,
                 backbone=None, # feature extraction
                 neck=None,      # feature aggregation
                 costvolume_gs=None,
                 camera_args=None, # camera/3D Range
                 dataset_params=None, # dataset params
                 losses_params=None,
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
        
        # gaussain renderers
        self.renderer = GaussianRenderer(self.device, **camera_args)
        
        
        self.losses_params = losses_params
        
        if self.losses_params is not None:
            # preception loss here
            self.perceptual_loss = LPIPS().eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
        
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

    @torch.no_grad()
    def k_nearest_camera_indices(
                                self,
                                extrinsics: torch.Tensor,
                                K: int,
                                pose_type: str = "cam2world",
                                include_self: bool = False) -> torch.Tensor:
        """
        Returns:
            indices: [B, V, K]  最近K个相机的索引；当 include_self=True 时，包含自身
        """
        assert extrinsics.ndim == 4 and extrinsics.size(-1) == 4 and extrinsics.size(-2) == 4
        B, V = extrinsics.shape[:2]
        if include_self:
            assert 0 < K <= V
        else:
            assert 0 < K < V

        # 相机中心 C: (B, V, 3)
        if pose_type.lower() in ("cam2world", "c2w"):
            centers = extrinsics[..., :3, 3]
        elif pose_type.lower() in ("world2cam", "w2c"):
            R = extrinsics[..., :3, :3]
            t = extrinsics[..., :3, 3]
            centers = (-R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError("pose_type must be 'cam2world' or 'world2cam'")

        dist = torch.cdist(centers, centers, p=2)  # (B, V, V)

        if not include_self:
            dist.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

        indices = torch.topk(dist, k=K, dim=-1, largest=False).indices  # (B, V, K)
        return indices.long()


    def prepare_input_multiview(self,
                                batch,
                                view_num=2,
                                matching_nums=2):
        
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
        
        
        stereo_pairs_nums = view_num//2
        
        if stereo_pairs_nums == 1:
            index = [0,3]
        elif stereo_pairs_nums == 2:
            index = [0,3,1,4]
        elif stereo_pairs_nums == 3:
            index = [0,3,1,4,2,5]
        else:
            raise ValueError("stereo_pairs_nums must be 1, 2, or 3")
        

        
        input_rgb = input_rgb[:,index,:,:,:]
        

        input_camera_intrinsics = input_camera_intrinsics[:,index,:,:]
        input_camera_extrinsics = input_camera_extrinsics[:,index,:,:]
        input_psuedo_depth = input_psuedo_depth[:,index,:,:]
        input_sparse_depth = input_sparse_depth[:,index,:,:]

            
        selected_cam_dist_index = self.k_nearest_camera_indices(extrinsics=input_camera_extrinsics,
                                                                K=matching_nums,
                                                                pose_type="cam2world",
                                                                include_self=True)
        
        # input_dict
        input_batch_dict['imgs'] = input_rgb.to(device_id, dtype=self.dtype)
        input_batch_dict['intrinsics'] = input_camera_intrinsics.to(device_id, dtype=self.dtype)
        input_batch_dict['extrinsics'] = input_camera_extrinsics.to(device_id, dtype=self.dtype)
        input_batch_dict['nn_matrix'] = selected_cam_dist_index.to(device_id, dtype=self.dtype)
        input_batch_dict['pseudo_depths'] = input_psuedo_depth.to(device_id, dtype=self.dtype)
        input_batch_dict['sparse_depths'] = input_sparse_depth.to(device_id, dtype=self.dtype)
        input_batch_dict['bin_token_name'] = bin_token_name
        

        
        current_w2i = copy.deepcopy(batch['inputs_vol']['w2i'])
        current_w2i = current_w2i[:,index,:,:]
        
        current_input_rgbs = copy.deepcopy(batch["inputs"]["rgb"])
        current_input_rgbs = current_input_rgbs[:,index,:,:,:]
        
        # for volume-gs
        img_metas = []
        bs, v, c, h, w = current_input_rgbs.shape
        for w2i in current_w2i:
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


    def prepare_data_complete(self,batch):
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
        
        
        output_list = []
        
        for ind in range(len(batch["outputs"]["rgb"])):
            # for render and loss and eval
            output_dict = dict()
            output_dict["output_imgs"] = batch["outputs"]["rgb"][ind].to(device_id, dtype=self.dtype)
            output_dict["output_depths"] = batch["outputs"]["depth"][ind].to(device_id, dtype=self.dtype)
            output_dict["output_depths_m"] = batch["outputs"]["depth_m"][ind].to(device_id, dtype=self.dtype)
            output_dict["output_confs_m"] = batch["outputs"]["conf_m"][ind].to(device_id, dtype=self.dtype)
            output_dict["output_positions"] = (batch["outputs"]["rays_o"][ind] + batch["outputs"]["rays_d"][ind] * \
                                batch["outputs"]["depth_m"][ind].unsqueeze(-1)).to(device_id, dtype=self.dtype)
            output_dict["output_rays_o"] = batch["outputs"]["rays_o"][ind].to(device_id, dtype=self.dtype)
            output_dict["output_rays_d"] = batch["outputs"]["rays_d"][ind].to(device_id, dtype=self.dtype)
            output_dict["output_c2ws"] = batch["outputs"]["c2w"][ind].to(device_id, dtype=self.dtype)
            output_dict["output_fovxs"] = batch["outputs"]["fovx"][ind].to(device_id, dtype=self.dtype)
            output_dict["output_fovys"] = batch["outputs"]["fovy"][ind].to(device_id, dtype=self.dtype)
            output_dict['output_sparse_gt_depth'] = batch['outputs']['sparse_gt_depth'][ind].to(device_id, dtype=self.dtype)
            output_list.append(output_dict)
            
            
        output_batch_dict['output_list'] = output_list

        
        return input_batch_dict,output_batch_dict


    def forward(self,batch,
                mode='train',
                view_num=4,
                matching_nums=3,            
                iter=0,
                cfg=None):
        # get inpout_batch_dict
        
        # get the input and the reference input
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=view_num,
                                                                         matching_nums=matching_nums)
        
        
        img =input_batch_dict["imgs"] #[B,6,3,H,W]
        
        
        height,width = img.shape[-2:]
        bs = img.shape[0]
        
        input_pseudo_depth = input_batch_dict['pseudo_depths']
        input_sparse_gt_depth = input_batch_dict['sparse_depths']
        img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
                
        # perform the cost volume-based 
        gaussians_cv,gaussians_feat,pred_depths = self.costvolume_gs(input_batch_dict,cfg=cfg,
                                                          images_feat=img_feats[0])
        

        
        # Make Sure the estimate gaussains are valid
        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
  
        gaussians_all = gaussians_cv
        bs = gaussians_all.shape[0] # batch size is 2
        
        # doing rendering here
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"] # [B,6*3]
        render_fovys = output_batch_dict["output_fovys"] # [B,6*3]
        

        # return a dicts: rendered images and rendered alphs and rendered depth
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        
        rendered_color_fuse = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_depth_fuse = rendered_results_fuse['depth'] # torch.Size([1, V, 1, 224, 832])
        rendered_alpha_fuse = rendered_results_fuse['alpha'] # torch.Size([1, V, 1, 224, 832])
        rendered_depth_fuse = rendered_depth_fuse.squeeze(2)
        rendered_alpha_fuse = rendered_alpha_fuse.squeeze(2)
        
        rendered_color_fuse = torch.clamp(rendered_color_fuse,min=0,max=1.0)
        rendered_depth_fuse = torch.clamp(rendered_depth_fuse,min=0,max=150)
        

        loss = 0.0
        loss_terms = {}
        def set_loss(key, split, loss_value, loss_weight=1.0):
            loss_terms[f"{split}/loss_{key}"] = loss_value.item()
            loss_terms[f"{split}/loss_{key}_w"] = loss_value.item() * loss_weight

        # GT Information For Supervision
        output_rgb = output_batch_dict['output_imgs']
        rgb_gt = output_rgb
        pseudo_depth_gt = output_batch_dict['output_depths_m']
        sparse_depth_gt = output_batch_dict['output_sparse_depth']
        valid_mask_01 = sparse_depth_gt>0
        valid_mask_01_float = valid_mask_01.float()
        
        # use this
        fusion_pseudo_with_sparse_gt = valid_mask_01_float * sparse_depth_gt + (1-valid_mask_01_float) * pseudo_depth_gt 

        # Loss
        if mode=='train' or mode=='val':
        
            pred_depth = pred_depths
            ## RGB Loss Here
            fusion_branch_loss = 0.0
            depth_estimation_branch_loss = 0.0
            
            
            # ==================== Depth Estimation Loss =====================
            if self.losses_params.depth_estimation:
                if self.losses_params.gt_depth_type=='sparse':
                    
                    valid_mask_01 = input_sparse_gt_depth>0
                    valid_mask_02 = input_sparse_gt_depth<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.bool()
                    sparse_depth_estimation_loss = F.l1_loss(input_sparse_gt_depth[valid_mask],pred_depth[valid_mask])            
                    depth_estimation_loss = sparse_depth_estimation_loss
                    
                elif self.losses_params.gt_depth_type=='pseudo':
                    
                    valid_mask_01 = input_pseudo_depth>0
                    valid_mask_02 = input_pseudo_depth<120
                    valid_mask = valid_mask_01 * valid_mask_02
                    valid_mask = valid_mask.float()
                    pseudo_depth_estimation_loss =Custom_Depth_Loss(depth_pred=pred_depth*valid_mask, 
                                                              depth_gt=input_pseudo_depth*valid_mask)
                    # pseudo_depth_estimation_loss = F.l1_loss(input_pseudo_depth[valid_mask],pred_depth[valid_mask])            
                    depth_estimation_loss = pseudo_depth_estimation_loss
                    
                elif self.losses_params.gt_depth_type =='sparse_pseudo':
                    
                    valid_mask_01 = input_sparse_gt_depth>0
                    valid_mask_01_float = valid_mask_01.float()
                    
                    input_pseudo_gt_fusion_depth = input_sparse_gt_depth * valid_mask_01_float \
                                + (1-valid_mask_01_float)*input_pseudo_depth
                    valid_mask_02 = input_pseudo_gt_fusion_depth>0
                    valid_mask_03 = input_pseudo_gt_fusion_depth<150
                    valid_mask = valid_mask_02 * valid_mask_03
                    valid_mask = valid_mask.float()
                    


                    pseudo_depth_estimation_loss =Custom_Depth_Loss(depth_pred=pred_depth*valid_mask, 
                                                              depth_gt=input_pseudo_gt_fusion_depth*valid_mask)
                    
                    # pseudo_depth_estimation_loss = F.l1_loss(input_pseudo_gt_fusion_depth[valid_mask],pred_depth[valid_mask])            
                    depth_estimation_loss = pseudo_depth_estimation_loss
                
                depth_estimation_branch_loss +=depth_estimation_loss
                
            else:
                depth_estimation_branch_loss = depth_estimation_branch_loss * 0.0
            
            set_loss("depth_est_loss", mode, depth_estimation_branch_loss, 
                     self.losses_params.depth_est_sup_dict.branch_weight)
            
            # ==================== RGB Loss Here =====================
            rec_loss = (rgb_gt - render_pkg_fuse["image"]) ** 2
            fusion_branch_loss = loss + (rec_loss.mean() * self.losses_params.fusion_sup_dict.weight_recon)
            set_loss("recon_fusion", mode, rec_loss.mean(), self.losses_params.fusion_sup_dict.weight_recon)              
                


            # ==================== Preception Loss Here ================
            current_height, current_width = rgb_gt.shape[-2:]
            preception_loss_fuse = self.perceptual_loss(rgb_gt.reshape(-1,3,current_height,current_width),
                                                render_pkg_fuse["image"].reshape(-1,3,current_height,current_width))
            fusion_branch_loss = fusion_branch_loss + (preception_loss_fuse.mean() \
                                * self.losses_params.fusion_sup_dict.weight_perceptual)
            
            set_loss("perceptual_fusion", mode, preception_loss_fuse.mean(), 
                        self.losses_params.fusion_sup_dict.weight_perceptual)
                

             # ==================== Rendered Depth Loss ================                
            if self.losses_params.gt_depth_type=='sparse':
                valid_mask_01 = sparse_depth_gt>0
                valid_mask_02 = sparse_depth_gt<120
                valid_mask = valid_mask_01 * valid_mask_02
                valid_mask = valid_mask.bool()
                rendered_gt_depth = sparse_depth_gt * valid_mask.float()
                
            elif self.losses_params.gt_depth_type=='pseudo':
                valid_mask_01 = pseudo_depth_gt>0
                valid_mask_02 = pseudo_depth_gt<120
                valid_mask = valid_mask_01 * valid_mask_02
                valid_mask = valid_mask.bool()
                rendered_gt_depth = pseudo_depth_gt * valid_mask.float()
                
                
            elif self.losses_params.gt_depth_type =='sparse_pseudo':
                valid_mask_01 = sparse_depth_gt>0
                valid_mask_01_float = valid_mask_01.float()
                fusion_pseudo_with_sparse_gt = valid_mask_01_float * sparse_depth_gt + (1-valid_mask_01_float) * pseudo_depth_gt 
                
                valid_mask_02 = fusion_pseudo_with_sparse_gt >0
                valid_mask_03 = fusion_pseudo_with_sparse_gt < 150
                valid_mask = valid_mask_02 * valid_mask_03
                valid_mask = valid_mask.bool()
                
                rendered_gt_depth = fusion_pseudo_with_sparse_gt  * valid_mask.float()
                
            else:
                raise NotImplementedError
                
            depth_abs_loss = torch.abs(render_pkg_fuse["depth"].squeeze(2) - rendered_gt_depth)
            depth_abs_loss = depth_abs_loss.mean()
            fusion_branch_loss = fusion_branch_loss + self.losses_params.fusion_sup_dict.weight_depth_abs * depth_abs_loss
            set_loss("depth_abs_fusion", mode, depth_abs_loss, self.losses_params.fusion_sup_dict.weight_depth_abs)

            
            loss = fusion_branch_loss * self.losses_params.fusion_sup_dict.branch_weight + \
                                depth_estimation_branch_loss * self.losses_params.depth_est_sup_dict.branch_weight
            
            
            rendered_fusion_list = [rendered_color_fuse,rendered_depth_fuse]

            
            if mode=='train':
                return loss, loss_terms,rendered_fusion_list
            
            elif mode=='val':
                return loss, loss_terms,rendered_fusion_list,pred_depth,input_sparse_gt_depth,output_rgb,sparse_depth_gt,img
            
        elif mode=='test':
            return rendered_fusion_list

        else:
            raise NotImplementedError

    
    def validation_step(self, batch, val_result_savedir,cfg=None):
        
        bin_token_name = batch['bin_token'][0][:-4]

        # loss and loss terms
        with torch.no_grad():
            loss, loss_terms,rendered_fusion_list,predicted_input_depth,input_sparse_gt_depth,\
                        output_rgb,sparse_depth_gt,input_images = self.forward(batch,mode='val',
                                                            cfg=cfg)

        batch_data_for_eval = {
            "output_gt_rgb": output_rgb,
            "output_gt_sparse_depth": sparse_depth_gt,
            "input_images": input_images,
            "input_gt_sparse_gt": input_sparse_gt_depth,
            "predicted_input_depth": predicted_input_depth,
            "rendered_fusion": rendered_fusion_list,
            "bin_token_name": bin_token_name
        }
            
        output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict = self.save_val_results(batch_data_for_eval,val_result_savedir,cfg=cfg)
        
        return output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict
        
    def save_val_results(self,batch_data_for_eval,saved_dir,cfg):
        
        '''input batch data for evaluation'''
        
        rendered_fusion = batch_data_for_eval['rendered_fusion']
        rendered_list = [rendered_fusion]
        names = ['fusion']
        
        metrics_rendered_rgb_list = []
        metrics_rendered_depth_list = []
        metrics_estimated_depth_list = []
        
        
        for idx, rendered_output in enumerate(rendered_list):
            
            
            current_name = names[idx]
            output_rgb_meter_dict = dict()
            # get the psnr and ssim for the output view
            output_rendered_rgb = rendered_output[0] #torch.Size([1, 6, 3, 224, 832])
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
            output_rendered_depth = rendered_output[1] #torch.Size([1, 6, 3, 224, 832])
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
            

            metrics_rendered_rgb_list.append(output_rgb_meter_dict)
            metrics_rendered_depth_list.append(output_depth_meter_dict)
            metrics_estimated_depth_list.append(input_depth_meter_dict)
            
            
            # saved into images.
            os.makedirs(saved_dir,exist_ok=True)

            if cfg.validation_vis_progress:
                saved_bin_token_name = batch_data_for_eval["bin_token_name"]

                # saved the output rendered images and the GT Images
                saved_folder_for_visualization = os.path.join(saved_dir,saved_bin_token_name,current_name)
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
            

        return metrics_rendered_rgb_list,metrics_rendered_depth_list,metrics_estimated_depth_list

    def validation_on_the_forward_views(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        view_num=2,
                                        matching_nums=2,
                                        cfg=None,
                                        vis=False,
                                        ):
        
        bin_token_name = bin_token_list[0][:-4]
        
        # loss and loss terms
        with torch.no_grad():
            loss, loss_terms,rendered_fusion_list,\
                    predicted_input_depth,input_sparse_gt_depth,\
                        output_rgb,sparse_depth_gt,input_images = self.forward(batch,mode='val',
                                                            view_num=view_num,
                                                            matching_nums=matching_nums,
                                                            cfg=cfg)


        rendered_images_fusion = rendered_fusion_list[0] #(1,6,3,H,W)
        rendered_depth_fusion = rendered_fusion_list[1] #(1,V,H，W)
        rendered_images_gt = output_rgb
        sparse_depth_gt = sparse_depth_gt
        
        
        # change the ordered.
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
            
            preds, saved_video_name = self.forward_kitti360_videos(batch=batch,cfg=cfg,view_num=view_num,matching_nums=matching_nums)
            
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

    def forward_kitti360_videos(self,
                                batch,
                                cfg,
                                view_num=4,
                                matching_nums=3):

        # get the input and the reference input
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,
                                                                         view_num=view_num,
                                                                         matching_nums=matching_nums)
        
        
        img =input_batch_dict["imgs"] #[B,6,3,H,W]
        height,width = img.shape[-2:]
        bs = img.shape[0]
        
        input_pseudo_depth = input_batch_dict['pseudo_depths']
        input_sparse_gt_depth = input_batch_dict['sparse_depths']
        img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
                
        # perform the cost volume-based 
        gaussians_cv,gaussians_feat,pred_depths = self.costvolume_gs(input_batch_dict,cfg=cfg,
                                                          images_feat=img_feats[0])

        # Make Sure the estimate gaussains are valid
        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
        gaussians_all = gaussians_cv
        bs = gaussians_all.shape[0] # batch size is 2
        
        
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        render_c2w = interleave_left_right_pose(render_c2w)
        
        c2w_ff_left = render_c2w[:,-2,:,:]
        c2w_ff_right = render_c2w[:,-1,:,:]
        c2w_cf_left = render_c2w[:,-6,:,:]
        c2w_cf_right = render_c2w[:,-5,:,:]
        c2w_lf_left = render_c2w[:,-4,:,:]
        c2w_lf_right = render_c2w[:,-3,:,:]
        
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

        num_frames_all = 60 * c2w_interp.shape[1]
        fovxs_interp =output_batch_dict["output_fovxs"][:, -2:-1].repeat(1, num_frames_all)
        fovys_interp =output_batch_dict["output_fovys"][:, -2:-1].repeat(1, num_frames_all)
        
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=c2w_interp,
            fovx=fovxs_interp,
            fovy=fovys_interp,
            rays_o=None,
            rays_d=None
        )
        
        output_imgs = render_pkg_fuse["image"] # b v 3 h w
        output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w
        preds = {"img": output_imgs, "depth": output_depths}


        return preds, input_batch_dict['bin_token_name']
        

    def get_additional_bev_novel_views(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        view_num=2,
                                        matching_nums=2,
                                        cfg=None,
                                        vis=False):
      
        # get the input and the reference input
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=view_num,
                                                                         matching_nums=matching_nums)
            
        bin_token_name = input_batch_dict['bin_token_name'][0][:-4]    
        img =input_batch_dict["imgs"] #[B,6,3,H,W]
        
        
        height,width = img.shape[-2:]
        bs = img.shape[0]
        
        input_pseudo_depth = input_batch_dict['pseudo_depths']
        input_sparse_gt_depth = input_batch_dict['sparse_depths']
        img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
                
        # perform the cost volume-based 
        gaussians_cv,gaussians_feat,pred_depths = self.costvolume_gs(input_batch_dict,cfg=cfg,
                                                          images_feat=img_feats[0])
        

        
        # Make Sure the estimate gaussains are valid
        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
  
        gaussians_all = gaussians_cv
        bs = gaussians_all.shape[0] # batch size is 2
        
        # doing rendering here
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"] # [B,6*3]
        render_fovys = output_batch_dict["output_fovys"] # [B,6*3]
        
        
        
        num_of_views_all = render_c2w.shape[1]
        
        rest_of_views_all = num_of_views_all - 2
        rest_besides_first_c2w = render_c2w[:,:-2,:,:]
        
        half_rest_of_views_all = rest_of_views_all // 2
        center_view_c2w_left_index = half_rest_of_views_all -2
        last_view_c2w_left_index = half_rest_of_views_all -1
        
        center_view_c2w_right_index = rest_of_views_all - 2
        last_view_c2w_right_index = rest_of_views_all - 1
        
    
        first_view_c2w_left = render_c2w[:,-2:-1,:,:]
        first_view_c2w_right = render_c2w[:,-1:,:,:]
        
        center_view_c2w_left = rest_besides_first_c2w[:,center_view_c2w_left_index,:,:].unsqueeze(1)
        last_view_c2w_left = rest_besides_first_c2w[:,last_view_c2w_left_index,:,:].unsqueeze(1)
        center_view_c2w_right = rest_besides_first_c2w[:,center_view_c2w_right_index,:,:].unsqueeze(1)
        last_view_c2w_right = rest_besides_first_c2w[:,last_view_c2w_right_index,:,:].unsqueeze(1)
        

        
        # novel view 1 
        rendered_c2w_center_bev_view1 = copy.deepcopy(center_view_c2w_left)
        rendered_c2w_center_bev_view1[0][0][2,3] = rendered_c2w_center_bev_view1[0][0][2,3] + 3
        rendered_c2w_center_bev_view1[0][0] = add_local_pitch(rendered_c2w_center_bev_view1[0][0], deg=-45.0)
        
        # novel view 2
        rendered_c2w_center_bev_view2 = copy.deepcopy(center_view_c2w_left)
        rendered_c2w_center_bev_view2[0][0][2,3] = rendered_c2w_center_bev_view2[0][0][2,3] + 3
        rendered_c2w_center_bev_view2[0][0] = add_local_pitch(rendered_c2w_center_bev_view2[0][0], deg=-30.0)
        

        # novel view 3
        rendered_c2w_center_bev_view3 = copy.deepcopy(center_view_c2w_right)
        rendered_c2w_center_bev_view3[0][0][2,3] = rendered_c2w_center_bev_view3[0][0][2,3] + 3
        rendered_c2w_center_bev_view3[0][0] = add_local_pitch(rendered_c2w_center_bev_view3[0][0], deg=-45.0)
        
        # novel view 4
        rendered_c2w_center_bev_view4 = copy.deepcopy(center_view_c2w_right)
        rendered_c2w_center_bev_view4[0][0][2,3] = rendered_c2w_center_bev_view4[0][0][2,3] + 3
        rendered_c2w_center_bev_view4[0][0] = add_local_pitch(rendered_c2w_center_bev_view4[0][0], deg=-30.0)
        
        
        t_short = torch.linspace(0, 1, 31, dtype=torch.float32, device=self.device)
        center_view_center_c2w = interpolate_extrinsics(center_view_c2w_left,
                                            center_view_c2w_right,
                                            t_short)[:,:,15,:,:]
        
        # novel view 5
        rendered_c2w_center_bev_view5 = copy.deepcopy(center_view_center_c2w)
        rendered_c2w_center_bev_view5[0][0][2,3] = rendered_c2w_center_bev_view5[0][0][2,3] + 3
        rendered_c2w_center_bev_view5[0][0] = add_local_pitch(rendered_c2w_center_bev_view5[0][0], deg=-30.0)

        
        # center_left_plus_3_d45, center_left_plus_3_d30, center_right_plus_3_d45, center_right_plus_3_d30, center_plus_3_d30
        rendered_bev_novel_views_c2w = torch.cat([rendered_c2w_center_bev_view1,rendered_c2w_center_bev_view2,
                                                  rendered_c2w_center_bev_view3,rendered_c2w_center_bev_view4,
                                                  rendered_c2w_center_bev_view5],dim=1)
        
        
        rendered_bev_fovxs = render_fovxs[:,0:1].repeat(1,rendered_bev_novel_views_c2w.shape[1])
        rendered_bev_fovys = render_fovys[:,0:1].repeat(1,rendered_bev_novel_views_c2w.shape[1])
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=rendered_bev_novel_views_c2w,
            fovx=rendered_bev_fovxs,
            fovy=rendered_bev_fovys,
            rays_o=None,
            rays_d=None
        )

        rendered_results_fuse = render_pkg_fuse

        rendered_color_fuse = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_depth_fuse = rendered_results_fuse['depth'] # torch.Size([1, V, 1, 224, 832])
        rendered_alpha_fuse = rendered_results_fuse['alpha'] # torch.Size([1, V, 1, 224, 832])
        rendered_depth_fuse = rendered_depth_fuse.squeeze(2)
        rendered_alpha_fuse = rendered_alpha_fuse.squeeze(2)
        
        rendered_color_fuse = torch.clamp(rendered_color_fuse,min=0,max=1.0)
        rendered_depth_fuse = torch.clamp(rendered_depth_fuse,min=0,max=150)
        
        
        




        if vis:
            saved_folder_for_visualization = os.path.join(val_result_savedir,bin_token_name)
            os.makedirs(saved_folder_for_visualization,exist_ok=True)
            
            rendered_images_folder_path = os.path.join(saved_folder_for_visualization,'rendered_images')
            rendered_depth_folder_path = os.path.join(saved_folder_for_visualization,'rendered_depth')
            
            os.makedirs(rendered_images_folder_path,exist_ok=True)
            os.makedirs(rendered_depth_folder_path,exist_ok=True)
            
            
            # center_left_plus_3_d45, center_left_plus_3_d30, center_right_plus_3_d45, center_right_plus_3_d30, center_plus_3_d30
            
            
            center_left_plus_3_d45 = rendered_color_fuse[:,0,:,:,:]
            center_left_plus_3_d30 = rendered_color_fuse[:,1,:,:,:]
            center_right_plus_3_d45 = rendered_color_fuse[:,2,:,:,:]
            center_right_plus_3_d30 = rendered_color_fuse[:,3,:,:,:]
            center_plus_3_d30 = rendered_color_fuse[:,4,:,:,:]

            skimage.io.imsave(os.path.join(rendered_images_folder_path,'center_left_plus_3_d45.png'),(center_left_plus_3_d45.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            skimage.io.imsave(os.path.join(rendered_images_folder_path,'center_left_plus_3_d30.png'),(center_left_plus_3_d30.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            skimage.io.imsave(os.path.join(rendered_images_folder_path,'center_right_plus_3_d45.png'),(center_right_plus_3_d45.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            skimage.io.imsave(os.path.join(rendered_images_folder_path,'center_right_plus_3_d30.png'),(center_right_plus_3_d30.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            skimage.io.imsave(os.path.join(rendered_images_folder_path,'center_plus_3_d30.png'),(center_plus_3_d30.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            
            rendered_depth_center_left_plus_3_d45 = rendered_depth_fuse[:,0,:,:].squeeze(0).cpu().numpy()
            rendered_depth_center_left_plus_3_d30 = rendered_depth_fuse[:,1,:,:].squeeze(0).cpu().numpy()
            rendered_depth_center_right_plus_3_d45 = rendered_depth_fuse[:,2,:,:].squeeze(0).cpu().numpy()
            rendered_depth_center_right_plus_3_d30 = rendered_depth_fuse[:,3,:,:].squeeze(0).cpu().numpy()
            rendered_depth_center_plus_3_d30 = rendered_depth_fuse[:,4,:,:].squeeze(0).cpu().numpy()
            
            rendered_depth_center_left_plus_3_d45_vis = convert_depth_to_disp(depth=rendered_depth_center_left_plus_3_d45)
            rendered_depth_center_left_plus_3_d30_vis = convert_depth_to_disp(depth=rendered_depth_center_left_plus_3_d30)
            rendered_depth_center_right_plus_3_d45_vis = convert_depth_to_disp(depth=rendered_depth_center_right_plus_3_d45)
            rendered_depth_center_right_plus_3_d30_vis = convert_depth_to_disp(depth=rendered_depth_center_right_plus_3_d30)
            rendered_depth_center_plus_3_d30_vis = convert_depth_to_disp(depth=rendered_depth_center_plus_3_d30)

            skimage.io.imsave(os.path.join(rendered_depth_folder_path,'center_left_plus_3_d45_depth.png'),rendered_depth_center_left_plus_3_d45_vis)
            skimage.io.imsave(os.path.join(rendered_depth_folder_path,'center_left_plus_3_d30_depth.png'),rendered_depth_center_left_plus_3_d30_vis)
            skimage.io.imsave(os.path.join(rendered_depth_folder_path,'center_right_plus_3_d45_depth.png'),rendered_depth_center_right_plus_3_d45_vis)
            skimage.io.imsave(os.path.join(rendered_depth_folder_path,'center_right_plus_3_d30_depth.png'),rendered_depth_center_right_plus_3_d30_vis)
            skimage.io.imsave(os.path.join(rendered_depth_folder_path,'center_plus_3_d30_depth.png'),rendered_depth_center_plus_3_d30_vis)

            

    
def get_mean(list):
    return sum(list)*1.0/len(list)


def saved_into_json(data_dict,path):
    import json
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)

if __name__=="__main__":
    pass