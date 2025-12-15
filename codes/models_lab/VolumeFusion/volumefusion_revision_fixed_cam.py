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

import skimage.io
import math



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



class VolumeFusionRevision(BaseModule):
    def __init__(self,
                 backbone=None, # feature extraction
                 neck=None,      # feature aggregation
                 costvolume_gs=None,
                 volume_gs = None,
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
        self.volume_gs = VolumeGaussian(**volume_gs)
        
        # gaussain renderers
        self.renderer = GaussianRenderer(self.device, **camera_args)
        # Loss Functions Configuration Here
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
        

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        # batch-wise saved the gaussain-pixel and the feature-pixel
        gaussians_cv_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_cv[b, :, 0] >= x_start) & (gaussians_cv[b, :, 0] <= x_end) & \
                        (gaussians_cv[b, :, 1] >= y_start) & (gaussians_cv[b, :, 1] <= y_end) & \
                        (gaussians_cv[b, :, 2] >= z_start) & (gaussians_cv[b, :, 2] <= z_end)
            # get the valid gaussains in the pixel splat
            gaussians_cv_mask_i = gaussians_cv[b][mask_pixel_i]
            # get the valid feature in the pixel splat
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_cv_mask.append(gaussians_cv_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        


        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                input_batch_dict['extrinsics'],
                gaussians_cv_mask,
                gaussians_feat_mask,
                input_batch_dict["img_metas"])
        

        
        # Make Sure the estimate gaussains are valid
        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        
        
        gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
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
        

        
        
        if mode =='train' or mode=='val':
            render_pkg_cv = self.renderer.render(
                gaussians=gaussians_cv,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )

            rendered_color_cv = render_pkg_cv['image'] # torch.Size([1, V, 3, 224, 832])
            rendered_depth_cv = render_pkg_cv['depth'] # torch.Size([1, V, 1, 224, 832])
            rendered_depth_cv = rendered_depth_cv.squeeze(2)
            rendered_color_cv = torch.clamp(rendered_color_cv,min=0,max=1.0)
            rendered_depth_cv = torch.clamp(rendered_depth_cv,min=0,max=150)


            render_pkg_volume = self.renderer.render(
                gaussians=gaussians_volume,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )

            rendered_color_volume = render_pkg_volume['image'] # torch.Size([1, V, 3, 224, 832])
            rendered_depth_volume = render_pkg_volume['depth'] # torch.Size([1, V, 1, 224, 832])
            rendered_depth_volume = rendered_depth_volume.squeeze(2)
            rendered_color_volume = torch.clamp(rendered_color_volume,min=0,max=1.0)
            rendered_depth_volume = torch.clamp(rendered_depth_volume,min=0,max=150)
            
            
        else:
            render_pkg_cv, render_pkg_volume = None, None
            rendered_color_volume = None
            rendered_color_cv = None
            
            rendered_depth_cv = None
            rendered_depth_volume = None
        
        
        # Loss Function Here

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

        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        
        
        # Loss
        if mode=='train' or mode=='val':
        
            output_intrinsics_recovered = output_intrinsics.clone()
            fusion_gt_pointmap = get_pointmap_from_depth(depth=fusion_pseudo_with_sparse_gt,
                                                         intrinsics=output_intrinsics_recovered,
                                                         c2w=render_c2w
                                                         )
            
            mask_dptm = (fusion_gt_pointmap[..., 0] >= x_start) & (fusion_gt_pointmap[..., 0] <= x_end) & \
                        (fusion_gt_pointmap[..., 1] >= y_start) & (fusion_gt_pointmap[..., 1] <= y_end) & \
                        (fusion_gt_pointmap[..., 2] >= z_start) & (fusion_gt_pointmap[..., 2] <= z_end)
            mask_dptm = mask_dptm.float()
            
            pred_depth = pred_depths
            
            ## RGB Loss Here
            cost_volume_branch_loss = 0.0
            trip_plane_branch_loss = 0.0
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
            if self.losses_params.use_fusion: # Must
                
                if self.losses_params.use_volume:
                    rec_loss_vol = (rgb_gt * mask_dptm.unsqueeze(2) - render_pkg_volume["image"] * mask_dptm.unsqueeze(2)) ** 2
                    trip_plane_branch_loss = trip_plane_branch_loss+ (rec_loss_vol.mean() * self.losses_params.volume_sup_dict.weight_recon_vol)
                    set_loss("recon_vol", mode, rec_loss_vol.mean(), self.losses_params.volume_sup_dict.weight_recon_vol)
                
                if self.losses_params.use_cv:
                    rec_loss_cv = (rgb_gt-render_pkg_cv['image'])**2
                    cost_volume_branch_loss = cost_volume_branch_loss + (rec_loss_cv.mean() * self.losses_params.cv_sup_dict.weight_recon_cv)
                    set_loss("recon_cv",mode,cost_volume_branch_loss.mean(),
                             self.losses_params.cv_sup_dict.weight_recon_cv) 
                

                rec_loss = (rgb_gt - render_pkg_fuse["image"]) ** 2
                fusion_branch_loss = loss + (rec_loss.mean() * self.losses_params.fusion_sup_dict.weight_recon)
                set_loss("recon_fusion", mode, rec_loss.mean(), self.losses_params.fusion_sup_dict.weight_recon)              
                
            else:
                raise NotImplementedError

            # ==================== Preception Loss Here ================
            if self.losses_params.use_fusion: # Must
                if self.losses_params.use_volume:
                    current_height, current_width = rgb_gt.shape[-2:]
                    preception_loss_volume = self.perceptual_loss(rgb_gt.reshape(-1,3,current_height,current_width)*mask_dptm.unsqueeze(2).reshape(-1,1,current_height,current_width),
                                                        render_pkg_volume["image"].reshape(-1,3,current_height,current_width)*mask_dptm.unsqueeze(2).reshape(-1,1,current_height,current_width)
                                                        )
                    trip_plane_branch_loss = trip_plane_branch_loss+ (preception_loss_volume.mean() \
                                        * self.losses_params.volume_sup_dict.weight_perceptual_vol)
                    set_loss("perceptual_vol", mode, preception_loss_volume.mean(), 
                                self.losses_params.volume_sup_dict.weight_perceptual_vol)
                    

                if self.losses_params.use_cv:
                    current_height, current_width = rgb_gt.shape[-2:]
                    preception_loss_cv = self.perceptual_loss(rgb_gt.reshape(-1,3,current_height,current_width),
                                                        render_pkg_cv["image"].reshape(-1,3,current_height,current_width)
                                                        )
                    cost_volume_branch_loss = cost_volume_branch_loss + (preception_loss_cv.mean() \
                                        * self.losses_params.cv_sup_dict.weight_perceptual_cv)
                    set_loss("perceptual_cv", mode, preception_loss_cv.mean(), 
                                self.losses_params.cv_sup_dict.weight_perceptual_cv)
                

                current_height, current_width = rgb_gt.shape[-2:]
                preception_loss_fuse = self.perceptual_loss(rgb_gt.reshape(-1,3,current_height,current_width),
                                                    render_pkg_fuse["image"].reshape(-1,3,current_height,current_width)
                                                    )
                fusion_branch_loss = fusion_branch_loss + (preception_loss_fuse.mean() \
                                    * self.losses_params.fusion_sup_dict.weight_perceptual)
                
                set_loss("perceptual_fusion", mode, preception_loss_fuse.mean(), 
                            self.losses_params.fusion_sup_dict.weight_perceptual)
                
            else:
                raise NotImplementedError
        
             # ==================== Rendered Depth Loss ================
            
            if self.losses_params.use_fusion: # Must
                
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
                
                
                
                if self.losses_params.use_volume:
                    
                    depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"].squeeze(2)*mask_dptm - rendered_gt_depth*mask_dptm)
                    depth_abs_loss_vol = depth_abs_loss_vol.mean()
                    trip_plane_branch_loss = trip_plane_branch_loss +\
                                self.losses_params.volume_sup_dict.weight_depth_abs_vol * depth_abs_loss_vol
                    set_loss("depth_abs_volume", mode, depth_abs_loss_vol, 
                                        self.losses_params.volume_sup_dict.weight_depth_abs_vol)
                    
                    
                if self.losses_params.use_cv:
                    depth_abs_loss_cv = torch.abs(render_pkg_cv["depth"].squeeze(2) - rendered_gt_depth)
                    depth_abs_loss_cv = depth_abs_loss_cv.mean()
                    cost_volume_branch_loss = cost_volume_branch_loss +\
                                self.losses_params.cv_sup_dict.weight_depth_abs_cv * depth_abs_loss_cv
                    set_loss("depth_abs_cv", mode, depth_abs_loss_cv, self.losses_params.cv_sup_dict.weight_depth_abs_cv)
            
       
                depth_abs_loss = torch.abs(render_pkg_fuse["depth"].squeeze(2) - rendered_gt_depth)
                depth_abs_loss = depth_abs_loss.mean()
                fusion_branch_loss = fusion_branch_loss + self.losses_params.fusion_sup_dict.weight_depth_abs * depth_abs_loss
                set_loss("depth_abs_fusion", mode, depth_abs_loss, self.losses_params.fusion_sup_dict.weight_depth_abs)
            
            
            else:
                raise NotImplementedError
                
            
            loss =cost_volume_branch_loss * self.losses_params.cv_sup_dict.branch_weight + \
                        trip_plane_branch_loss * self.losses_params.volume_sup_dict.branch_weight + \
                            fusion_branch_loss * self.losses_params.fusion_sup_dict.branch_weight + \
                                depth_estimation_branch_loss * self.losses_params.depth_est_sup_dict.branch_weight
            
            
            rendered_fusion_list = [rendered_color_fuse,rendered_depth_fuse]
            rendered_volume_list = [rendered_color_volume,rendered_depth_volume]
            rendered_cv_list = [rendered_color_cv,rendered_depth_cv]
            
            
            if mode=='train':
                return loss, loss_terms,rendered_fusion_list,rendered_volume_list,rendered_cv_list
            
            elif mode=='val':
                return loss, loss_terms,rendered_fusion_list,rendered_volume_list,rendered_cv_list,pred_depth,input_sparse_gt_depth,output_rgb,sparse_depth_gt,img
            
        elif mode=='test':
            return rendered_fusion_list

        else:
            raise NotImplementedError

    
    def validation_step(self, batch, val_result_savedir,cfg=None):
        
        bin_token_name = batch['bin_token'][0][:-4]

        # loss and loss terms
        with torch.no_grad():
            loss, loss_terms,rendered_fusion_list,\
                rendered_volume_list,rendered_cv_results_list, \
                    predicted_input_depth,input_sparse_gt_depth,\
                        output_rgb,sparse_depth_gt,input_images = self.forward(batch,mode='val',
                                                            cfg=cfg)

        batch_data_for_eval = {
            "output_gt_rgb": output_rgb,
            "output_gt_sparse_depth": sparse_depth_gt,
            "input_images": input_images,
            "input_gt_sparse_gt": input_sparse_gt_depth,
            "predicted_input_depth": predicted_input_depth,
            "rendered_fusion": rendered_fusion_list,
            "rendered_cost_volume":rendered_cv_results_list,
            "rendered_volume":rendered_volume_list,
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
        
        rendered_fusion = batch_data_for_eval['rendered_fusion']
        rendered_volume = batch_data_for_eval['rendered_volume']
        rendered_cv = batch_data_for_eval['rendered_cost_volume']
        
        rendered_list = [rendered_fusion,rendered_volume,rendered_cv]
        names = ['fusion','volume','cost_volume']
        
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
                rendered_volume_list,rendered_cv_results_list, \
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
        

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        # batch-wise saved the gaussain-pixel and the feature-pixel
        gaussians_cv_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_cv[b, :, 0] >= x_start) & (gaussians_cv[b, :, 0] <= x_end) & \
                        (gaussians_cv[b, :, 1] >= y_start) & (gaussians_cv[b, :, 1] <= y_end) & \
                        (gaussians_cv[b, :, 2] >= z_start) & (gaussians_cv[b, :, 2] <= z_end)
            # get the valid gaussains in the pixel splat
            gaussians_cv_mask_i = gaussians_cv[b][mask_pixel_i]
            # get the valid feature in the pixel splat
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_cv_mask.append(gaussians_cv_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        


        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                input_batch_dict['extrinsics'],
                gaussians_cv_mask,
                gaussians_feat_mask,
                input_batch_dict["img_metas"])
        

        # Make Sure the estimate gaussains are valid
        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        
        gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
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
    
    
    
    def move_forward_z_by_N_meters(self,first_left_pose, N_meters=4.0):
        # first left pose is 4x4
        
        Rotation_Part = torch.eye(3,3).type(first_left_pose.dtype).to(first_left_pose.device)
        
        z_translations_part = torch.tensor([0,0,-1.0* N_meters],dtype=first_left_pose.dtype).to(first_left_pose.device)
        z_translations_part = z_translations_part.reshape(1,3)
        
        first_left_to_target_pose = torch.eye(4,4).type(first_left_pose.dtype).to(first_left_pose.device)
        first_left_to_target_pose[:3,:3] = Rotation_Part
        first_left_to_target_pose[:3,3] = z_translations_part
        
        return first_left_to_target_pose
    
    
    def generate_novel_views(self,first_left_pose, N1_meters=4.0, N2_meters=8.0,left2right=None):
        
        first_left_to_target_pose1 = self.move_forward_z_by_N_meters(first_left_pose, N_meters=N1_meters)
        frist_left_to_target_pose2 = self.move_forward_z_by_N_meters(first_left_pose, N_meters=N2_meters)
        
        
        target_pose1_left_c2w = torch.inverse(first_left_to_target_pose1 @ torch.inverse(first_left_pose))
        target_pose2_left_c2w = torch.inverse(frist_left_to_target_pose2 @ torch.inverse(first_left_pose))
        
        
        target_pose1_right_c2w = torch.inverse(left2right @ torch.inverse(target_pose1_left_c2w))
        target_pose2_right_c2w = torch.inverse(left2right @ torch.inverse(target_pose2_left_c2w))
        
        
        target_pose1_stereo_c2w = torch.cat([target_pose1_left_c2w.unsqueeze(0).unsqueeze(0),
                                             target_pose1_right_c2w.unsqueeze(0).unsqueeze(0)], dim=1)
        
        target_pose2_stereo_c2w = torch.cat([target_pose2_left_c2w.unsqueeze(0).unsqueeze(0),
                                             target_pose2_right_c2w.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return target_pose1_stereo_c2w, target_pose2_stereo_c2w
    

    def get_world2image(self,
                        cam2world: torch.Tensor,
                        intrinsics: torch.Tensor) -> torch.Tensor:
        """
        cam2world:  (B, V, 4, 4)  camera -> world
        intrinsics: (B, V, 3, 3)  K
        return:     (B, V, 4, 4)  world -> image (projection in homogeneous coords)
        """
        # 1. 从 cam2world 拿出 R_wc, t_wc
        R_wc = cam2world[..., :3, :3]          # (B, V, 3, 3)
        t_wc = cam2world[..., :3, 3:4]         # (B, V, 3, 1)

        # 2. 求 world2cam = [R_cw | t_cw] = [R_wc^T | -R_wc^T t_wc]
        R_cw = R_wc.transpose(-1, -2)          # (B, V, 3, 3)
        t_cw = -R_cw @ t_wc                    # (B, V, 3, 1)
        extrinsic = torch.cat([R_cw, t_cw], dim=-1)  # (B, V, 3, 4)

        # 3. 投影矩阵 P = K [R|t]  => (B, V, 3, 4)
        P = intrinsics @ extrinsic             # (B, V, 3, 4)

        # 4. 填到 4x4 里（最后一行 [0,0,0,1]）
        B, V = P.shape[:2]
        world2image = torch.zeros(B, V, 4, 4, device=P.device, dtype=P.dtype)
        world2image[..., :3, :4] = P
        world2image[..., 3, 3] = 1.0

        return world2image

    
    # iteration twice
    def validation_on_the_forward_views_progressive_fixed_cam_batch_inference(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        start_images_views = 2,
                                        use_diffix3d=False,
                                        diffix3d_network=None,
                                        use_ref=False,
                                        cfg=None,
                                        vis=False,
                                        ):
        
        bin_token_name = bin_token_list[0][:-4]
        
        
        if start_images_views == 2:
            view_num = 2
            matching_nums = 2
        else:
            raise NotImplementedError


       # First Time Iteration rendered images from the first frame stereo
        with torch.no_grad():
            input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=view_num,
                                                                         matching_nums=matching_nums)
            
            
            start_time1 = time.time()
            img =input_batch_dict["imgs"] #[B,6,3,H,W]
            height,width = img.shape[-2:]
            bs = img.shape[0]   
            
            img_feats = self.extract_img_feat(img=img)
            gaussians_cv,gaussians_feat,pred_depths = self.costvolume_gs(input_batch_dict,cfg=cfg,
                                                            images_feat=img_feats[0])

            # volume-gs prediction
            pc_range = self.dataset_params.pc_range
            x_start, y_start, z_start, x_end, y_end, z_end = pc_range
            # batch-wise saved the gaussain-pixel and the feature-pixel
            gaussians_cv_mask, gaussians_feat_mask = [], []
            for b in range(bs):
                mask_pixel_i = (gaussians_cv[b, :, 0] >= x_start) & (gaussians_cv[b, :, 0] <= x_end) & \
                            (gaussians_cv[b, :, 1] >= y_start) & (gaussians_cv[b, :, 1] <= y_end) & \
                            (gaussians_cv[b, :, 2] >= z_start) & (gaussians_cv[b, :, 2] <= z_end)
                # get the valid gaussains in the pixel splat
                gaussians_cv_mask_i = gaussians_cv[b][mask_pixel_i]
                # get the valid feature in the pixel splat
                gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
                gaussians_cv_mask.append(gaussians_cv_mask_i)
                gaussians_feat_mask.append(gaussians_feat_mask_i)

        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                input_batch_dict['extrinsics'],
                gaussians_cv_mask,
                gaussians_feat_mask,
                input_batch_dict["img_metas"])

        # Make Sure the estimate gaussains are valid
        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        
        
        gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
        bs = gaussians_all.shape[0] # batch size is 2
        
        
        # doing rendering here
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        interleave_render_c2w = interleave_left_right_pose(render_c2w)
        
        frist_stereo_render_c2w = interleave_render_c2w[:,-2:,:,:]
        last_stereo_render_c2w = interleave_render_c2w[:,-4:-2,:,:]
        center_stereo_render_c2w = interleave_render_c2w[:,-6:-4,:,:]
        
        first_left_cam_pose = frist_stereo_render_c2w[0][0]
        first_right_cam_pose = frist_stereo_render_c2w[0][1]
        
        center_left_cam_pose = center_stereo_render_c2w[0][0]
        center_right_cam_pose = center_stereo_render_c2w[0][1]
        
        last_left_cam_pose = last_stereo_render_c2w[0][0]
        last_right_cam_pose = last_stereo_render_c2w[0][1]
        
        
        left2right_cam_pose = torch.inverse(first_right_cam_pose) @ first_left_cam_pose
        
        target_pose1_stereo_c2w, target_pose2_stereo_c2w = self.generate_novel_views(first_left_cam_pose, 
                                                                                     N1_meters=4.0, 
                                                                                     N2_meters=8.0,
                                                                                     left2right=left2right_cam_pose)
        
        
        render_c2w = torch.cat([target_pose1_stereo_c2w, target_pose2_stereo_c2w], dim=1)
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-6:-2]# [B,6*3]
        render_fovys = output_batch_dict["output_fovys"][:,-6:-2] # [B,6*3]
        gt_center_frame = interleave_left_right(output_batch_dict["output_imgs"])
        # gt_center_frame =gt_center_frame[:,-6:-2,:,:,:]
        
        gaussians_stage1 = gaussians_all
        
        
        render_c2w_cl = interleave_render_c2w[:,-6:-2,:,:]
        intrinsics_cl = input_batch_dict['intrinsics']
        intrinsics_cl = intrinsics.clone()     
        output_intrinsics_cl = intrinsics_cl[:,0:1,:,:].repeat(1,render_c2w_cl.shape[1],1,1)
        render_fovxs_cl = output_batch_dict["output_fovxs"][:,-6:-2]# [B,6*3]
        render_fovys_cl = output_batch_dict["output_fovys"][:,-6:-2] # [B,6*3]
        
        
        rendered_cl_pkg = self.renderer.render(
            gaussians=gaussians_stage1,
            c2w=render_c2w_cl,
            fovx=render_fovxs_cl,
            fovy=render_fovys_cl,
            rays_o=None,
            rays_d=None
        )  

        rendered_cl_images = rendered_cl_pkg['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_cl_images = torch.clamp(rendered_cl_images,min=0,max=1.0)
        rendered_center_frames = rendered_cl_images[:,:2,:,:,:]
        
        gt_center_frames = interleave_left_right(output_batch_dict["output_imgs"])
        gt_center_frames =gt_center_frames[:,-6:-4,:,:,:]
        
        my_input_center_frames = rendered_center_frames*0.6 + gt_center_frames*0.4
        my_input_center_frames = torch.clamp(my_input_center_frames,min=0,max=1.0)

        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_stage1,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_frames = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_frames = torch.clamp(rendered_frames,min=0,max=1.0)
        
        
        if use_diffix3d:
            # enhance the center frame
            rendered_center_frame = rendered_frames 
            rendered_center_left = rendered_center_frame[0,0,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_center_right = rendered_center_frame[0,1,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_last_left = rendered_center_frame[0,2,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_last_right = rendered_center_frame[0,3,:,:,:].permute(1,2,0).cpu().numpy()
            
            
            
            rendered_center_left = (rendered_center_left*255).astype(np.uint8)
            rendered_center_right = (rendered_center_right*255).astype(np.uint8)
            rendered_last_left = (rendered_last_left*255).astype(np.uint8)
            rendered_last_right = (rendered_last_right*255).astype(np.uint8)
            
            
            
            rendered_center_left_pil = Image.fromarray(rendered_center_left)
            rendered_center_right_pil = Image.fromarray(rendered_center_right)
            rendered_last_left_pil = Image.fromarray(rendered_last_left)
            rendered_last_right_pil = Image.fromarray(rendered_last_right)
            
            width,height = rendered_center_left_pil.size
            
            # get the ref image
            ref_image_left = input_batch_dict["imgs"][0,0,:,:,:].permute(1,2,0).cpu().numpy()
            ref_image_left = (ref_image_left*255).astype(np.uint8)
            ref_image_left_pil = Image.fromarray(ref_image_left)
            ref_image_right = input_batch_dict["imgs"][0,1,:,:,:].permute(1,2,0).cpu().numpy()
            ref_image_right = (ref_image_right*255).astype(np.uint8)
            ref_image_right_pil = Image.fromarray(ref_image_right)

            enhanced_rendered_center_left_pil = diffix3d_network.sample(
                    rendered_center_left_pil,
                    height=112,
                    width=544,
                    ref_image=ref_image_left_pil,
                    prompt=cfg.prompt
                )
            enhanced_rendered_center_right_pil = diffix3d_network.sample(
                    rendered_center_right_pil,
                    height=112,
                    width=544,
                    ref_image=ref_image_right_pil,
                    prompt=cfg.prompt
                )
            enhanced_rendered_last_left_pil = diffix3d_network.sample(
                    rendered_last_left_pil,
                    height=112,
                    width=544,
                    ref_image=ref_image_left_pil,
                    prompt=cfg.prompt
                )
            enhanced_rendered_last_right_pil = diffix3d_network.sample(
                    rendered_last_right_pil,
                    height=112,
                    width=544,
                    ref_image=ref_image_right_pil,
                    prompt=cfg.prompt
                )

            enhanced_rendered_center_left = np.array(enhanced_rendered_center_left_pil).astype(np.float32)/255.0
            enhanced_rendered_center_right = np.array(enhanced_rendered_center_right_pil).astype(np.float32)/255.0
            enhanced_rendered_last_left = np.array(enhanced_rendered_last_left_pil).astype(np.float32)/255.0
            enhanced_rendered_last_right = np.array(enhanced_rendered_last_right_pil).astype(np.float32)/255.0
            
            enhanced_rendered_center_left = torch.from_numpy(enhanced_rendered_center_left).to(rendered_center_frame.device)
            enhanced_rendered_center_right = torch.from_numpy(enhanced_rendered_center_right).to(rendered_center_frame.device)
            enhanced_rendered_last_left = torch.from_numpy(enhanced_rendered_last_left).to(rendered_center_frame.device)
            enhanced_rendered_last_right = torch.from_numpy(enhanced_rendered_last_right).to(rendered_center_frame.device)
            enhanced_rendered_center_left = enhanced_rendered_center_left.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_center_right = enhanced_rendered_center_right.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_last_left = enhanced_rendered_last_left.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_last_right = enhanced_rendered_last_right.permute(2,0,1).unsqueeze(0)
            
            enhanced_rendered_center_frame = torch.cat([enhanced_rendered_center_left,
                                                        enhanced_rendered_center_right,
                                                        enhanced_rendered_last_left,
                                                        enhanced_rendered_last_right],dim=0).unsqueeze(0)
            
            rendered_center_frame = enhanced_rendered_center_frame
            
        
            rendered_frames = rendered_center_frame 



        '''second time inference'''
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=6,
                                                                         matching_nums=4)
        
        
        rendered_additional_views = rendered_frames
        rendered_additional_views = torch.cat([img,rendered_additional_views],dim=1)
        
        
        
        true_first_center_last_stereo = torch.cat([torch.cat([input_batch_dict['imgs'][0][0],
                                                             input_batch_dict['imgs'][0][1]],dim=-1),
                                                   torch.cat([input_batch_dict['imgs'][0][2],
                                                             input_batch_dict['imgs'][0][3]],dim=-1),
                                                   torch.cat([input_batch_dict['imgs'][0][4],
                                                             input_batch_dict['imgs'][0][5]],dim=-1)],
                                                  dim=-2).permute(1,2,0).cpu().numpy()
        
        
        
        fixed_pose_frist_center_last_stereo = torch.cat([
                                                        torch.cat([rendered_additional_views[0][0],
                                                        rendered_additional_views[0][1]],
                                                        dim=-1),
                                                        torch.cat([rendered_additional_views[0][2],
                                                        rendered_additional_views[0][3]],
                                                        dim=-1),
                                                        torch.cat([rendered_additional_views[0][4],
                                                        rendered_additional_views[0][5]],
                                                        dim=-1)],
                                                        dim=-2).permute(1,2,0).cpu().numpy()
        
        
        true_first_center_last_stereo_vis =true_first_center_last_stereo
        fixed_pose_frist_center_last_stereo_vis = fixed_pose_frist_center_last_stereo
        true_first_center_last_stereo_vis = (true_first_center_last_stereo_vis*255).astype(np.uint8)
        fixed_pose_frist_center_last_stereo_vis = (fixed_pose_frist_center_last_stereo_vis*255).astype(np.uint8)
        
        
            
        # adjust input views and camera poses
        # input_batch_dict["imgs"][:,2:4,:,:,:] = my_input_center_frames
        # input_batch_dict["imgs"][:,4:,:,:,:] = rendered_frames[:,2:,:,:,:]
        # input_batch_dict['extrinsics'][:,4:,:,:] = render_c2w[:,2:,:,:]
        # world2image_pose = self.get_world2image(input_batch_dict['extrinsics'],
        #                                         input_batch_dict['intrinsics'])
        # input_batch_dict['img_metas'][0]['lidar2img'][4:,:,:] = world2image_pose[0][4:]
        

        input_batch_dict["imgs"][:,2:,:,:,:] = rendered_frames
        input_batch_dict['extrinsics'][:,2:,:,:] = render_c2w
        world2image_pose = self.get_world2image(input_batch_dict['extrinsics'],
                                                input_batch_dict['intrinsics'])
        input_batch_dict['img_metas'][0]['lidar2img'][2:,:,:] = world2image_pose[0][2:]

        

        img =input_batch_dict["imgs"] #[B,6,3,H,W]

        height,width = img.shape[-2:]
        bs = img.shape[0]   
        img_feats = self.extract_img_feat(img=img)
        gaussians_cv,gaussians_feat,pred_depths = self.costvolume_gs(input_batch_dict,cfg=cfg,
                                                        images_feat=img_feats[0])

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        # batch-wise saved the gaussain-pixel and the feature-pixel
        gaussians_cv_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_cv[b, :, 0] >= x_start) & (gaussians_cv[b, :, 0] <= x_end) & \
                        (gaussians_cv[b, :, 1] >= y_start) & (gaussians_cv[b, :, 1] <= y_end) & \
                        (gaussians_cv[b, :, 2] >= z_start) & (gaussians_cv[b, :, 2] <= z_end)
            # get the valid gaussains in the pixel splat
            gaussians_cv_mask_i = gaussians_cv[b][mask_pixel_i]
            # get the valid feature in the pixel splat
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_cv_mask.append(gaussians_cv_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)

        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                input_batch_dict['extrinsics'],
                gaussians_cv_mask,
                gaussians_feat_mask,
                input_batch_dict["img_metas"])

        # Make Sure the estimate gaussains are valid
        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        
        
        gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
        bs = gaussians_all.shape[0] # batch size is 2
        

        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)        
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"]
        render_fovys = output_batch_dict["output_fovys"]
        

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
        
        output_rgb = output_batch_dict['output_imgs']
        sparse_depth_gt = output_batch_dict['output_sparse_depth']  
        
        
        

        '''Do the visualization and the evaluation here'''
        rendered_images_fusion = interleave_left_right(rendered_color_fuse)
        rendered_depth_fusion = interleave_left_right_depth(rendered_depth_fuse)
        sparse_depth_gt = interleave_left_right_depth(sparse_depth_gt)
        rendered_images_gt = interleave_left_right(output_rgb)
        
        
        
        
        
    
        if cfg.use_diffix3d_postprocessing:
            
            enhanced_rendered_images_list = []
            
            for current_view_idx in range(rendered_images_fusion.shape[1]):
                current_rendered_image = rendered_images_fusion[0,current_view_idx,:,:,:].permute(1,2,0).cpu().numpy()
                current_rendered_image = (current_rendered_image*255).astype(np.uint8)
                current_rendered_image_pil = Image.fromarray(current_rendered_image)
                
                current_rendered_image_pil = diffix3d_network.sample(
                        current_rendered_image_pil,
                        height=112,
                        width=544,
                        ref_image=ref_image_left_pil,
                        prompt=cfg.prompt
                    )
                current_rendered_image_np = np.array(current_rendered_image_pil).astype(np.float32)/255.0
                current_rendered_image = torch.from_numpy(current_rendered_image_np).to(rendered_center_frame.device)
                current_rendered_image = current_rendered_image.permute(2,0,1).unsqueeze(0).unsqueeze(0)
                enhanced_rendered_images_list.append(current_rendered_image)
                
            rendered_images_fusion = torch.cat(enhanced_rendered_images_list,dim=1)
        


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
        
        
        
        # visualization
        
        rendered_first_stereo = torch.cat((rendered_images_first_stereo[0,0,:,:,:],
                                           rendered_images_first_stereo[0,1,:,:,:]),dim=-2).permute(1,2,0).cpu().numpy()
        
        rendered_last_stereo = torch.cat((rendered_images_last_stereo[0,0,:,:,:],
                                           rendered_images_last_stereo[0,1,:,:,:]),dim=-2).permute(1,2,0).cpu().numpy()
        
        rendered_center_stereo = torch.cat((rendered_images_center_stereo[0,0,:,:,:],
                                           rendered_images_center_stereo[0,1,:,:,:]),dim=-2).permute(1,2,0).cpu().numpy()
        
        rendered_first_stereo = (rendered_first_stereo*255).astype(np.uint8)
        rendered_last_stereo = (rendered_last_stereo*255).astype(np.uint8)
        rendered_center_stereo = (rendered_center_stereo*255).astype(np.uint8)
        

    
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
            
            
            # input first center and fixed pose.
            Input_First_Center_Last_Stereo_Folder_Path = os.path.join(saved_folder_for_visualization,
                                                                      'Input_First_Center_Last_Stereo')
            Fixed_Pose_First_Center_Last_Stereo_Folder_Path = os.path.join(saved_folder_for_visualization,
                                                                            'Fixed_Pose_First_Center_Last_Stereo')
            os.makedirs(Input_First_Center_Last_Stereo_Folder_Path,exist_ok=True)
            os.makedirs(Fixed_Pose_First_Center_Last_Stereo_Folder_Path,exist_ok=True)
            
            
            skimage.io.imsave(os.path.join(Input_First_Center_Last_Stereo_Folder_Path,
                                           'true_first_center_last_stereo.png'),
                              true_first_center_last_stereo_vis)
            skimage.io.imsave(os.path.join(Fixed_Pose_First_Center_Last_Stereo_Folder_Path,
                                           'fixed_pose_first_center_last_stereo.png'),
                              fixed_pose_frist_center_last_stereo_vis)
            
            
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




        
        # first_left_cam_to_right_cam = torch.inverse(first_right_cam_pose) @ first_left_cam_pose
        
        # center_left_cam_to_right_cam = torch.inverse(center_right_cam_pose) @ center_left_cam_pose
        
        # last_left_cam_to_right_cam = torch.inverse(last_right_cam_pose) @ last_left_cam_pose
        
        
        # print(first_left_cam_to_right_cam)
        # print("--------------------------------")
        # print(center_left_cam_to_right_cam)
        # print("--------------------------------")
        # print(last_left_cam_to_right_cam)
        # quit()
        
        # # first_left_cam_pose = first_stereo


        # print(frist_stereo_render_c2w.shape)
        # print(last_stereo_render_c2w.shape)
        # print(center_stereo_render_c2w.shape)
        # quit()




def get_mean(list):
    return sum(list)*1.0/len(list)


def saved_into_json(data_dict,path):
    import json
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)

if __name__=="__main__":
    pass