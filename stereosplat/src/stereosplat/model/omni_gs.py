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
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from .gaussian import GaussianRenderer

#from .losses import LPIPS, LossDepthTVS
from .losses import LPIPS
from .utils.image import maybe_resize
from .utils.benchmarker import Benchmarker
from torchmetrics import PearsonCorrCoef
from .utils.interpolation import interpolate_extrinsics

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
import copy
import lpips
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from tqdm import tqdm


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


@MODELS.register_module()
class OmniGaussian(BaseModule):
    
    def __init__(self,
                 backbone=None, # feature extraction
                 neck=None,      # feature aggregation
                 pixel_gs=None,  # pixel gaussains modeling
                 volume_gs=None, # volume gaussains modeling
                 camera_args=None, # camera/3D Range
                 loss_args=None,    # loss args setings
                 dataset_params=None, # dataset params
                 use_checkpoint=False, # using checkpoints or not
                 **kwargs,
                 ):

        super().__init__()

        assert pixel_gs is not None and volume_gs is not None
        self.use_checkpoint = use_checkpoint # The Feature Extraction Part Using Checkpoints or Not.
        
        # build the backbone here for feature extraction
        if backbone:
            self.backbone = MODELS.build(backbone)
        if neck:
            self.neck = MODELS.build(neck)
        
        # pixel feed forward 3DGS    
        self.pixel_gs = MODELS.build(pixel_gs)
        
        # volume feed forward 3DGS
        self.volume_gs = MODELS.build(volume_gs)

        self.dataset_params = dataset_params
        self.camera_args = camera_args
        self.loss_args = loss_args

        # gaussain renderers
        self.renderer = GaussianRenderer(self.device, **camera_args)

        # Perceptual loss
        if self.loss_args.weight_perceptual > 0:
            # self.perceptual_loss = LPIPS(net="vgg")
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = None

        # record runtime
        self.benchmarker = Benchmarker()

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
       
    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    def get_data(self, batch,mode='train'):

        # ================== batch data process ================== #
        device_id = self.device
        data_dict = {}
        # for img feature extraction
        data_dict["imgs"] = batch["inputs"]["rgb"].to(device_id, dtype=self.dtype)
        # for pixel-gs
        rays_o = batch["inputs_pix"]["rays_o"].to(device_id, dtype=self.dtype)
        rays_d = batch["inputs_pix"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["rays_o"] = rays_o
        data_dict["rays_d"] = rays_d
        data_dict["pluckers"] = self.plucker_embedder(rays_o, rays_d)
        data_dict["fxs"] = batch["inputs_pix"]["fx"].to(device_id, dtype=self.dtype)
        data_dict["fys"] = batch["inputs_pix"]["fy"].to(device_id, dtype=self.dtype)
        data_dict["cxs"] = batch["inputs_pix"]["cx"].to(device_id, dtype=self.dtype)
        data_dict["cys"] = batch["inputs_pix"]["cy"].to(device_id, dtype=self.dtype)
        data_dict["c2ws"] = batch["inputs_pix"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["cks"] = batch["inputs_pix"]["ck"].to(device_id, dtype=self.dtype)
        data_dict["depths"] = batch["inputs_pix"]["depth_m"].to(device_id, dtype=self.dtype) # using metric depth.
        data_dict["confs"] = batch["inputs_pix"]["conf_m"].to(device_id, dtype=self.dtype)   # using confidence map.
        # for volume-gs
        img_metas = []
        bs, v, c, h, w = batch["inputs"]["rgb"].shape
        for w2i in batch["inputs_vol"]["w2i"]:
            img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
        data_dict["img_metas"] = img_metas
        # for render and loss and eval
        data_dict["output_imgs"] = batch["outputs"]["rgb"].to(device_id, dtype=self.dtype)
        data_dict["output_depths"] = batch["outputs"]["depth"].to(device_id, dtype=self.dtype)
        data_dict["output_depths_m"] = batch["outputs"]["depth_m"].to(device_id, dtype=self.dtype)
        data_dict["output_confs_m"] = batch["outputs"]["conf_m"].to(device_id, dtype=self.dtype)
        data_dict["output_positions"] = (batch["outputs"]["rays_o"] + batch["outputs"]["rays_d"] * \
                            batch["outputs"]["depth_m"].unsqueeze(-1)).to(device_id, dtype=self.dtype)
        data_dict["output_rays_o"] = batch["outputs"]["rays_o"].to(device_id, dtype=self.dtype)
        data_dict["output_rays_d"] = batch["outputs"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["output_c2ws"] = batch["outputs"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["output_fovxs"] = batch["outputs"]["fovx"].to(device_id, dtype=self.dtype)
        data_dict["output_fovys"] = batch["outputs"]["fovy"].to(device_id, dtype=self.dtype)

        data_dict["bin_token"] = batch["bin_token"]
        
        if mode!='train':
            data_dict['output_sparse_gt_depth'] = batch['outputs']['sparse_gt_depth'].to(device_id, dtype=self.dtype)

        return data_dict
    
    def configure_optimizers(self, lr):
        backbone_layers = torch.nn.ModuleList([self.backbone])
        backbone_layers_params = list(map(id, backbone_layers.parameters()))
        base_params = list(filter(lambda p: id(p) not in backbone_layers_params, self.parameters()))
        
        opt = torch.optim.AdamW(
            [{'params': base_params}, {'params': backbone_layers.parameters(), 'lr': lr*0.1}],
            lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
        return [opt]
    
    def forward(self, batch, split="train", iter=0, iter_end=100000):
        
        
        """Forward training function."""
        data_dict = self.get_data(batch,mode=split)
        img = data_dict["imgs"] #[B,6,3,H,W]
        
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
        '''
        - torch.Size([2, 6, 128, 56, 100]) ---> 1/4
        - torch.Size([2, 6, 128, 28, 50])  ---> 1/8
        - torch.Size([2, 6, 128, 14, 25])  ---> 1/16
        - torch.Size([2, 6, 128, 7, 13])   ---> 1/32
        '''
        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # gaussians_feat: torch.Size([2, 537600, 128])
        # gaussians_pixel: torch.Size([2, 537600, 14])

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        # batch-wise saved the gaussain-pixel and the feature-pixel
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            # get the valid gaussains in the pixel splat
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            # get the valid feature in the pixel splat
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        
        # volume gs: input the features and gaussains pixel mask and guassain fature mask
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        # Make Sure the estimate gaussains are valid
        gaussians_pixel = sanitize_gaussians_tensor(gaussians_pixel)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)
        
        bs = gaussians_all.shape[0] # batch size is 2
        

        '''
        data dicts: 
                dict_keys(['imgs', 'rays_o', 'rays_d', 'pluckers', 'fxs', 'fys', 'cxs', 'cys', 'c2ws', 
                'cks', 'depths', 'confs', 'img_metas', 'output_imgs', 'output_depths', 'output_depths_m', 
                'output_confs_m', 'output_positions', 
                'output_rays_o', 'output_rays_d', 'output_c2ws', 
                'output_fovxs', 'output_fovys', 'bin_token'])
        '''

        
        #  first 2 dimension is the novel final ,final dimension is the input view
        render_c2w = data_dict["output_c2ws"] # render last and first camera 2 word: [B,6*3,4,4]
        render_fovxs = data_dict["output_fovxs"] # [B,6*3]
        render_fovys = data_dict["output_fovys"] # [B,6*3]
        

        # return a dicts: rendered images and rendered alphs and rendered depth
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  
        # print(render_pkg_fuse["depth"].shape)       #[2,18,1,H,W]
        # print(render_pkg_fuse['image'].shape)       #[2,18,3,H,W]
        # print(render_pkg_fuse['alpha'].shape)       #[2,18,1,H,W]
        # print(render_pkg_fuse.keys())    # dict_keys(['image', 'alpha', 'depth'])

        if split == "train" or split == "val":
            render_pkg_pixel = self.renderer.render(
                gaussians=gaussians_pixel,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
            render_pkg_volume = self.renderer.render(
                gaussians=gaussians_volume,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
        else:
            render_pkg_pixel, render_pkg_volume = None, None
        
        # Get the Loss Here
        # ======================== losses ======================== #
        loss = 0.0
        loss_terms = {}
        def set_loss(key, split, loss_value, loss_weight=1.0):
            loss_terms[f"{split}/loss_{key}"] = loss_value.item()
            loss_terms[f"{split}/loss_{key}_w"] = loss_value.item() * loss_weight

        # =================== Data preparation =================== #        
        rgb_gt = data_dict["output_imgs"]
     
        data_dict["rgb_gt"] = rgb_gt
        depth_m_gt = data_dict["output_depths_m"] # Depth from Metric3D-V2
        conf_m_gt = data_dict["output_confs_m"]
        data_dict["depth_m_gt"] = depth_m_gt
        data_dict["conf_m_gt"] = conf_m_gt
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        
        # Masked Mask, which is to get the mask shared for volume and the pixel-based 3D Gaussains
        if self.loss_args.mask_dptm and self.loss_args.recon_loss_vol_type == "l2_mask_self":
            # obtain the depth from the pixel gaussains: using the render depths
            depth_for_mask = render_pkg_pixel["depth"].squeeze(2).detach().unsqueeze(-1)
            output_positions = data_dict["output_rays_o"] + data_dict["output_rays_d"] * depth_for_mask # get the 3D loctions
            
            mask_dptm = (output_positions[..., 0] >= x_start) & (output_positions[..., 0] <= x_end) & \
                        (output_positions[..., 1] >= y_start) & (output_positions[..., 1] <= y_end) & \
                        (output_positions[..., 2] >= z_start) & (output_positions[..., 2] <= z_end)
            # only capture the depth which is bigger than 0.1
            mask_dptm = mask_dptm & (depth_for_mask[..., 0] > 0.1)
            mask_dptm = mask_dptm.float()
            
        elif self.loss_args.mask_dptm and self.loss_args.recon_loss_vol_type == "l2_mask":
            output_positions = data_dict["output_positions"]
            # using depth from a pre-trained metric3D Model
            mask_dptm = (output_positions[..., 0] >= x_start) & (output_positions[..., 0] <= x_end) & \
                        (output_positions[..., 1] >= y_start) & (output_positions[..., 1] <= y_end) & \
                        (output_positions[..., 2] >= z_start) & (output_positions[..., 2] <= z_end)
            mask_dptm = mask_dptm.float()
        data_dict["mask_dptm"] = mask_dptm
        
        
        # ======================== RGB loss ======================== #
        if self.loss_args.weight_recon > 0:
            # RGB loss for omni-gs
            # Total Rendered Loss using global gaussains
            if self.loss_args.recon_loss_type == "l1":
                rec_loss = torch.abs(rgb_gt - render_pkg_fuse["image"])
            elif self.loss_args.recon_loss_type == "l2":
                rec_loss = (rgb_gt - render_pkg_fuse["image"]) ** 2
            loss = loss + (rec_loss.mean() * self.loss_args.weight_recon)
            set_loss("recon", split, rec_loss.mean(), self.loss_args.weight_recon)
            
        if self.loss_args.weight_recon_vol > 0 and iter < iter_end - 1000:
            # RGB loss for volume-gs
            if self.loss_args.recon_loss_vol_type == "l1":
                rec_loss_vol = torch.abs(rgb_gt - render_pkg_volume["image"])
            elif self.loss_args.recon_loss_vol_type == "l2":
                rec_loss_vol = (rgb_gt - render_pkg_volume["image"]) ** 2
                
            elif self.loss_args.recon_loss_vol_type == "l2_mask" or self.loss_args.recon_loss_vol_type == "l2_mask_self":
                rec_loss_vol = (rgb_gt * mask_dptm.unsqueeze(2) - render_pkg_volume["image"] * mask_dptm.unsqueeze(2)) ** 2
                
            loss = loss + (rec_loss_vol.mean() * self.loss_args.weight_recon_vol)
            set_loss("recon_vol", split, rec_loss_vol.mean(), self.loss_args.weight_recon_vol)

        # ==================== Perceptual loss ===================== #
        if self.loss_args.weight_perceptual > 0:
            # Perceptual loss for omni-gs
            ## resize images to smaller size to save memory
            p_inp_pred = maybe_resize(
                render_pkg_fuse["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_inp_gt = maybe_resize(
                rgb_gt.reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_loss = self.perceptual_loss(p_inp_pred, p_inp_gt)
            p_loss = rearrange(p_loss, "(b v) c h w -> b v c h w", b=bs)
            p_loss = p_loss.mean()
            loss = loss + (p_loss * self.loss_args.weight_perceptual)
            set_loss("perceptual", split, p_loss, self.loss_args.weight_perceptual)
            
        if self.loss_args.weight_perceptual_vol > 0 and iter < iter_end - 1000:
            # Perceptual loss for volume-gs
            p_inp_pred_vol = maybe_resize(
                render_pkg_volume["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_inp_mask_vol = maybe_resize(
                mask_dptm.reshape(-1, 1, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_loss_vol = self.perceptual_loss(p_inp_pred_vol * p_inp_mask_vol, p_inp_gt * p_inp_mask_vol)
            p_loss_vol = rearrange(p_loss_vol, "(b v) c h w -> b v c h w", b=bs)
            p_loss_vol = p_loss_vol.mean()
            loss = loss + (p_loss_vol * self.loss_args.weight_perceptual_vol)
            set_loss("perceptual_vol", split, p_loss_vol, self.loss_args.weight_perceptual_vol)

        # ==================== Depth loss ===================== #
        ### Depth loss for omni-gs. For regularization use.
        if self.loss_args.weight_depth_abs > 0:
            # using metric3D metric depth for supervision
            depth_abs_loss = torch.abs(render_pkg_fuse["depth"].squeeze(2) - depth_m_gt)
            depth_abs_loss = depth_abs_loss * conf_m_gt
            depth_abs_loss = depth_abs_loss.mean()
            loss = loss + self.loss_args.weight_depth_abs * depth_abs_loss
            set_loss("depth_abs", split, depth_abs_loss, self.loss_args.weight_depth_abs)
            
        ### Depth loss for volume-gs
        if self.loss_args.weight_depth_abs_vol > 0 and iter < iter_end - 1000:
            if self.loss_args.depth_abs_loss_vol_type == "mask":
                depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"].squeeze(2) * mask_dptm - depth_m_gt * mask_dptm)
                depth_abs_loss_vol = depth_abs_loss_vol * conf_m_gt
                
            elif self.loss_args.depth_abs_loss_vol_type == "mask_self":
                # consistency loss used
                depth_m_gt_pseudo = render_pkg_pixel["depth"].squeeze(2).detach()
                depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"].squeeze(2) * mask_dptm - depth_m_gt_pseudo * mask_dptm)
            depth_abs_loss_vol = depth_abs_loss_vol.mean()
            loss = loss + self.loss_args.weight_depth_abs_vol * depth_abs_loss_vol
            set_loss("depth_abs_vol", split, depth_abs_loss_vol, self.loss_args.weight_depth_abs_vol)
        
        return loss, loss_terms, render_pkg_fuse, render_pkg_pixel, render_pkg_volume, gaussians_all, gaussians_pixel, gaussians_volume, data_dict
    
    def validation_step(self, batch, val_result_savedir):
        (loss_val, loss_term_val, render_pkg_fuse,
         render_pkg_pixel, render_pkg_volume, gaussians_all,
         gaussians_pixel, gaussians_volume, batch_data) = \
            self.forward(batch, "val")
            
        self.save_val_results(batch_data, render_pkg_fuse, render_pkg_pixel, render_pkg_volume,
                                gaussians_all, gaussians_pixel, gaussians_volume, val_result_savedir)
        return loss_term_val
    
    def forward_test(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img, status="test")


        # pixel-gs prediction
        with self.benchmarker.time("pixel_gs"):
            gaussians_pixel, gaussians_feat = self.pixel_gs(
                    rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                    data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                    data_dict["rays_o"], data_dict["rays_d"], status="test")

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        with self.benchmarker.time("volume_gs"):
            gaussians_volume = self.volume_gs(
                    [img_feats[0]],
                    gaussians_pixel_mask,
                    gaussians_feat_mask,
                    data_dict["img_metas"], status="test")
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)
        bs = gaussians_all.shape[0]
        render_c2w = data_dict["output_c2ws"]
        render_fovxs = data_dict["output_fovxs"]
        render_fovys = data_dict["output_fovys"]
        
        with self.benchmarker.time("render", num_calls=render_c2w.shape[1]):
            render_pkg_fuse = self.renderer.render(
                gaussians=gaussians_all,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )

        output_imgs = render_pkg_fuse["image"] # b v 3 h w
        output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w

        target_imgs = data_dict["output_imgs"] # b v 3 h w
        target_depths = data_dict["output_depths"] # b v h w
        target_depths_m = data_dict["output_depths_m"] # b v h w

        preds = {"img": output_imgs, "depth": output_depths, "gaussian": gaussians_all}
        gts = {"img": target_imgs, "depth": target_depths, "depth_m": target_depths_m}

        return preds, gts, data_dict["bin_token"]
    
    def forward_demo(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"] #(4,6,3,H,W)        
        bs = img.shape[0] # batch size is 4
        
        # first feature extraction: here is using?
        img_feats = self.extract_img_feat(img=img, status="test")
        
        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # volume-gs prediction
        pc_range = self.dataset_params.pc_range # whole pc range is the default nuScene
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        
        # masks, here get the share space location with the pixelGS
        gaussians_pixel_mask, gaussians_feat_mask = [], [] #-----> saved to gaussain pixel / feat masks
        for b in range(bs):
            # this masks is applied into gaussians_pixel: according to the mean locations
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            # pixel gaussain masks
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        
        
        # Estimate Volume Guassains
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        # print(gaussians_volume.shape) #torch.Size([4, 1769472, 14])
        # print(gaussians_pixel.shape) #torch.Size([4, 537600, 14])
        # quit()
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1) # cancate them together for final rendering
        bs = gaussians_all.shape[0]     # batch size
        

        # forward 3 meters, return, and then rotate. backward 3 meters, return, and then rotate.
        '''
        cf: center front         ------> 0 (-6)
        cfr: center front right  ------> 1 (-5)
        cfl: center front left   ------> 2 (-4)
        cb: center back          ------> 3 (-3)
        cbr: center back right   ------> 4 (-2)
        cbl: center back left    ------->5 (-1)
        '''
        c2w_cf = data_dict["output_c2ws"][:, -6] # final 6 is the input, [4,1,4,4]--> Front
        c2w_cf_forward = c2w_cf.clone()
        c2w_cf_forward[..., 1, 3] = c2w_cf_forward[..., 1, 3] + 3 # forward
        c2w_cfr = data_dict["output_c2ws"][:, -5]  # front right
        c2w_cfl = data_dict["output_c2ws"][:, -4]  # front left
        c2w_cb = data_dict["output_c2ws"][:, -3]   # back
        c2w_cb[..., 1, 3] = c2w_cb[..., 1, 3] + 1.5 
        c2w_cb_backward = c2w_cb.clone()
        c2w_cb_backward[..., 1, 3] = c2w_cb_backward[..., 1, 3] - 3 # 向后退 3 米
        c2w_cbl = data_dict["output_c2ws"][:, -2]
        c2w_cbr = data_dict["output_c2ws"][:, -1]
        
        #-------------------------------------------------------------------------------------#
        #--------------# cf -> cfr -> cbr -> cb -> cbl -> cfl -> cf---------------------------#
        #-------------------------------------------------------------------------------------#
        
        # cf -> cfr -> cbr -> cb -> cbl -> cfl -> cf
        # TODO: set as parameters
        '''
            - num_frames_short: 前后移动的短动作（例如向前推进、后退）持续 60 帧；
            - num_frames_long: 旋转、转弯动作持续 120 帧；
            - num_frames_all: 整体轨迹总帧数为 60*4 + 120*6 = 1080,其中 4 个短动作 + 6 个长动作
        
        '''
        
        num_frames_short = 60      # 短动作（例如前后移动）
        num_frames_long = 120      # 长动作（例如转弯或旋转）
        num_frames_all = 60 * 4 + 120 * 6  # 整个轨迹共 1080 帧
        
        t_short = torch.linspace(0, 1, num_frames_short, dtype=torch.float32, device=self.device)
        # 终点值是 1 - 1 / (num_frames_long + 1)，也就是稍微小于 1.
        # 是为了 避免在下一个插值段中重复计算终点帧，保持平滑衔接。
        t_long = torch.linspace(0, 1 - 1 / (num_frames_long + 1), num_frames_long, dtype=torch.float32, device=self.device)
        
        # obtain camera trajectories for each clip
        c2w_interp_forward0 = interpolate_extrinsics(c2w_cf, c2w_cf_forward, t_short) # from front to front-forward.
        c2w_interp_forward1 = interpolate_extrinsics(c2w_cf_forward, c2w_cf, t_short) # from front-forward to front.
        c2w_interp_0 = interpolate_extrinsics(c2w_cf, c2w_cfr, t_long)                # front to front-right 
        c2w_interp_1 = interpolate_extrinsics(c2w_cfr, c2w_cbr, t_long)               # front-right to bottom-right
        c2w_interp_2 = interpolate_extrinsics(c2w_cbr, c2w_cb, t_long)                # bottom-right to bottom
        c2w_interp_backward0 = interpolate_extrinsics(c2w_cb, c2w_cb_backward, t_short) # bottom to bottom-back
        c2w_interp_backward1 = interpolate_extrinsics(c2w_cb_backward, c2w_cb, t_short) # bottom-back to bottom
        c2w_interp_3 = interpolate_extrinsics(c2w_cb, c2w_cbl, t_long)      # bottom to bottom-left
        c2w_interp_4 = interpolate_extrinsics(c2w_cbl, c2w_cfl, t_long)     # bottom-left to forward-left
        c2w_interp_5 = interpolate_extrinsics(c2w_cfl, c2w_cf, t_long)      # forward-left to forward
        
        c2w_interp = torch.cat([c2w_interp_forward0, c2w_interp_forward1,
                                c2w_interp_0, c2w_interp_1, c2w_interp_2,
                                c2w_interp_backward0, c2w_interp_backward1,
                                c2w_interp_3, c2w_interp_4, c2w_interp_5], dim=1) # torch.Size([4, 960, 4, 4])
                
        fovxs_interp = data_dict["output_fovxs"][:, -6:-5].repeat(1, num_frames_all)   # [4,960] --> Center
        fovys_interp = data_dict["output_fovys"][:, -6:-5].repeat(1, num_frames_all)   # [4,960] --> Center
        
        
        # render_pkg_fuse 是调用 self.renderer.render(...) 函数的返回结果，
        # 本质上是一个字典 (dict)，它包含了渲染结果中的各种输出信息，主要包括:
        # (1) "image"
        # (2) "depth"
        # (3) "alpha"
        # (4) "visibility"
        # (5) "weights"
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=c2w_interp,
            fovx=fovxs_interp,
            fovy=fovys_interp,
            rays_o=None,
            rays_d=None
        )
        # dict_keys(['image', 'alpha', 'depth'])
        output_imgs = render_pkg_fuse["image"] # b v 3 h w  #torch.Size([4, 960, 3, 224, 400]) 
        output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w    --->  Metric Depths
        
        preds = {"img": output_imgs, "depth": output_depths}

        return preds, data_dict["bin_token"]

    def forward_demo_kitti360(self,batch,mode='s_center'):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"] #(1,2,3,H,W)        
        bs = img.shape[0] # batch size is 4
        
        img_feats = self.extract_img_feat(img=img, status="test")

        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        gaussians_pixel = sanitize_gaussians_tensor(gaussians_pixel)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1) #(1,N,14)

        bs = gaussians_all.shape[0]     
        
        
        
        if mode=='s_center':
        
            c2w_lf_left = data_dict["output_c2ws"][:, 1]
            c2w_lf_right = data_dict["output_c2ws"][:, 3]
            c2w_ff_left = data_dict["output_c2ws"][:, 0]
            c2w_ff_right = data_dict["output_c2ws"][:, 2]
            c2w_cf_left = data_dict["output_c2ws"][:, -2] #(1,2,4,4)
            c2w_cf_right = data_dict["output_c2ws"][:, -1] #(1,2,4,4)
            
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

            num_frames_all = 60 * c2w_interp.shape[1]
            fovxs_interp = data_dict["output_fovxs"][:, -2:-1].repeat(1, num_frames_all)
            fovys_interp = data_dict["output_fovys"][:, -2:-1].repeat(1, num_frames_all)
            
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
        
        elif mode =='s_first':
            
            c2w_lf_left = data_dict["output_c2ws"][:, 1]
            c2w_lf_right = data_dict["output_c2ws"][:, 3]
            c2w_ff_left = data_dict["output_c2ws"][:, 4]
            c2w_ff_right = data_dict["output_c2ws"][:, 5]
            c2w_cf_left = data_dict["output_c2ws"][:, 0] #(1,2,4,4)
            c2w_cf_right = data_dict["output_c2ws"][:, 2] #(1,2,4,4)
            
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

            num_frames_all = 60 * c2w_interp.shape[1]
            fovxs_interp = data_dict["output_fovxs"][:, -2:-1].repeat(1, num_frames_all)
            fovys_interp = data_dict["output_fovys"][:, -2:-1].repeat(1, num_frames_all)
            
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


        
        return preds, data_dict["bin_token"]

    def save_val_results(self, batch_gt, render_pkg_fuse, render_pkg_pixel, render_pkg_volume,
                         gaussians_all, gaussians_pixel, gaussians_volume, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        batch_size = render_pkg_fuse["image"].shape[0]
        
        # 18 views?
        n_rand_view = render_pkg_fuse["image"].shape[1]

        rgbs_gt = batch_gt["output_imgs"].cpu()
        depths_gt = batch_gt["output_depths"]
        depths_gt = (depths_gt / depths_gt.max()).unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        depths_m_gt = batch_gt["output_depths_m"]
        depths_m_gt = (depths_m_gt / 255.0).unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        confs_m_gt = batch_gt["output_confs_m"]
        confs_m_gt = confs_m_gt.unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        mask_dptm = batch_gt["mask_dptm"].unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()

        def save_vis(prefix, i, save_dir, n_rand_view, render_pkg, gaussians, rgbs_gt, depths_m_gt, mask_dptm, renderer):
            sample_save_dir = os.path.join(save_dir, f"sample-{i}-{prefix}")
            os.makedirs(sample_save_dir, exist_ok=True)

            for v in range(n_rand_view):
                rgb = render_pkg["image"][i, v].cpu()
                depth = render_pkg["depth"][i, v]
                h, w = depth.shape[1:]
                depth_abs = depth.repeat(3, 1, 1).cpu() / 255.0
                cat_gt = torch.cat(
                        [rgbs_gt[i, v], depths_m_gt[i, v], mask_dptm[i, v]],
                        dim=-1
                    )
                cat_pred = torch.cat(
                        [rgb, depth_abs, mask_dptm[i, v]], dim=-1
                    )
                grid = torch.cat(
                    [cat_gt, cat_pred], dim=1
                )
                grid = (grid.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite(os.path.join(sample_save_dir, f"{v}.png"), grid)
            if gaussians is not None:
                gs_save_path = os.path.join(sample_save_dir, f"sample-{i}-{prefix}.ply")
                gaussians_reformat = torch.cat([gaussians[i:i+1, :, 0:3],
                                                gaussians[i:i+1, :, 6:7],
                                                gaussians[i:i+1, :, 11:14],
                                                gaussians[i:i+1, :, 7:11],
                                                gaussians[i:i+1, :, 3:6]], dim=-1)
                renderer.save_ply(gaussians_reformat, gs_save_path)
                
                
        for i in range(batch_size):
            save_vis("omni", i, save_dir, n_rand_view, 
                                render_pkg_fuse, gaussians_all, 
                                rgbs_gt, depths_m_gt, 
                                mask_dptm, self.renderer)
        
        if render_pkg_pixel is not None:
            for i in range(batch_size):
                save_vis("pixel", i, save_dir, n_rand_view, render_pkg_pixel, 
                                    None, rgbs_gt, depths_m_gt, 
                                    mask_dptm, self.renderer)
        
        if render_pkg_volume is not None:
            for i in range(batch_size):
                save_vis("volume", i, save_dir, n_rand_view, render_pkg_volume, 
                                None, rgbs_gt, depths_m_gt, 
                                mask_dptm, self.renderer)
    
    # validations with bins tokens
    def validation_step_with_bin_tokens(self, batch, 
                                        val_result_savedir,
                                        bin_token_list,
                                        output_the_predicted_images=False
                                        ):
        
        # get batched gaussians repressions
        (loss_val, loss_term_val, render_pkg_fuse,
         render_pkg_pixel, render_pkg_volume, gaussians_all,
         gaussians_pixel, gaussians_volume, batch_data) = \
            self.forward(batch, "val")
        
        assert gaussians_all.shape[0] == len(bin_token_list)    
            
        (rendered_rgb_by_omni_batch,rendered_depth_by_omni_batch), (rendered_rgb_by_volume_batch,rendered_depth_by_volume_batch),(rendered_rgb_by_pixel_batch,rendered_depth_by_pixel_batch),rgb_gt,metric_v2_depth_batch = \
            self.save_val_results_with_bin_token_names(batch_data, render_pkg_fuse, render_pkg_pixel, render_pkg_volume,
                                gaussians_all, gaussians_pixel, gaussians_volume, val_result_savedir,bin_token_list,
                                output_the_predicted_images
                                )
        return loss_term_val, (rendered_rgb_by_omni_batch,rendered_depth_by_omni_batch), (rendered_rgb_by_volume_batch,rendered_depth_by_volume_batch),(rendered_rgb_by_pixel_batch,rendered_depth_by_pixel_batch),rgb_gt,metric_v2_depth_batch 

    def save_val_results_with_bin_token_names(self, batch_gt, render_pkg_fuse, render_pkg_pixel, render_pkg_volume,
                         gaussians_all, 
                         gaussians_pixel, 
                         gaussians_volume, 
                         save_dir,
                         bin_token_list, 
                         output_the_predicted_images=False
                         ):
        
        # # saved as a batch
        # os.makedirs(save_dir, exist_ok=True)
        
        # saved_omni_folder = os.path.join(save_dir,"omni")
        # saved_pixel_folder = os.path.join(save_dir,'pixel')
        # saved_volume_folder = os.path.join(save_dir,'volume')
        
        # os.makedirs(saved_omni_folder,exist_ok=True)
        # os.makedirs(saved_pixel_folder,exist_ok=True)
        # os.makedirs(saved_volume_folder,exist_ok=True)
        
        batch_size = render_pkg_fuse["image"].shape[0]
        
        # 18 views?
        n_rand_view = render_pkg_fuse["image"].shape[1]

        rgbs_gt = batch_gt["output_imgs"].cpu()
        depths_gt = batch_gt["output_depths"]
        depths_gt = (depths_gt / depths_gt.max()).unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        depths_m_gt = batch_gt["output_depths_m"]
        depths_m_gt = (depths_m_gt / 255.0).unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        confs_m_gt = batch_gt["output_confs_m"]
        confs_m_gt = confs_m_gt.unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        mask_dptm = batch_gt["mask_dptm"].unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        
        sparse_gt_depth_data = batch_gt['output_sparse_gt_depth'].unsqueeze(2)
        
        
        rendered_rgb_by_volume_batch = render_pkg_volume["image"]
        rendered_depth_by_volume_batch = render_pkg_volume['depth']
        
        rendered_rgb_by_pixel_batch = render_pkg_pixel['image']
        rendered_depth_by_pixel_batch = render_pkg_pixel['depth']
        
        
        rendered_rgb_by_omni_batch = render_pkg_fuse['image']
        rendered_depth_by_omni_batch = render_pkg_fuse['depth']
        
        metric_v2_depth_batch = batch_gt["output_imgs"]
        

        def save_vis(sample_save_dir, i,n_rand_view, render_pkg, gaussians, rgbs_gt, depths_m_gt, mask_dptm, renderer,
                     bin_token_name,
                     ):
            
            # sampled folder_names
            sample_save_dir = os.path.join(sample_save_dir, f"{bin_token_name}")
            os.makedirs(sample_save_dir, exist_ok=True)

            for v in range(n_rand_view):
                rgb = render_pkg["image"][i, v].cpu()
                depth = render_pkg["depth"][i, v]
                h, w = depth.shape[1:]
                depth_abs = depth.repeat(3, 1, 1).cpu() / 255.0
                cat_gt = torch.cat(
                        [rgbs_gt[i, v], depths_m_gt[i, v], mask_dptm[i, v]],
                        dim=-1
                    )
                cat_pred = torch.cat(
                        [rgb, depth_abs, mask_dptm[i, v]], dim=-1
                    )
                grid = torch.cat(
                    [cat_gt, cat_pred], dim=1
                )
                grid = (grid.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite(os.path.join(sample_save_dir, f"{v}.png"), grid)
            
            # saved the guassinas
            if gaussians is not None:
                gs_save_path = os.path.join(sample_save_dir, f"{bin_token_name}.ply")
                gaussians_reformat = torch.cat([gaussians[i:i+1, :, 0:3],
                                                gaussians[i:i+1, :, 6:7],
                                                gaussians[i:i+1, :, 11:14],
                                                gaussians[i:i+1, :, 7:11],
                                                gaussians[i:i+1, :, 3:6]], dim=-1)
                renderer.save_ply(gaussians_reformat, gs_save_path)
        
    
        # # visualizations
        # if output_the_predicted_images:
            
            
        #     for i in range(batch_size):
                
        #         current_bin_token_name = bin_token_list[i]
        #         save_vis(saved_omni_folder, i, n_rand_view, render_pkg_fuse, gaussians_all, rgbs_gt, depths_m_gt, mask_dptm, self.renderer,
        #                 current_bin_token_name
        #                 )
            
            
        #     if render_pkg_pixel is not None:
        #         for i in range(batch_size):
        #             current_bin_token_name = bin_token_list[i]
        #             save_vis(saved_pixel_folder, i, n_rand_view, render_pkg_pixel, None, rgbs_gt, depths_m_gt, mask_dptm, self.renderer,
        #                     current_bin_token_name
        #                     )
            
            
            
        #     if render_pkg_volume is not None:
        #         for i in range(batch_size):
        #             current_bin_token_name = bin_token_list[i]
        #             save_vis(saved_volume_folder, i, n_rand_view, render_pkg_volume, None, rgbs_gt, depths_m_gt, mask_dptm, self.renderer,
        #                     current_bin_token_name
        #                     )

        
        
        return (rendered_rgb_by_omni_batch,rendered_depth_by_omni_batch), (rendered_rgb_by_volume_batch,rendered_depth_by_volume_batch),(rendered_rgb_by_pixel_batch,rendered_depth_by_pixel_batch),rgbs_gt,sparse_gt_depth_data 
    
    def prepare_data_complete(self,batch):
        # ================== batch data process ================== #
        device_id = self.device
        data_dict = {}
        # for img feature extraction
        data_dict["imgs"] = batch["inputs"]["rgb"].to(device_id, dtype=self.dtype)
        # for pixel-gs
        rays_o = batch["inputs_pix"]["rays_o"].to(device_id, dtype=self.dtype)
        rays_d = batch["inputs_pix"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["rays_o"] = rays_o
        data_dict["rays_d"] = rays_d
        data_dict["pluckers"] = self.plucker_embedder(rays_o, rays_d)
        data_dict["fxs"] = batch["inputs_pix"]["fx"].to(device_id, dtype=self.dtype)
        data_dict["fys"] = batch["inputs_pix"]["fy"].to(device_id, dtype=self.dtype)
        data_dict["cxs"] = batch["inputs_pix"]["cx"].to(device_id, dtype=self.dtype)
        data_dict["cys"] = batch["inputs_pix"]["cy"].to(device_id, dtype=self.dtype)
        data_dict["c2ws"] = batch["inputs_pix"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["cks"] = batch["inputs_pix"]["ck"].to(device_id, dtype=self.dtype)
        data_dict["depths"] = batch["inputs_pix"]["depth_m"].to(device_id, dtype=self.dtype) # using metric depth.
        data_dict["confs"] = batch["inputs_pix"]["conf_m"].to(device_id, dtype=self.dtype)   # using confidence map.
        # for volume-gs
        img_metas = []
        bs, v, c, h, w = batch["inputs"]["rgb"].shape
        for w2i in batch["inputs_vol"]["w2i"]:
            img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
        data_dict["img_metas"] = img_metas
        
        
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
            
            
        data_dict['output_list'] = output_list
        data_dict["bin_token"] = batch["bin_token"]
        
        return data_dict
    
    def prepare_data_multi_views(self,batch):
        
        # input stereo images pairs
        input_image_index_selection = [0,3]
        
        # ================== batch data process ================== #
        device_id = self.device
        data_dict = {}
        # for img feature extraction
        data_dict["imgs"] = batch["inputs"]["rgb"].to(device_id, dtype=self.dtype)
        data_dict["imgs"] = data_dict["imgs"][:,input_image_index_selection,:,:,:]
        
        
        # for pixel-gs
        rays_o = batch["inputs_pix"]["rays_o"].to(device_id, dtype=self.dtype)
        rays_d = batch["inputs_pix"]["rays_d"].to(device_id, dtype=self.dtype)
        rays_o = rays_o[:,input_image_index_selection,:,:,:]
        rays_d = rays_d[:,input_image_index_selection,:,:,:]
        data_dict["rays_o"] = rays_o
        data_dict["rays_d"] = rays_d
        
        data_dict["pluckers"] = self.plucker_embedder(rays_o, rays_d)
        

        data_dict["fxs"] = batch["inputs_pix"]["fx"].to(device_id, dtype=self.dtype)
        data_dict["fys"] = batch["inputs_pix"]["fy"].to(device_id, dtype=self.dtype)
        data_dict["cxs"] = batch["inputs_pix"]["cx"].to(device_id, dtype=self.dtype)
        data_dict["cys"] = batch["inputs_pix"]["cy"].to(device_id, dtype=self.dtype)
        
        data_dict["fxs"] = data_dict["fxs"][:,input_image_index_selection]
        data_dict["fys"] = data_dict["fys"][:,input_image_index_selection]
        data_dict["cxs"] = data_dict["cxs"][:,input_image_index_selection]
        data_dict["cys"] = data_dict["cys"][:,input_image_index_selection]
        
        data_dict["c2ws"] = batch["inputs_pix"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["cks"] = batch["inputs_pix"]["ck"].to(device_id, dtype=self.dtype)
        data_dict["depths"] = batch["inputs_pix"]["depth_m"].to(device_id, dtype=self.dtype) # using metric depth.
        data_dict["confs"] = batch["inputs_pix"]["conf_m"].to(device_id, dtype=self.dtype)   # using confidence map.
        
        
        data_dict["c2ws"] = data_dict["c2ws"][:,input_image_index_selection,:,:]
        data_dict["cks"] = data_dict["cks"][:,input_image_index_selection,:,:]
        data_dict["depths"] = data_dict["depths"][:,input_image_index_selection,:,:]
        data_dict["confs"] = data_dict["confs"][:,input_image_index_selection,:,:]
        

        # for volume-gs
        img_metas = []
        bs, v, c, h, w = batch["inputs"]["rgb"].shape
        for w2i in batch["inputs_vol"]["w2i"]:
            w2i = w2i[input_image_index_selection,:,:]
            img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
        data_dict["img_metas"] = img_metas
        
                

        output_batch_dict = dict()
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


        data_dict["bin_token"] = batch["bin_token"]
        
        return data_dict,output_batch_dict
      
    def validation_complete_with_bin_tokens(self,
                                            batch,
                                            val_result_savedir,
                                            bin_token_list,
                                            cfg=None,
                                            vis=False
                                            ):
        
        bin_token_name = bin_token_list[0][:-4]
        
        data_dict,output_batch_dict = self.prepare_data_multi_views(batch=batch)

        img = data_dict['imgs']
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
        '''
        - torch.Size([2, 6, 128, 56, 100]) ---> 1/4
        - torch.Size([2, 6, 128, 28, 50])  ---> 1/8
        - torch.Size([2, 6, 128, 14, 25])  ---> 1/16
        - torch.Size([2, 6, 128, 7, 13])   ---> 1/32
        '''
        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # gaussians_feat: torch.Size([2, 537600, 128])
        # gaussians_pixel: torch.Size([2, 537600, 14])

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        # batch-wise saved the gaussain-pixel and the feature-pixel
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            # get the valid gaussains in the pixel splat
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            # get the valid feature in the pixel splat
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        
        # volume gs: input the features and gaussains pixel mask and guassain fature mask
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        # Make Sure the estimate gaussains are valid
        gaussians_pixel = sanitize_gaussians_tensor(gaussians_pixel)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)
        
        bs = gaussians_all.shape[0] # batch size is 2
        
        # doing rendering here
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        intrinsics = data_dict["cks"]
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"] # [B,6*3]
        render_fovys = output_batch_dict["output_fovys"] # [B,6*3]
        
        output_gt_images = output_batch_dict["output_imgs"]
        output_sparse_depth = output_batch_dict['output_sparse_depth']


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
        
        rendered_images_fusion = rendered_color_fuse #(1,6,3,H,W)
        rendered_depth_fusion = rendered_depth_fuse  #(1,V,H，W)
        rendered_images_gt = output_gt_images
        sparse_depth_gt = output_sparse_depth
        
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


        ''' The Evaluation of the RGB Metrics '''
        first_rgb_eval_info = metrics_mean(pred=rendered_images_first_stereo,
                                           gt=gt_images_first_stereo)
        first_rgb_lpips = first_rgb_eval_info['lpips']
        first_rgb_ssim = first_rgb_eval_info['ssim']
        first_rgb_psnr = first_rgb_eval_info['psnr']
        
        
        center_rgb_eval_info = metrics_mean(pred=rendered_images_center_stereo,
                                           gt=gt_images_center_stereo)
        center_rgb_lpips = center_rgb_eval_info['lpips']
        center_rgb_ssim = center_rgb_eval_info['ssim']
        center_rgb_psnr = center_rgb_eval_info['psnr']
        
        last_rgb_eval_info = metrics_mean(pred=rendered_images_last_stereo,
                                           gt=gt_images_last_stereo)
        last_rgb_lpips = last_rgb_eval_info['lpips']
        last_rgb_ssim = last_rgb_eval_info['ssim']
        last_rgb_psnr = last_rgb_eval_info['psnr']
        
        
        all_rgb_eval_info = metrics_mean(pred=rendered_images_all_stereo,
                                           gt=gt_images_all_stereo)
        all_rgb_lpips = all_rgb_eval_info['lpips']
        all_rgb_ssim = all_rgb_eval_info['ssim']
        all_rgb_psnr = all_rgb_eval_info['psnr']


        ''' The Evaluation of the Depth Metrics '''
        first_view_depth_eval_info = depth_metrics_absrel_sqrel_rmse_log(
                                                        pred=renderded_depth_first_stereo,
                                                         gt=gt_depth_first_stereo)
        
        frist_view_abs_rel = first_view_depth_eval_info['AbsRel']
        frist_view_sq_rel = first_view_depth_eval_info['SqRel']
        frist_view_rmse_log = first_view_depth_eval_info['RMSE_log']
        
        center_view_depth_eval_info = depth_metrics_absrel_sqrel_rmse_log(
                                                        pred=renderded_depth_center_stereo,
                                                         gt=gt_depth_center_stereo)
        center_view_abs_rel = center_view_depth_eval_info['AbsRel']
        center_view_sq_rel = center_view_depth_eval_info['SqRel']
        center_view_rmse_log = center_view_depth_eval_info['RMSE_log']
        
        last_view_depth_eval_info = depth_metrics_absrel_sqrel_rmse_log(
                                                        pred=renderded_depth_last_stereo,
                                                         gt=gt_depth_last_stereo)
        last_view_abs_rel = last_view_depth_eval_info['AbsRel']
        last_view_sq_rel = last_view_depth_eval_info['SqRel']
        last_view_rmse_log = last_view_depth_eval_info['RMSE_log']
        
        all_view_depth_eval_info = depth_metrics_absrel_sqrel_rmse_log(
                                                        pred=renderded_depth_all_stereo,
                                                         gt=gt_depth_all_stereo)
        all_view_abs_rel = all_view_depth_eval_info['AbsRel']
        all_view_sq_rel = all_view_depth_eval_info['SqRel']
        all_view_rmse_log = all_view_depth_eval_info['RMSE_log']
        
        
        
        evaluation_rgb_results_stat = {
            "first_view_psnr_average": first_rgb_psnr.data.item(),
            "first_view_ssim_average": first_rgb_ssim.data.item(),
            "first_view_lpips_average": first_rgb_lpips.data.item(),
            
            "center_view_psnr_average": center_rgb_psnr.data.item(),
            "center_view_ssim_average": center_rgb_ssim.data.item(),
            "center_view_lpips_average": center_rgb_lpips.data.item(),
            
            "last_view_psnr_average": last_rgb_psnr.data.item(),
            "last_view_ssim_average": last_rgb_ssim.data.item(),
            "last_view_lpips_average": last_rgb_lpips.data.item(),
            
            "all_view_psnr_average": all_rgb_psnr.data.item(),
            "all_view_ssim_average": all_rgb_ssim.data.item(),
            "all_view_lpips_average": all_rgb_lpips.data.item()
        }
        
        
        evaluation_depth_results_stat = {
            
            "first_view_Abs_Rel_average": frist_view_abs_rel.data.item(),
            "frist_view_Sq_Rel_average": frist_view_sq_rel.data.item(),
            "first_view_RMSE_log_average": frist_view_rmse_log.data.item(),
            
            "center_view_Abs_Rel_average": center_view_abs_rel.data.item(),
            "center_view_Sq_Rel_average": center_view_sq_rel.data.item(),
            "center_view_RMSE_log_average": center_view_rmse_log.data.item(),
            
            
            "last_view_Abs_Rel_average": last_view_abs_rel.data.item(),
            "last_view_Sq_Rel_average": last_view_sq_rel.data.item(),
            "last_view_RMSE_log_average": last_view_rmse_log.data.item(),
            
            "all_view_Abs_Rel_average": all_view_abs_rel.data.item(),
            "all_view_Sq_Rel_average": all_view_sq_rel.data.item(),
            "all_view_RMSE_log_average": all_view_rmse_log.data.item(),            
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
            
            preds, saved_video_name = self.forward_kitti360_videos(batch=batch)
            
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
    
    def forward_kitti360_videos(self,batch):
        
        data_dict,output_batch_dict = self.prepare_data_multi_views(batch=batch)
        img = data_dict["imgs"] #(1,2,3,H,W)       
        bs = img.shape[0] # batch size is 4
        img_feats = self.extract_img_feat(img=img, status="test")
        
        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        gaussians_pixel = sanitize_gaussians_tensor(gaussians_pixel)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1) #(1,N,14)
        bs = gaussians_all.shape[0]   
        
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        render_c2w = interleave_left_right_pose(render_c2w)

        
        
    
        '''Output C2W'''
        # last 
        c2w_lf_left = render_c2w[:,-4,:,:]
        c2w_lf_right = render_c2w[:,-3,:,:]
        # first
        c2w_ff_left = render_c2w[:,-2,:,:]
        c2w_ff_right = render_c2w[:,-1,:,:]
        # center
        c2w_cf_left = render_c2w[:,-6,:,:]
        c2w_cf_right = render_c2w[:,-5,:,:]

        
        
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


        return preds, data_dict["bin_token"]


    def bev_video_kitti360(self,batch,cfg=None,
                           rescale_h=3.0,rescale_w=1.0):
        
        data_dict,output_batch_dict = self.prepare_data_multi_views(batch=batch)
        img = data_dict['imgs']
        bs = img.shape[0]
        current_resolution = [img.shape[-2], img.shape[-1]]
        
        img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
        '''
        - torch.Size([2, 6, 128, 56, 100]) ---> 1/4
        - torch.Size([2, 6, 128, 28, 50])  ---> 1/8
        - torch.Size([2, 6, 128, 14, 25])  ---> 1/16
        - torch.Size([2, 6, 128, 7, 13])   ---> 1/32
        '''
        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # gaussians_feat: torch.Size([2, 537600, 128])
        # gaussians_pixel: torch.Size([2, 537600, 14])

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        # batch-wise saved the gaussain-pixel and the feature-pixel
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            # get the valid gaussains in the pixel splat
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            # get the valid feature in the pixel splat
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        
        # volume gs: input the features and gaussains pixel mask and guassain fature mask
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        # Make Sure the estimate gaussains are valid
        gaussians_pixel = sanitize_gaussians_tensor(gaussians_pixel)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)
        
        bs = gaussians_all.shape[0] # batch size is 2
        
        # doing rendering here
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        intrinsics = data_dict["cks"]
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



        rendered_c2w_center_bev_view1_movement1 = copy.deepcopy(center_view_c2w_left)
        rendered_c2w_center_bev_view1_movement1[0][0][2,3] = rendered_c2w_center_bev_view1_movement1[0][0][2,3] + 3
        
        
        rendered_c2w_center_bev_view1_movement2 = copy.deepcopy(rendered_c2w_center_bev_view1_movement1)
        rendered_c2w_center_bev_view1_movement2[0][0] = add_local_pitch(rendered_c2w_center_bev_view1_movement2[0][0], deg=-45.0)
        
        

        num_frames_short = 60
        t_short = torch.linspace(0, 1, num_frames_short, dtype=torch.float32, device=self.device)
        
        movement_0 = interpolate_extrinsics(center_view_c2w_left,
                                            rendered_c2w_center_bev_view1_movement1,
                                            t_short)
        movement_1 = interpolate_extrinsics(rendered_c2w_center_bev_view1_movement1,
                                            rendered_c2w_center_bev_view1_movement2,
                                            t_short)
        
        c2w_interp = torch.cat([movement_0[0], 
                                movement_1[0],
                                ], dim=1)
        
        
        N_Chunks = 10
        interval = int(c2w_interp.shape[1]//N_Chunks)
        
        rendered_rgb_list = []
        rendered_depth_list = []


        for idx in tqdm(range(N_Chunks)):
            
            rendered_bev_novel_views_c2w = c2w_interp[:,idx*interval:(idx+1)*interval,:]
            rendered_bev_fovxs = render_fovxs[:,0:1].repeat(1,rendered_bev_novel_views_c2w.shape[1])
            rendered_bev_fovys = render_fovys[:,0:1].repeat(1,rendered_bev_novel_views_c2w.shape[1])
            

            intrinsics_info = intrinsics[0,0,:,:]
            current_cx = intrinsics_info[0,2]
            current_cy = intrinsics_info[1,2]
            current_fx = intrinsics_info[0,0]
            current_fy = intrinsics_info[1,1]
            


            rendered_bev_fovxs = 2 * torch.arctan(rescale_w*current_cx  / current_fx).to(self.device).unsqueeze(0).repeat(1,rendered_bev_novel_views_c2w.shape[1])
            rendered_bev_fovys = 2 * torch.arctan(rescale_h*current_cy / current_fy).to(self.device).unsqueeze(0).repeat(1,rendered_bev_novel_views_c2w.shape[1])

            rendered_resolution = [int(current_resolution[0]*rescale_h), int(current_resolution[1]*rescale_w)]



            render_pkg_fuse = self.renderer.render_customized_resolution(
                gaussians=gaussians_all,
                c2w=rendered_bev_novel_views_c2w,
                fovx=rendered_bev_fovxs,
                fovy=rendered_bev_fovys,
                rays_o=None,
                rays_d=None,
                new_resolution=rendered_resolution 
            )

            rendered_results_fuse = render_pkg_fuse

            rendered_color_fuse = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
            rendered_depth_fuse = rendered_results_fuse['depth'] # torch.Size([1, V, 1, 224, 832])
            rendered_alpha_fuse = rendered_results_fuse['alpha'] # torch.Size([1, V, 1, 224, 832])
            rendered_depth_fuse = rendered_depth_fuse.squeeze(2)
            rendered_alpha_fuse = rendered_alpha_fuse.squeeze(2)
            
            
            rendered_color = rendered_color_fuse
            rendered_depth = rendered_depth_fuse


            
            rendered_color = torch.clamp(rendered_color,min=0,max=1.0)
            rendered_depth = torch.clamp(rendered_depth,min=0,max=150)

            rendered_color_fuse = rendered_color 
            rendered_depth_fuse = rendered_depth

            rendered_rgb_list.append(rendered_color)
            rendered_depth_list.append(rendered_depth)
            

        rendered_rgb_final = torch.cat(rendered_rgb_list,dim=1)
        rendered_depth_final = torch.cat(rendered_depth_list,dim=1)
        
        preds = {"img":rendered_rgb_final,"depth":rendered_depth_final}
        
        return preds


    def get_rgbs_bev_novel_view(self, batch, 
                                val_result_savedir,
                                bin_token_list,
                                cfg=None,
                                vis=False,
                                rescale_h=3.0,rescale_w=1.0):
        
        bin_token_name = bin_token_list[0][:-4]
        
        data_dict,output_batch_dict = self.prepare_data_multi_views(batch=batch)

        img = data_dict['imgs']
        bs = img.shape[0]
        
        
        current_resolution = [img.shape[-2], img.shape[-1]]
        
        img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
        '''
        - torch.Size([2, 6, 128, 56, 100]) ---> 1/4
        - torch.Size([2, 6, 128, 28, 50])  ---> 1/8
        - torch.Size([2, 6, 128, 14, 25])  ---> 1/16
        - torch.Size([2, 6, 128, 7, 13])   ---> 1/32
        '''
        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # gaussians_feat: torch.Size([2, 537600, 128])
        # gaussians_pixel: torch.Size([2, 537600, 14])

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        # batch-wise saved the gaussain-pixel and the feature-pixel
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            # get the valid gaussains in the pixel splat
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            # get the valid feature in the pixel splat
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        
        # volume gs: input the features and gaussains pixel mask and guassain fature mask
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        # Make Sure the estimate gaussains are valid
        gaussians_pixel = sanitize_gaussians_tensor(gaussians_pixel)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)
        
        bs = gaussians_all.shape[0] # batch size is 2
        
        # doing rendering here
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        intrinsics = data_dict["cks"]
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


        

        intrinsics_info = intrinsics[0,0,:,:]
        current_cx = intrinsics_info[0,2]
        current_cy = intrinsics_info[1,2]
        current_fx = intrinsics_info[0,0]
        current_fy = intrinsics_info[1,1]
        


        rendered_bev_fovxs = 2 * torch.arctan(rescale_w*current_cx  / current_fx).to(self.device).unsqueeze(0).repeat(1,rendered_bev_novel_views_c2w.shape[1])
        rendered_bev_fovys = 2 * torch.arctan(rescale_h*current_cy / current_fy).to(self.device).unsqueeze(0).repeat(1,rendered_bev_novel_views_c2w.shape[1])

        rendered_resolution = [int(current_resolution[0]*rescale_h), int(current_resolution[1]*rescale_w)]
        
        render_pkg_fuse = self.renderer.render_customized_resolution(
            gaussians=gaussians_all,
            c2w=rendered_bev_novel_views_c2w,
            fovx=rendered_bev_fovxs,
            fovy=rendered_bev_fovys,
            rays_o=None,
            rays_d=None,
            new_resolution=rendered_resolution 
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


            rendered_interploated_images_folder_path = os.path.join(saved_folder_for_visualization,
                                                              'rendered_interploated_images_views')
            rendered_interploated_depth_folder_path = os.path.join(saved_folder_for_visualization,
                                                              'rendered_interploated_depth_views')
            os.makedirs(rendered_interploated_images_folder_path,exist_ok=True)
            os.makedirs(rendered_interploated_depth_folder_path,exist_ok=True)

            
            
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


            # Make videos
            video_preds = self.bev_video_kitti360(batch=batch,cfg=cfg,
                                                  rescale_h=rescale_h,rescale_w=rescale_w)
            
            bs = video_preds["img"].shape[0]  
            pred_imgs = video_preds["img"] #(4,960,3,224,400)
            pred_depths = video_preds["depth"] #(4,960,3,224,400)
            
            
            # saved the results with batch
            for b in range(bs):
                # dump rgb view
                dump_path = osp.join(rendered_interploated_images_folder_path, "rgb.mp4")
                video = (pred_imgs[b].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
                video_rec = wandb.Video(video[None], fps=30, format="mp4")
                video_tensor = video_rec._prepare_video(video_rec.data)
                clip = mpy.ImageSequenceClip(list(video_tensor), fps=30)
                clip.write_videofile(dump_path, codec='libx264', preset='medium', logger=None)
                
                # dump depth view
                dump_path_dpt = osp.join(rendered_interploated_depth_folder_path, "depth.mp4")
                pred_depth = pred_depths[b].clamp(0.0, 100.0)
                max_val = float(pred_depth.max())
                video_dpt = depths_to_colors(pred_depths[b], concat="frame", max_val=max_val)
                video_dpt = video_dpt.transpose((0, 3, 1, 2))
                video_rec_dpt = wandb.Video(video_dpt[None], fps=30, format="mp4")
                video_tensor_dpt = video_rec_dpt._prepare_video(video_rec_dpt.data)
                clip_dpt = mpy.ImageSequenceClip(list(video_tensor_dpt), fps=30)
                clip_dpt.write_videofile(dump_path_dpt, codec='libx264', preset='medium', logger=None)






@torch.no_grad()
def lpips_mean(pred: torch.Tensor, gt: torch.Tensor, net: str = "alex") -> torch.Tensor:
    """
    pred, gt: [1, V, 3, H, W], float in [0, 1]
    returns: scalar tensor (mean LPIPS over V)
    """
    assert pred.shape == gt.shape
    assert pred.ndim == 5 and pred.shape[0] == 1 and pred.shape[2] == 3

    device = pred.device
    loss_fn = lpips.LPIPS(net=net).to(device).eval()

    B, V, C, H, W = pred.shape  # B=1
    pred_ = pred.view(B * V, C, H, W) * 2.0 - 1.0
    gt_   = gt.view(B * V, C, H, W) * 2.0 - 1.0

    d = loss_fn(pred_, gt_)          # [B*V, 1, 1, 1]
    return d.mean()       


@torch.no_grad()
def metrics_mean(pred: torch.Tensor, gt: torch.Tensor, lpips_net: str = "alex"):
    assert pred.shape == gt.shape
    assert pred.ndim == 5 and pred.shape[0] == 1 and pred.shape[2] == 3

    device = pred.device
    B, V, C, H, W = pred.shape
    N = B * V

    pred_01 = pred.view(N, C, H, W).clamp(0.0, 1.0)
    gt_01   = gt.view(N, C, H, W).clamp(0.0, 1.0)

    # PSNR
    mse = (pred_01 - gt_01).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    psnr = (10.0 * torch.log10(1.0 / mse)).mean()

    # SSIM (torchmetrics expects data_range)
    ssim = ssim_fn(pred_01, gt_01, data_range=1.0).mean()

    # LPIPS
    loss_fn = lpips.LPIPS(net=lpips_net).to(device).eval()
    lp = loss_fn(pred_01 * 2 - 1, gt_01 * 2 - 1).mean()

    return {"lpips": lp, "psnr": psnr, "ssim": ssim}


@torch.no_grad()
def depth_metrics_absrel_sqrel_rmse_log(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    per_view: bool = False,
):
    """
    pred: [B, V, H, W] estimated depth
    gt:   [B, V, H, W] ground-truth depth
    valid_mask (optional): [B, V, H, W] boolean mask where True means valid
    eps: clamp min value for numerical stability
    per_view: if True, return metrics per (B,V); else return scalar over all valid pixels
    """
    assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
    assert pred.ndim == 4, f"Expected [B,V,H,W], got {pred.shape}"

    # Basic validity: gt > 0 and finite
    valid = (gt > 0) & torch.isfinite(gt) & torch.isfinite(pred)
    if valid_mask is not None:
        valid = valid & valid_mask.bool()

    # Clamp to avoid division by zero and log(0)
    pred_c = pred.clamp(min=eps)
    gt_c   = gt.clamp(min=eps)

    if per_view:
        # Compute per (B,V): average over H,W only on valid pixels
        B, V, H, W = pred.shape
        valid_f = valid.view(B, V, -1)
        pred_f  = pred_c.view(B, V, -1)
        gt_f    = gt_c.view(B, V, -1)

        # counts per (B,V) for safe division
        cnt = valid_f.sum(dim=-1).clamp(min=1)

        diff = pred_f - gt_f
        absrel = (diff.abs() / gt_f).masked_fill(~valid_f, 0).sum(dim=-1) / cnt
        sqrel  = (diff.pow(2) / gt_f).masked_fill(~valid_f, 0).sum(dim=-1) / cnt
        rmse_log = ((torch.log(pred_f) - torch.log(gt_f)).pow(2)).masked_fill(~valid_f, 0).sum(dim=-1) / cnt
        rmse_log = torch.sqrt(rmse_log)

        return {
            "AbsRel": absrel,       # [B, V]
            "SqRel": sqrel,         # [B, V]
            "RMSE_log": rmse_log,   # [B, V]
            "valid_count": cnt,     # [B, V]
        }

    else:
        # Scalar over all valid pixels (across B,V,H,W)
        diff = pred_c - gt_c
        v = valid

        absrel = (diff.abs() / gt_c)[v].mean()
        sqrel  = (diff.pow(2) / gt_c)[v].mean()
        rmse_log = (torch.log(pred_c) - torch.log(gt_c)).pow(2)[v].mean().sqrt()

        return {
            "AbsRel": absrel,     # scalar tensor
            "SqRel": sqrel,       # scalar tensor
            "RMSE_log": rmse_log  # scalar tensor
        }