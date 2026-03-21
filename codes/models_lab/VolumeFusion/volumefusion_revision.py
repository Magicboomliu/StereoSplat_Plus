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

import lpips
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn



def build_w2i_from_c2w(cam2world: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Args:
        cam2world: [B, V, 4, 4]  standard camera-to-world matrix
        K:         [B, V, 3, 3]  intrinsic matrix

    Returns:
        w2i:       [B, V, 4, 4]

    This matches the *effective* logic of your original code.
    """
    assert cam2world.ndim == 4 and cam2world.shape[-2:] == (4, 4)
    assert K.ndim == 4 and K.shape[-2:] == (3, 3)
    assert cam2world.shape[:2] == K.shape[:2]

    B, V = cam2world.shape[:2]
    device = cam2world.device
    dtype = cam2world.dtype

    # standard world-to-camera
    w2c = torch.linalg.inv(cam2world)   # [B, V, 4, 4]

    # pad K to 4x4
    K_pad = torch.eye(4, device=device, dtype=dtype)[None, None].repeat(B, V, 1, 1)
    K_pad[:, :, :3, :3] = K.to(dtype)

    # correct result
    w2i = K_pad @ w2c
    return w2i

def compute_psnr_ssim_batch(est, gt, eps=1e-8):
    """
    est, gt: [B, V, C, H, W], values in [0, 1]
    """
    assert est.shape == gt.shape
    assert est.ndim == 5

    B, V, C, H, W = est.shape

    # PSNR
    mse = torch.mean((est - gt) ** 2, dim=(2, 3, 4))   # [B, V]
    psnr = -10.0 * torch.log10(mse + eps)              # [B, V]

    # SSIM
    est_ = est.reshape(B * V, C, H, W)
    gt_  = gt.reshape(B * V, C, H, W)
    ssim = ssim_fn(
        est_, gt_, data_range=1.0, reduction='none'
    ).reshape(B, V)

    return {
        "psnr_per_view": psnr,
        "psnr_mean": psnr.mean(),
        "ssim_per_view": ssim,
        "ssim_mean": ssim.mean(),
    }


def visualize_mask_heatmap(
    mask,
    save_path=None,
    title="mask (0~1)",
    cmap="coolwarm",
    vmin=0.6,
    vmax=1.0,
    interpolation="nearest",
    figsize=(8, 6),
    colorbar_label="value",
    show=False,
):
    """
    将 0~1 的 mask 画成热力图并带 colorbar（默认 coolwarm，显示范围 [0.6, 1.0]）。

    Args:
        mask: torch.Tensor 或 np.ndarray，形状可为 [H,W]、[1,H,W]、[B,1,H,W] 等，取第一个 batch/通道展平为 2D。
        save_path: 若给定则保存到该路径（建议 .png）。
        title: 图标题。
        cmap: matplotlib colormap 名称，默认 coolwarm。
        vmin, vmax: colorbar / imshow 映射范围，默认 0.6~1.0（低于 vmin 会钳到 colormap 低端）。
        interpolation: imshow 插值，'nearest' 或 'bilinear'。
        figsize: 图像尺寸。
        colorbar_label: colorbar 标签。
        show: 是否 plt.show()。

    Returns:
        fig, ax: matplotlib 对象（便于继续改图）。
    """
    x = mask
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"mask 应为可规约为 [H,W] 的张量，当前 shape={np.asarray(mask).shape}")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


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


    def prepare_tripleview_by_ratio_index(self,
                                batch,
                                pseudo_ratio_index):
        
        device_id = self.device
        input_batch_dict = dict()
        input_batch_dict_build = dict()
        
        output_batch_dict = dict()                                        
        bin_token_name = batch['bin_token']
                                        
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
        
        
         # create input data from the output source
        input_camera_cks = batch['inputs_pix']['ck'][:,0:1,:,:]
        temporary_images_all =  interleave_left_right(output_batch_dict["output_imgs"])
        temporary_pose_all = interleave_left_right_pose(output_batch_dict["output_c2ws"])
        temporary_depth_all = interleave_left_right_depth(output_batch_dict["output_depths"])
        temporary_sparse_depth_all = interleave_left_right_depth(output_batch_dict["output_sparse_depth"])
        temporary_intrinsic_all = input_camera_cks.repeat(1,temporary_images_all.shape[1],1,1)
        
        
        # get all first frame information
        first_frames_images = temporary_images_all[:,-2:,:,:,:]
        first_frames_extrinsics = temporary_pose_all[:,-2:,:,:]
        first_frames_depths = temporary_depth_all[:,-2:,:,:]
        first_frames_sparse_depths = temporary_sparse_depth_all[:,-2:,:,:]
        first_frames_intrinsics = temporary_intrinsic_all[:,-2:,:,:]
        
        
        # get all center frame information
        center_frames_images = temporary_images_all[:,-6:-4,:,:,:]
        center_frames_extrinsics = temporary_pose_all[:,-6:-4,:,:]
        center_frames_depths = temporary_depth_all[:,-6:-4,:,:]
        center_frames_sparse_depths = temporary_sparse_depth_all[:,-6:-4,:,:]
        center_frames_intrinsics = temporary_intrinsic_all[:,-6:-4,:,:]
        
    
        # get all last frame information
        last_frames_images = temporary_images_all[:,-4:-2,:,:,:]
        last_frames_extrinsics = temporary_pose_all[:,-4:-2,:,:]
        last_frames_depths = temporary_depth_all[:,-4:-2,:,:]
        last_frames_sparse_depths = temporary_sparse_depth_all[:,-4:-2,:,:]
        last_frames_intrinsics = temporary_intrinsic_all[:,-4:-2,:,:]
        
        
        if pseudo_ratio_index[0] == 0.5 and pseudo_ratio_index[1] == 1.0:
            input_rgb_current = torch.cat((first_frames_images,center_frames_images,last_frames_images),dim=1)
            input_camera_intrinsics_current = torch.cat((first_frames_intrinsics,center_frames_intrinsics,last_frames_intrinsics),dim=1)
            input_camera_extrinsics_current = torch.cat((first_frames_extrinsics,center_frames_extrinsics,last_frames_extrinsics),dim=1)
            input_psuedo_depth_current = torch.cat((first_frames_depths,center_frames_depths,last_frames_depths),dim=1)
            input_sparse_depth_current = torch.cat((first_frames_sparse_depths,center_frames_sparse_depths,last_frames_sparse_depths),dim=1)
            
            
        else:
            rest_images = temporary_images_all[:,:-6,:,:,:]
            rest_extrinsics = temporary_pose_all[:,:-6,:,:]
            rest_depths = temporary_depth_all[:,:-6,:,:]
            rest_sparse_depths = temporary_sparse_depth_all[:,:-6,:,:]
            rest_intrinsics = temporary_intrinsic_all[:,:-6,:,:]
            
            rest_images = torch.cat([first_frames_images,rest_images,last_frames_images],dim=1)
            rest_extrinsics = torch.cat([first_frames_extrinsics,rest_extrinsics,last_frames_extrinsics],dim=1)
            rest_depths = torch.cat([first_frames_depths,rest_depths,last_frames_depths],dim=1)
            rest_sparse_depths = torch.cat([first_frames_sparse_depths,rest_sparse_depths,last_frames_sparse_depths],dim=1)
            rest_intrinsics = torch.cat([first_frames_intrinsics,rest_intrinsics,last_frames_intrinsics],dim=1)
            
            
            
            rest_stereo_pairs_nums = rest_images.shape[1]//2
            
            
            # odd is the left and even in the right
            
            second_frame_left_id = int(rest_stereo_pairs_nums * pseudo_ratio_index[0]) * 2.0
            second_frame_right_id = second_frame_left_id + 1.0
            
            third_frame_left_id = int(rest_stereo_pairs_nums * pseudo_ratio_index[1]) *2.0
            third_frame_right_id = third_frame_left_id + 1.0
            
            
            second_frame_left_id = int(second_frame_left_id)
            second_frame_right_id = int(second_frame_right_id)
            third_frame_left_id = int(third_frame_left_id)
            third_frame_right_id = int(third_frame_right_id)
            
            
            second_input_rgb = rest_images[:,second_frame_left_id:second_frame_right_id+1,:,:,:]
            second_input_extrinsics = rest_extrinsics[:,second_frame_left_id:second_frame_right_id+1,:,:]
            second_input_depths = rest_depths[:,second_frame_left_id:second_frame_right_id+1,:,:]
            second_input_sparse_depths = rest_sparse_depths[:,second_frame_left_id:second_frame_right_id+1,:,:]
            second_input_intrinsics = rest_intrinsics[:,second_frame_left_id:second_frame_right_id+1,:,:]

            third_input_rgb = rest_images[:,third_frame_left_id:third_frame_right_id+1,:,:,:]
            third_input_extrinsics = rest_extrinsics[:,third_frame_left_id:third_frame_right_id+1,:,:]
            third_input_depths = rest_depths[:,third_frame_left_id:third_frame_right_id+1,:,:]
            third_input_sparse_depths = rest_sparse_depths[:,third_frame_left_id:third_frame_right_id+1,:,:]
            third_input_intrinsics = rest_intrinsics[:,third_frame_left_id:third_frame_right_id+1,:,:]
            
            
            # update the input data
            
            input_rgb_current = torch.cat((first_frames_images,second_input_rgb,third_input_rgb),dim=1)
            input_camera_extrinsics_current = torch.cat((first_frames_extrinsics,second_input_extrinsics,third_input_extrinsics),dim=1)
            input_psuedo_depth_current = torch.cat((first_frames_depths,second_input_depths,third_input_depths),dim=1)
            input_sparse_depth_current = torch.cat((first_frames_sparse_depths,second_input_sparse_depths,third_input_sparse_depths),dim=1)
            input_camera_intrinsics_current = torch.cat((first_frames_intrinsics,second_input_intrinsics,third_input_intrinsics),dim=1)
            
            

        
        selected_cam_dist_index_current = self.k_nearest_camera_indices(
                                            extrinsics=input_camera_extrinsics_current,
                                            K=4,pose_type="cam2world",
                                            include_self=True)
        
        current_w2i_build = build_w2i_from_c2w(input_camera_extrinsics_current,
                                               input_camera_intrinsics_current)
        current_input_rgbs_build = copy.deepcopy(input_rgb_current)

            
        # for volume-gs
        img_metas_build = []
        bs, v, c, h, w = current_input_rgbs_build.shape
        for w2i in current_w2i_build:
            img_metas_build.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
        
        input_batch_dict_build['imgs'] = current_input_rgbs_build.to(device_id, dtype=self.dtype)
        input_batch_dict_build['intrinsics'] = input_camera_intrinsics_current.to(device_id, dtype=self.dtype)
        input_batch_dict_build['extrinsics'] = input_camera_extrinsics_current.to(device_id, dtype=self.dtype)
        input_batch_dict_build['nn_matrix'] = selected_cam_dist_index_current.to(device_id, dtype=self.dtype)
        input_batch_dict_build['pseudo_depths'] = input_psuedo_depth_current.to(device_id, dtype=self.dtype)
        input_batch_dict_build['sparse_depths'] = input_sparse_depth_current.to(device_id, dtype=self.dtype)
        input_batch_dict_build['bin_token_name'] = bin_token_name
        input_batch_dict_build['img_metas'] = img_metas_build
        
    

        return input_batch_dict_build,output_batch_dict


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
                                        vis=False):
        
        bin_token_name = bin_token_list[0][:-4]
        
        # loss and loss terms
        with torch.no_grad():
            loss, loss_terms,rendered_fusion_list,\
                rendered_volume_list,rendered_cv_results_list, \
                    predicted_input_depth,input_sparse_gt_depth,\
                        output_rgb,sparse_depth_gt,input_images = self.forward(
                                                            batch,
                                                            mode='val',
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
        
        
        # evaluate the rgb metrics
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
        
        # evaluate the depth metrics
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
    
    
    def save_3dgs_ply(self,
                    batch,
                    val_result_savedir,
                    bin_token_list,
                    view_num=2,
                    matching_nums=2,
                    cfg=None,
                    vis=False,
                      ):
        bin_token_name = bin_token_list[0][:-4]
        

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
        
        saved_ply_name = bin_token_name + ".ply"

        saved_ply_path = os.path.join(val_result_savedir, saved_ply_name,"estimated_3dgs.ply")
        os.makedirs(os.path.dirname(saved_ply_path), exist_ok=True)
        
        self.renderer.save_ply(gaussians_all, saved_ply_path)
    
    
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
        
        
        
        c2w_ff_left = render_c2w[:,-2,:,:]  # first frame left
        c2w_ff_right = render_c2w[:,-1,:,:] # first frame right
        c2w_cf_left = render_c2w[:,-6,:,:]  # center frame left
        c2w_cf_right = render_c2w[:,-5,:,:] # center frame right
        c2w_lf_left = render_c2w[:,-4,:,:]  # last frame left
        c2w_lf_right = render_c2w[:,-3,:,:] # last frame right
        
        
        
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
    
    # for inside-bin fusion
    def process_inside_bin_fusion(self,batch):

        device_id = self.device
        
        input_batch_dict_list = []
        
        
        output_batch_dict = dict()

        bin_token_name = batch['bin_token']
        input_cam_batch_data_list = batch['inputs_pix']                                 
        input_batch_data_list = batch['inputs']
        
        input_rgb_list = input_batch_data_list['rgb']
        input_camera_intrinsics_list = input_cam_batch_data_list['ck']
        input_camera_extrinsics_list = input_cam_batch_data_list['c2w']
        input_camera_ego_extrinsics_list = input_cam_batch_data_list['c2w_ego']
        input_psuedo_depth_list = input_cam_batch_data_list['depth_m']
        input_sparse_depth_list = input_cam_batch_data_list['sparse_gt_depth']
        
        
        for internal_input_frame_idx in range(len(input_sparse_depth_list)):
            
            input_sparse_depth = input_sparse_depth_list[internal_input_frame_idx]
            input_rgb = input_rgb_list[internal_input_frame_idx]
            input_camera_intrinsics = input_camera_intrinsics_list[internal_input_frame_idx]
            input_camera_extrinsics = input_camera_extrinsics_list[internal_input_frame_idx]
            input_camera_ego_extrinsics = input_camera_ego_extrinsics_list[internal_input_frame_idx]
            input_psuedo_depth = input_psuedo_depth_list[internal_input_frame_idx]
            input_sparse_depth = input_sparse_depth_list[internal_input_frame_idx]
            
            input_batch_dict = dict()
            cameras_dist_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # [2, 2] [V,K]
            cameras_dist_index= cameras_dist_index.unsqueeze(0).repeat(input_sparse_depth.shape[0],1,1)
        
        
            # input_dict
            input_batch_dict['imgs'] = input_rgb.to(device_id, dtype=self.dtype)
            input_batch_dict['intrinsics'] = input_camera_intrinsics.to(device_id, dtype=self.dtype)
            
            # using the ego-c2w
            input_batch_dict['extrinsics'] = input_camera_ego_extrinsics.to(device_id, dtype=self.dtype)
            #input_batch_dict['extrinsics'] = input_camera_extrinsics.to(device_id, dtype=self.dtype)
            
            
            input_batch_dict['nn_matrix'] =cameras_dist_index.to(device_id, dtype=self.dtype)
            input_batch_dict['pseudo_depths'] = input_psuedo_depth.to(device_id, dtype=self.dtype)
            input_batch_dict['sparse_depths'] = input_sparse_depth.to(device_id, dtype=self.dtype)
            
            #using the c2w 
            input_batch_dict['extrinsics_true'] = input_camera_extrinsics.to(device_id, dtype=self.dtype)
            
        
            img_metas = []
            bs, v, c, h, w = input_rgb.shape
            for w2i in batch["inputs_vol"]["w2i_ego"][internal_input_frame_idx]:
                img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
            input_batch_dict["img_metas"] = img_metas
                 
            input_batch_dict_list.append(input_batch_dict)

            # img_metas = []
            # bs, v, c, h, w = input_rgb.shape
            # for w2i in batch["inputs_vol"]["w2i"][internal_input_frame_idx]:
            #     img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
            # input_batch_dict["img_metas"] = img_metas
                 
            # input_batch_dict_list.append(input_batch_dict)
       

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
        output_batch_dict['bin_token'] = bin_token_name
        
        return input_batch_dict_list,output_batch_dict


    def filter_points_visible_in_either_view(self,
                                             points, 
                                             w2i_left, 
                                             w2i_right, 
                                             H, 
                                             W):
        """
        输入:
            points: [N, 3] float
            w2i_left, w2i_right: [4, 4] float
            H, W: 图像尺寸
        返回:
            filtered_points: [N_visible, 3]
            mask: [N]，bool类型，表示保留的点
        """
        N = points.shape[0]
        device = points.device

        # 齐次变换
        points_h = torch.cat([points, torch.ones((N,1), device=device)], dim=-1)  # [N,4]

        def project_and_check(points_h, w2i):
            cam = (points_h @ w2i.T)  # [N, 4] @ [4,4]ᵗ → [N, 4]
            x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]

            # 防止除0
            z = z + 1e-6
            u = x / z
            v = y / z

            valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
            return valid

        valid_left = project_and_check(points_h, w2i_left)
        valid_right = project_and_check(points_h, w2i_right)

        # 至少落在一张图上
        visible_mask = valid_left | valid_right

        return visible_mask

    
    def validataion_inside_fusion(self,batch,saved_dir,
                                  saved_label,
                                  cfg=None):
        
        input_batch_dict_list,output_batch_dict = self.process_inside_bin_fusion(batch=batch)
        
        
        output_info_list_all = output_batch_dict['output_list']
        
        gaussians_all_list = []
        
        # debug here
        gaussain_cv_list = []
        gaussain_volume_list = []
        pred_depths_list = []
        
        for internal_frame_index in range(len(input_batch_dict_list)):
            input_batch_dict = input_batch_dict_list[internal_frame_index]
        
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
                    gaussians_cv_mask,
                    gaussians_feat_mask,
                    input_batch_dict["img_metas"])
            

     
            
            # Make Sure the estimate gaussains are valid
            gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
            gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)


            current_w2i = input_batch_dict['img_metas'][0]['lidar2img']
            left_current_w2i = current_w2i[0]
            right_current_w2i = current_w2i[1]
            current_img_shape = input_batch_dict['img_metas'][0]['img_shape'][0]
            gaussians_volume_mask = self.filter_points_visible_in_either_view(points=gaussians_volume[...,:3].squeeze(0),
                                                      w2i_left=left_current_w2i,
                                                      w2i_right=right_current_w2i,
                                                      H=current_img_shape[0],
                                                      W=current_img_shape[1])
            
            gaussians_volume = gaussians_volume[0][gaussians_volume_mask]
            gaussians_volume = gaussians_volume.unsqueeze(0)
            
            
            
            gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
            bs = gaussians_all.shape[0] # batch size is 2
            
            gaussians_all_list.append(gaussians_all)
            gaussain_cv_list.append(gaussians_cv)
            gaussain_volume_list.append(gaussians_volume)
            pred_depths_list.append(pred_depths)
        
        
        output_info_list_all = output_batch_dict['output_list']
        output_rendered_pkg_fuse_list = []
        # for the validation
        rendered_left_images_list = []
        rendered_right_images_list = []
        gt_left_images_list = []
        gt_right_images_list = []
        
        rendered_left_depth_list = []
        rendered_right_depth_list = []
        
        gt_left_depth_list = []
        gt_right_depth_list = []
        
        
        left_psnr_list = []
        left_ssim_list = []
        right_psnr_list = []
        right_ssim_list = []
        
        left_depth_mae_list = []
        left_depth_mse_list = []
        
        right_depth_mae_list = []
        right_depth_mse_list = []
        
        if saved_label:
            saved_bin_token_name =output_batch_dict['bin_token'][0][:-4]
            current_saved_bin_folder = os.path.join(saved_dir,saved_bin_token_name)
            os.makedirs(current_saved_bin_folder,exist_ok=True)
        
        left_cam_to_lidar_matrix = output_info_list_all[0]['output_c2ws'][0][0]
        
        # gaussians_all = gaussians_all_list[1]


 
        g0,g2_trans= transform_g2_to_g1(gaussians_all_list[0],
                                        gaussians_all_list[2],
                                        output_info_list_all[2]["output_c2ws"][0][0]@torch.linalg.inv(left_cam_to_lidar_matrix) ,
                                     )

        _,g1_trans= transform_g2_to_g1(gaussians_all_list[0],
                                        gaussians_all_list[1],
                                        output_info_list_all[1]["output_c2ws"][0][0]@torch.linalg.inv(left_cam_to_lidar_matrix) ,
                                     )



        # gaussians_all =g2_trans
        gaussians_all =  g2_trans
        
        
        for internal_view_index, output_dict_info_temp in enumerate(output_info_list_all):
            render_c2w = output_dict_info_temp["output_c2ws"] # render last and first camera 2 word: [B,6*3,4,4]
            render_fovxs = output_dict_info_temp["output_fovxs"] # [B,6*3]
            render_fovys = output_dict_info_temp["output_fovys"] # [B,6*3]
            
            gt_RGB = output_dict_info_temp["output_imgs"]
            gt_sparse_depth = output_dict_info_temp['output_sparse_gt_depth']
            
            render_pkg_fuse = self.renderer.render(
                gaussians=gaussians_all,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )     
            output_rendered_pkg_fuse_list.append(render_pkg_fuse)
            
            rendered_RGB = render_pkg_fuse['image'] #(B,V,3,H,W)
            rendered_depth = render_pkg_fuse['depth'].squeeze(2)
            
            rendered_left_rgb = rendered_RGB[:,0,:,:,:]
            gt_left_rgb = gt_RGB[:,0,:,:,:]
            
            rendered_right_rgb = rendered_RGB[:,1,:,:,:]
            gt_right_rgb = gt_RGB[:,1,:,:,:]
            
            rendered_left_depth = rendered_depth[:,0,:,:]
            gt_left_depth = gt_sparse_depth[:,0,:,:]
            

            rendered_right_depth = rendered_depth[:,1,:,:]
            gt_right_depth = gt_sparse_depth[:,1,:,:]
            
            
            left_psnr, left_ssim = compute_psnr_ssim(pred=rendered_left_rgb,
                                                     target=gt_left_rgb)
            
            right_psnr, right_ssim = compute_psnr_ssim(pred=rendered_right_rgb,
                                                       target=gt_right_rgb)
            
            left_mae, left_mse = compute_depth_mae_mse(depth_pred=rendered_left_depth,
                                                       depth_gt=gt_left_depth)
            
            right_mae, right_mse = compute_depth_mae_mse(depth_pred=rendered_right_depth,
                                                         depth_gt=gt_right_depth)

            left_psnr_list.append(left_psnr)
            left_ssim_list.append(left_ssim)
            
            right_psnr_list.append(right_psnr)
            right_ssim_list.append(right_ssim)
            
            left_depth_mae_list.append(left_mae)
            left_depth_mse_list.append(left_mse)
            
            right_depth_mae_list.append(right_mae)
            right_depth_mse_list.append(right_mse)
            
            rendered_left_images_list.append(rendered_left_rgb)
            rendered_right_images_list.append(rendered_right_rgb)
            
            gt_left_images_list.append(gt_left_rgb)
            gt_right_images_list.append(gt_right_rgb)
            
            rendered_left_depth_list.append(rendered_left_depth)
            rendered_right_depth_list.append(rendered_right_depth)
            
            gt_left_depth_list.append(gt_left_depth)
            gt_right_depth_list.append(gt_right_depth)


            rendered_left_images_vis = torch.cat([rendered_left_rgb,gt_left_rgb],dim=-2)
            rendered_right_images_vis = torch.cat([rendered_right_rgb,gt_right_rgb],dim=-2)
            rendered_view = torch.cat([rendered_left_images_vis,rendered_right_images_vis],dim=-1)

            skimage.io.imsave(os.path.join(current_saved_bin_folder,"rendered_views_{}.png".format(internal_view_index)),
                                (rendered_view.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            
            rendered_left_depth_vis = torch.cat([rendered_left_depth,gt_left_depth],dim=-2)
            rendered_right_depth_vis = torch.cat([rendered_right_depth,gt_right_depth],dim=-2)
            rendered_depth_vis = torch.cat([rendered_left_depth_vis,rendered_right_depth_vis],dim=-1)
            rendered_depth_vis = rendered_depth_vis.squeeze(0).cpu().numpy()
            rendered_depth_vis = convert_depth_to_disp(depth=rendered_depth_vis)
            skimage.io.imsave(os.path.join(current_saved_bin_folder,"last_depth_{}.png".format(internal_view_index)),
                                rendered_depth_vis)


        results_dict = dict()
        results_dict['left_psnr_first'] = left_psnr_list[0].data.item()
        results_dict['left_psnr_center'] = left_psnr_list[1].data.item()
        results_dict['left_psnr_last'] = left_psnr_list[2].data.item()
        results_dict['left_psnr_avg'] = get_mean(left_psnr_list).data.item()

        results_dict['right_psnr_first'] = right_psnr_list[0].data.item()
        results_dict['right_psnr_center'] = right_psnr_list[1].data.item()
        results_dict['right_psnr_last'] = right_psnr_list[2].data.item()
        results_dict['right_psnr_avg'] = get_mean(right_psnr_list).data.item()
        
        results_dict['rendered_depth_left_mae'] = get_mean(left_depth_mae_list).data.item()
        results_dict['rendered_depth_right_mae'] = get_mean(right_depth_mae_list).data.item()
        
        
        saved_into_json(data_dict=results_dict,path=os.path.join(current_saved_bin_folder,
                                                                 "results.json"
                                                                 ))


    def generate_low_quality_gt_pairs(self,batch,
                                      saved_dir,
                                      bin_token_list,
                                      cfg=None,
                                      view_nums=2,
                                      matching_nums=2,
                                      ):
        
        bin_token_name = bin_token_list[0][:-4]

        # loss and loss terms
        with torch.no_grad():
            loss, loss_terms,rendered_fusion_list,\
                rendered_volume_list,rendered_cv_results_list, \
                    predicted_input_depth,input_sparse_gt_depth,\
                        output_rgb,sparse_depth_gt,input_images = self.forward(batch,mode='val',
                                                            view_num=view_nums,
                                                            matching_nums=matching_nums,
                                                            cfg=cfg)


        rendered_images_fusion = rendered_fusion_list[0] #(1,6,3,H,W)
        gt_images = output_rgb
        
        
    
        # change the ordered.
        rendered_images_fusion = interleave_left_right(rendered_images_fusion)
        gt_images = interleave_left_right(gt_images)
        
        # saved rendered left and right images
        saved_rendered_image_root_path = os.path.join(saved_dir,"rendered_images")
        saved_rendered_left_image_folder = os.path.join(saved_rendered_image_root_path,"left_images")
        saved_rendered_right_image_folder = os.path.join(saved_rendered_image_root_path,"right_images")
        os.makedirs(saved_rendered_left_image_folder,exist_ok=True)
        os.makedirs(saved_rendered_right_image_folder,exist_ok=True)
        
        
        # saved gt left and right images
        saved_gt_image_root_path = os.path.join(saved_dir,"gt_images")
        saved_gt_left_image_folder = os.path.join(saved_gt_image_root_path,"left_images")
        saved_gt_right_image_folder = os.path.join(saved_gt_image_root_path,"right_images")
        os.makedirs(saved_gt_left_image_folder,exist_ok=True)
        os.makedirs(saved_gt_right_image_folder,exist_ok=True)
        
        
        # saving the rendering left and right images
        for i in range(rendered_images_fusion.shape[1]//2):
            rendered_left_image = rendered_images_fusion[:,i*2,:,:,:]
            rendered_right_image = rendered_images_fusion[:,i*2+1,:,:,:]
            
            current_saved_rendered_left_image_name = os.path.join(saved_rendered_left_image_folder,
                                                                  "{}_left_image_{}.png".format(bin_token_name,i))
            
            current_saved_rendered_right_image_name = os.path.join(saved_rendered_right_image_folder,
                                                                  "{}_right_image_{}.png".format(bin_token_name,i))
            
            skimage.io.imsave(current_saved_rendered_left_image_name,
                              (rendered_left_image.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            skimage.io.imsave(current_saved_rendered_right_image_name,
                              (rendered_right_image.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
    

            gt_left_image = gt_images[:,i*2,:,:,:]
            gt_right_image = gt_images[:,i*2+1,:,:,:]
            
            current_saved_gt_left_image_name = os.path.join(saved_gt_left_image_folder,
                                                                  "{}_left_image_{}.png".format(bin_token_name,i))
            
            current_saved_gt_right_image_name = os.path.join(saved_gt_right_image_folder,
                                                                  "{}_right_image_{}.png".format(bin_token_name,i))
            
            skimage.io.imsave(current_saved_gt_left_image_name,
                              (gt_left_image.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
            skimage.io.imsave(current_saved_gt_right_image_name,
                              (gt_right_image.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))


    def bev_video_kitti360(self,batch,cfg=None,
                           rescale_h=3.0,rescale_w=1.0):
        
        
        view_num = 2
        matching_nums = 2
        
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
        
        # doing rendering here
        render_c2w = output_batch_dict["output_c2ws"] #(1,6,4,4)
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"] # [B,6*3]
        render_fovys = output_batch_dict["output_fovys"] # [B,6*3]
        
        nums_of_views_all = render_c2w.shape[1]
        nums_of_center_left_index = (nums_of_views_all-2)//2 -2
        nums_of_center_right_index = nums_of_views_all -2 -2
        
        rendered_c2w_center_left = render_c2w[:,nums_of_center_left_index,:,:].unsqueeze(1)
        
    def validation_on_the_novel_bev_views(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        view_num=2,
                                        matching_nums=2,
                                        cfg=None,
                                        vis=False,
                                        ):
        
        bin_token_name = bin_token_list[0][:-4]
        

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
        
        nums_of_views_all = render_c2w.shape[1]
        nums_of_center_left_index = (nums_of_views_all-2)//2 -2
        nums_of_center_right_index = nums_of_views_all -2 -2
        
        rendered_c2w_center_left = render_c2w[:,nums_of_center_left_index,:,:].unsqueeze(1)
        
        rendered_c2w_center_left_movment1 = copy.deepcopy(rendered_c2w_center_left)
        rendered_c2w_center_left_movment1[0][0][2,3] = rendered_c2w_center_left_movment1[0][0][2,3] + 3
        
        rendered_c2w_center_left_movment2 = copy.deepcopy(rendered_c2w_center_left_movment1)
        rendered_c2w_center_left_movment2[0][0] = add_local_pitch(rendered_c2w_center_left_movment2[0][0], deg=-30.0)
        

        rendered_c2w_center_left_movment3 = copy.deepcopy(rendered_c2w_center_left_movment1)
        rendered_c2w_center_left_movment3[0][0][2,3] = rendered_c2w_center_left_movment3[0][0][2,3] + 2
        
        rendered_c2w_center_left_movment4 = copy.deepcopy(rendered_c2w_center_left_movment3)
        rendered_c2w_center_left_movment4[0][0] = add_local_pitch(rendered_c2w_center_left_movment4[0][0], deg=-30.0)
        
        
        rendered_c2w_center_left_movment5 = copy.deepcopy(rendered_c2w_center_left_movment3)
        rendered_c2w_center_left_movment5[0][0][2,3] = rendered_c2w_center_left_movment5[0][0][2,3] + 3
        
        rendered_c2w_center_left_movment6 = copy.deepcopy(rendered_c2w_center_left_movment5)
        rendered_c2w_center_left_movment6[0][0] = add_local_pitch(rendered_c2w_center_left_movment6[0][0], deg=-30.0)
        
        
        num_frames_short = 60
        num_frames_long = 60
        
        t_short = torch.linspace(0, 1, num_frames_short, dtype=torch.float32, device=self.device)
        t_long = torch.linspace(0, 1 - 1 / (num_frames_long + 1), num_frames_long, dtype=torch.float32, device=self.device)
        # center left rot
        movement_0 = interpolate_extrinsics(rendered_c2w_center_left,
                                            rendered_c2w_center_left_movment1,
                                            t_short)
        # center left rot back
        movement_1 = interpolate_extrinsics(rendered_c2w_center_left_movment1,rendered_c2w_center_left_movment2,t_short)
        # center left to right
        movement_2 = interpolate_extrinsics(rendered_c2w_center_left_movment2,rendered_c2w_center_left_movment3,t_short)
        # center right rot
        movement_3 = interpolate_extrinsics(rendered_c2w_center_left_movment3,rendered_c2w_center_left_movment4,t_short)
        # center right rot back
        movement_4 = interpolate_extrinsics(rendered_c2w_center_left_movment4,rendered_c2w_center_left_movment3,t_short)
        # center right to left
        movement_5 = interpolate_extrinsics(rendered_c2w_center_left_movment3,rendered_c2w_center_left_movment5,t_short)
        # center left to last left
        movement_6 = interpolate_extrinsics(rendered_c2w_center_left_movment5,rendered_c2w_center_left_movment6,t_short)


        c2w_interp = torch.cat([movement_0, movement_1, movement_2,
                                movement_3, movement_4,movement_5,
                                movement_6
                                ], dim=1)        

        num_frames_all = 60 * c2w_interp.shape[1]
        fovxs_interp =output_batch_dict["output_fovxs"][:, -2:-1].repeat(1, num_frames_all)
        fovys_interp =output_batch_dict["output_fovys"][:, -2:-1].repeat(1, num_frames_all)
 
        # return a dicts: rendered images and rendered alphs and rendered depth
        render_novel_bev_view_pkg = self.renderer.render(
            gaussians=gaussians_all,
            c2w=c2w_interp.view(1,-1,4,4),
            fovx=fovxs_interp,
            fovy=fovys_interp,
            rays_o=None,
            rays_d=None
        )  
        
        output_imgs = render_novel_bev_view_pkg["image"] # b v 3 h w
        
        dump_path = "rendered_novel_bev_view.mp4"
        video = (output_imgs[0].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        video_rec = wandb.Video(video[None], fps=30, format="mp4")
        video_tensor = video_rec._prepare_video(video_rec.data)
        clip = mpy.ImageSequenceClip(list(video_tensor), fps=30)
        clip.write_videofile(dump_path, codec='libx264', preset='medium', logger=None)
        
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
        
        rendered_fusion_list = [rendered_color_fuse,rendered_depth_fuse]
        
        output_rgb = output_batch_dict['output_imgs']
        rgb_gt = output_rgb

        rendered_images_fusion = rendered_fusion_list[0] #(1,6,3,H,W)
        rendered_depth_fusion = rendered_fusion_list[1] #(1,V,H，W)
        rendered_images_gt = output_rgb

    
        # change the ordered.
        rendered_images_fusion = interleave_left_right(rendered_images_fusion)

        rendered_images_gt = interleave_left_right(rendered_images_gt)
    
        # first view
        rendered_images_first_stereo = rendered_images_fusion[:,-2:,:,:,:]
        gt_images_first_stereo = rendered_images_gt[:,-2:,:,:,:]
        renderded_depth_first_stereo = rendered_depth_fusion[:,-2:,:,:]

        # last view
        rendered_images_last_stereo = rendered_images_fusion[:,-4:-2,:,:,:]
        gt_images_last_stereo = rendered_images_gt[:,-4:-2,:,:,:]
        renderded_depth_last_stereo = rendered_depth_fusion[:,-4:-2,:,:]

        # center view
        rendered_images_center_stereo = rendered_images_fusion[:,-6:-4,:,:,:]
        gt_images_center_stereo = rendered_images_gt[:,-6:-4,:,:,:]
        renderded_depth_center_stereo = rendered_depth_fusion[:,-6:-4,:,:]

        # all view
        rendered_images_all_stereo = rendered_images_fusion
        gt_images_all_stereo =  rendered_images_gt
        renderded_depth_all_stereo = rendered_depth_fusion

        
        if vis:
            saved_folder_for_visualization = os.path.join(val_result_savedir,bin_token_name)
            os.makedirs(saved_folder_for_visualization,exist_ok=True)
            rendered_images_folder_path = os.path.join(saved_folder_for_visualization,'rendered_images')
            GT_images_folder_path = os.path.join(saved_folder_for_visualization,'GT Images')
            Rendered_Depth_Error_Folder_Path = os.path.join(saved_folder_for_visualization,"Rendered_Depth_Error")
            
            os.makedirs(rendered_images_folder_path,exist_ok=True)
            os.makedirs(GT_images_folder_path,exist_ok=True)
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

    # iteration twice
    def validation_on_the_forward_views_progressive(self,
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
            
            start_time = time.time()
            
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
        
        render_c2w = interleave_render_c2w[:,-6:-4,:,:]
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-6:-4]# [B,6*3]
        render_fovys = output_batch_dict["output_fovys"][:,-6:-4] # [B,6*3]
        gt_center_frame = interleave_left_right(output_batch_dict["output_imgs"])
        gt_center_frame =gt_center_frame[:,-6:-4,:,:,:]

        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_center_frame = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_center_frame = torch.clamp(rendered_center_frame,min=0,max=1.0)
        
        end_time1 = time.time()

        
        
        if use_diffix3d:
            # logic here
            # enhance the center frame
            rendered_center_frame = rendered_center_frame
            rendered_center_left = rendered_center_frame[0,0,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_center_right = rendered_center_frame[0,1,:,:,:].permute(1,2,0).cpu().numpy()
            
            rendered_center_left = (rendered_center_left*255).astype(np.uint8)
            rendered_center_right = (rendered_center_right*255).astype(np.uint8)
            rendered_center_left_pil = Image.fromarray(rendered_center_left)
            rendered_center_right_pil = Image.fromarray(rendered_center_right)
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
            
            enhanced_rendered_center_left = np.array(enhanced_rendered_center_left_pil).astype(np.float32)/255.0
            enhanced_rendered_center_right = np.array(enhanced_rendered_center_right_pil).astype(np.float32)/255.0
            
            enhanced_rendered_center_left = torch.from_numpy(enhanced_rendered_center_left).to(rendered_center_frame.device)
            enhanced_rendered_center_right = torch.from_numpy(enhanced_rendered_center_right).to(rendered_center_frame.device)
            enhanced_rendered_center_left = enhanced_rendered_center_left.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_center_right = enhanced_rendered_center_right.permute(2,0,1).unsqueeze(0)
            
            enhanced_rendered_center_frame = torch.cat([enhanced_rendered_center_left,
                                                        enhanced_rendered_center_right],dim=0).unsqueeze(0)
            
            rendered_center_frame = enhanced_rendered_center_frame
            
            # FIXME
            fusion_rendered_center_frame = rendered_center_frame * 0.66 + gt_center_frame * 0.34
            fusion_rendered_center_frame = torch.clamp(fusion_rendered_center_frame,min=0,max=1.0)
            rendered_center_frame = fusion_rendered_center_frame

            
        '''second time inference'''
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=4,
                                                                         matching_nums=3)
        
        input_batch_dict["imgs"][:,2:,:,:,:] = rendered_center_frame
        

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
        interleave_render_c2w = interleave_left_right_pose(render_c2w)
        render_c2w = interleave_render_c2w[:,-4:-2,:,:]
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-4:-2]
        render_fovys = output_batch_dict["output_fovys"][:,-4:-2]
        gt_last_frame = interleave_left_right(output_batch_dict["output_imgs"])
        gt_last_frame =gt_last_frame[:,-4:-2,:,:,:]
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_last_frame = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_last_frame = torch.clamp(rendered_last_frame,min=0,max=1.0)

        if use_diffix3d:
            # logic here
            # enhance the last frame
            rendered_last_frame = rendered_last_frame
            rendered_last_left = rendered_last_frame[0,0,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_last_right = rendered_last_frame[0,1,:,:,:].permute(1,2,0).cpu().numpy()
            
            rendered_last_left = (rendered_last_left*255).astype(np.uint8)
            rendered_last_right = (rendered_last_right*255).astype(np.uint8)
            rendered_last_left_pil = Image.fromarray(rendered_last_left)
            rendered_last_right_pil = Image.fromarray(rendered_last_right)
            width,height = rendered_last_left_pil.size
            
            # get the ref image
            ref_image_left = input_batch_dict["imgs"][0,0,:,:,:].permute(1,2,0).cpu().numpy()
            ref_image_left = (ref_image_left*255).astype(np.uint8)
            ref_image_left_pil = Image.fromarray(ref_image_left)
            ref_image_right = input_batch_dict["imgs"][0,1,:,:,:].permute(1,2,0).cpu().numpy()
            ref_image_right = (ref_image_right*255).astype(np.uint8)
            ref_image_right_pil = Image.fromarray(ref_image_right)

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
            
            enhanced_rendered_last_left = np.array(enhanced_rendered_last_left_pil).astype(np.float32)/255.0
            enhanced_rendered_last_right = np.array(enhanced_rendered_last_right_pil).astype(np.float32)/255.0
            
            enhanced_rendered_last_left = torch.from_numpy(enhanced_rendered_last_left).to(rendered_last_frame.device)
            enhanced_rendered_last_right = torch.from_numpy(enhanced_rendered_last_right).to(rendered_last_frame.device)
            enhanced_rendered_last_left = enhanced_rendered_last_left.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_last_right = enhanced_rendered_last_right.permute(2,0,1).unsqueeze(0)
            
            enhanced_rendered_last_frame = torch.cat([enhanced_rendered_last_left,
                                                        enhanced_rendered_last_right],dim=0).unsqueeze(0)
            
            rendered_last_frame = enhanced_rendered_last_frame
            
            fusion_rendered_last_frame = rendered_last_frame * 0.68 + gt_last_frame * 0.32
            fusion_rendered_last_frame = torch.clamp(fusion_rendered_last_frame,min=0,max=1.0)
            rendered_last_frame = fusion_rendered_last_frame
            
            
            
        
        '''third time inference'''
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=6,
                                                                         matching_nums=4)
        
        
        input_batch_dict["imgs"][:,2:4,:,:,:] = rendered_center_frame
        input_batch_dict["imgs"][:,4:,:,:,:] = rendered_last_frame 
        
        
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
        
        
        end_time2 = time.time()

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
            
            
            # # rendered videos
            # saved_videos_path = os.path.join(saved_folder_for_visualization,'videos')
            # os.makedirs(saved_videos_path,exist_ok=True)
            
            # preds, saved_video_name = self.forward_kitti360_videos(batch=batch,cfg=cfg,view_num=view_num,matching_nums=matching_nums)
            
            # bs = preds["img"].shape[0]  
            # pred_imgs = preds["img"] #(4,960,3,224,400)
            # pred_depths = preds["depth"] #(4,960,3,224,400)
            
            
            # # saved the results with batch
            # for b in range(bs):
            #     bin_token = saved_video_name[b]
            #     # dump rgb view
            #     dump_path = osp.join(saved_videos_path, "{}_rgb.mp4".format(bin_token))
            #     video = (pred_imgs[b].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
            #     video_rec = wandb.Video(video[None], fps=30, format="mp4")
            #     video_tensor = video_rec._prepare_video(video_rec.data)
            #     clip = mpy.ImageSequenceClip(list(video_tensor), fps=30)
            #     clip.write_videofile(dump_path, codec='libx264', preset='medium', logger=None)
                
            #     # dump depth view
            #     dump_path_dpt = osp.join(saved_videos_path, "{}_depth.mp4".format(bin_token))
            #     pred_depth = pred_depths[b].clamp(0.0, 100.0)
            #     max_val = float(pred_depth.max())
            #     video_dpt = depths_to_colors(pred_depths[b], concat="frame", max_val=max_val)
            #     video_dpt = video_dpt.transpose((0, 3, 1, 2))
            #     video_rec_dpt = wandb.Video(video_dpt[None], fps=30, format="mp4")
            #     video_tensor_dpt = video_rec_dpt._prepare_video(video_rec_dpt.data)
            #     clip_dpt = mpy.ImageSequenceClip(list(video_tensor_dpt), fps=30)
            #     clip_dpt.write_videofile(dump_path_dpt, codec='libx264', preset='medium', logger=None)
        
        
        
        return evaluation_results_stat    
        
    def generating_difix_training_dataset(self,
                                          batch,
                                          val_result_savedir,
                                          bin_token_list,
                                          start_images_views = 2,
                                          cfg=None,
                                          vis=False):
    
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
            input_img = img.clone()
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
        
        render_c2w = interleave_render_c2w[:,-6:-2,:,:]
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-6:-2]# [B,6*3]
        render_fovys = output_batch_dict["output_fovys"][:,-6:-2] # [B,6*3]
        gt_center_frame = interleave_left_right(output_batch_dict["output_imgs"])
        gt_center_frame =gt_center_frame[:,-6:-2,:,:,:]
        


        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_center_frame = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_center_frame = torch.clamp(rendered_center_frame,min=0,max=1.0)
        
        
        nums_of_views = 4
        
        saved_folder_for_rendered_subfolder = os.path.join(val_result_savedir,"rendered_views",bin_token_name)
        os.makedirs(saved_folder_for_rendered_subfolder,exist_ok=True)
        
        saved_folder_for_gt_subfolder_views = os.path.join(val_result_savedir,"gt_views",bin_token_name)
        os.makedirs(saved_folder_for_gt_subfolder_views,exist_ok=True)
        
        saved_folder_for_reference_views = os.path.join(val_result_savedir,"reference_views",bin_token_name)
        os.makedirs(saved_folder_for_reference_views,exist_ok=True)
        
        for views_idx in range(nums_of_views):
            
            if views_idx == 0:
                saved_render_filename = 'rendered_center_left.png'
            elif views_idx == 1:
                saved_render_filename = 'rendered_center_right.png'
            elif views_idx == 2:
                saved_render_filename = 'rendered_last_left.png'
            elif views_idx == 3:
                saved_render_filename = 'rendered_last_right.png'
            else:
                raise NotImplementedError
            
            if views_idx == 0:
                saved_gt_filename = 'gt_center_left.png'
            elif views_idx == 1:
                saved_gt_filename = 'gt_center_right.png'
            elif views_idx == 2:
                saved_gt_filename = 'gt_last_left.png'
            elif views_idx == 3:
                saved_gt_filename = 'gt_last_right.png'
            else:
                raise NotImplementedError
            
            if views_idx == 0:
                saved_reference_filename = 'reference_first_center_left.png'
                reference_id = 0
            elif views_idx == 1:
                saved_reference_filename = 'reference_first_center_right.png'
                reference_id = 1
            elif views_idx == 2:
                saved_reference_filename = 'reference_first_last_left.png'
                reference_id = 0
            elif views_idx == 3:
                saved_reference_filename = 'reference_first_last_right.png'
                reference_id = 1
            else:
                pass
            
            
            saved_render_filename = os.path.join(saved_folder_for_rendered_subfolder,saved_render_filename)
            saved_gt_filename = os.path.join(saved_folder_for_gt_subfolder_views,saved_gt_filename)
            
            saved_reference_filename = os.path.join(saved_folder_for_reference_views,saved_reference_filename)


            current_rendered_views = rendered_center_frame[0,views_idx,:,:,:].permute(1,2,0).cpu().numpy()
            current_rendered_views = (current_rendered_views*255).astype(np.uint8)
            current_rendered_views_pil = Image.fromarray(current_rendered_views)
            if not os.path.exists(saved_render_filename):
                current_rendered_views_pil.save(saved_render_filename)
            
            current_gt_views = gt_center_frame[0,views_idx,:,:,:].permute(1,2,0).cpu().numpy()
            current_gt_views = (current_gt_views*255).astype(np.uint8)
            current_gt_views_pil = Image.fromarray(current_gt_views)
            if not os.path.exists(saved_gt_filename):
                current_gt_views_pil.save(saved_gt_filename)

            current_reference_views = input_img[0,reference_id,:,:,:].permute(1,2,0).cpu().numpy()
            current_reference_views = (current_reference_views*255).astype(np.uint8)
            current_reference_views_pil = Image.fromarray(current_reference_views)
            if not os.path.exists(saved_reference_filename):
                current_reference_views_pil.save(saved_reference_filename)
        
    # iteration twice
    def validation_on_the_forward_views_progressive_iter_once_revised(self,
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
        
        render_c2w = interleave_render_c2w[:,-6:-2,:,:]
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-6:-2]# [B,6*3]
        render_fovys = output_batch_dict["output_fovys"][:,-6:-2] # [B,6*3]
        gt_center_frame = interleave_left_right(output_batch_dict["output_imgs"])
        gt_center_frame =gt_center_frame[:,-6:-2,:,:,:]
        


        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_center_frame = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_center_frame = torch.clamp(rendered_center_frame,min=0,max=1.0)
        

        if use_diffix3d:
            # logic here
            # enhance the center frame
            rendered_center_frame = rendered_center_frame
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
            


            
        ''' Second Time Inference '''
        input_batch_dict,output_batch_dict = self.prepare_input_multiview(batch=batch,view_num=6,
                                                                         matching_nums=4)
        input_batch_dict["imgs"][:,2:,:,:,:] = rendered_center_frame
        
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
        

        # Post Processing
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


    # FIXME: This is the bug version of the first submission to IROS2026, please delete in the future.
    def validation_on_the_forward_views_progressive_iter_once_bug_version(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        start_images_views=2,
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

        # First Time Iteration
        with torch.no_grad():
            input_batch_dict, output_batch_dict = self.prepare_input_multiview(batch=batch, view_num=view_num,
                                                                               matching_nums=matching_nums)

            img = input_batch_dict["imgs"]
            height, width = img.shape[-2:]
            bs = img.shape[0]

            img_feats = self.extract_img_feat(img=img)
            gaussians_cv, gaussians_feat, pred_depths = self.costvolume_gs(input_batch_dict, cfg=cfg,
                                                                          images_feat=img_feats[0])

            pc_range = self.dataset_params.pc_range
            x_start, y_start, z_start, x_end, y_end, z_end = pc_range
            gaussians_cv_mask, gaussians_feat_mask = [], []
            for b in range(bs):
                mask_pixel_i = (gaussians_cv[b, :, 0] >= x_start) & (gaussians_cv[b, :, 0] <= x_end) & \
                               (gaussians_cv[b, :, 1] >= y_start) & (gaussians_cv[b, :, 1] <= y_end) & \
                               (gaussians_cv[b, :, 2] >= z_start) & (gaussians_cv[b, :, 2] <= z_end)
                gaussians_cv_mask_i = gaussians_cv[b][mask_pixel_i]
                gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
                gaussians_cv_mask.append(gaussians_cv_mask_i)
                gaussians_feat_mask.append(gaussians_feat_mask_i)

            gaussians_volume = self.volume_gs(
                [img_feats[0]],
                input_batch_dict['extrinsics'],
                gaussians_cv_mask,
                gaussians_feat_mask,
                input_batch_dict["img_metas"])

            gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
            gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)

            gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
            bs = gaussians_all.shape[0]

            render_c2w = output_batch_dict["output_c2ws"]
            interleave_render_c2w = interleave_left_right_pose(render_c2w)
            render_c2w = interleave_render_c2w[:, -6:-2, :, :]
            intrinsics = input_batch_dict['intrinsics'].clone()
            output_intrinsics = intrinsics[:, 0:1, :, :].repeat(1, render_c2w.shape[1], 1, 1)
            render_fovxs = output_batch_dict["output_fovxs"][:, -6:-2]
            render_fovys = output_batch_dict["output_fovys"][:, -6:-2]
            gt_center_frame = interleave_left_right(output_batch_dict["output_imgs"])
            gt_center_frame = gt_center_frame[:, -6:-2, :, :, :]

            render_pkg_fuse = self.renderer.render(
                gaussians=gaussians_all,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )

            rendered_results_fuse = render_pkg_fuse
            rendered_center_frame = rendered_results_fuse['image']
            rendered_center_frame = torch.clamp(rendered_center_frame, min=0, max=1.0)

            if use_diffix3d:
              
                original_rendered_center_frame = rendered_center_frame.clone()

                rendered_center_left = rendered_center_frame[0, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
                rendered_center_right = rendered_center_frame[0, 1, :, :, :].permute(1, 2, 0).cpu().numpy()
                rendered_last_left = rendered_center_frame[0, 2, :, :, :].permute(1, 2, 0).cpu().numpy()
                rendered_last_right = rendered_center_frame[0, 3, :, :, :].permute(1, 2, 0).cpu().numpy()

                rendered_center_left = (rendered_center_left * 255).astype(np.uint8)
                rendered_center_right = (rendered_center_right * 255).astype(np.uint8)
                rendered_last_left = (rendered_last_left * 255).astype(np.uint8)
                rendered_last_right = (rendered_last_right * 255).astype(np.uint8)

                rendered_center_left_pil = Image.fromarray(rendered_center_left)
                rendered_center_right_pil = Image.fromarray(rendered_center_right)
                rendered_last_left_pil = Image.fromarray(rendered_last_left)
                rendered_last_right_pil = Image.fromarray(rendered_last_right)

                width, height = rendered_center_left_pil.size

                ref_image_left = input_batch_dict["imgs"][0, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
                ref_image_left = (ref_image_left * 255).astype(np.uint8)
                ref_image_left_pil = Image.fromarray(ref_image_left)
                ref_image_right = input_batch_dict["imgs"][0, 1, :, :, :].permute(1, 2, 0).cpu().numpy()
                ref_image_right = (ref_image_right * 255).astype(np.uint8)
                ref_image_right_pil = Image.fromarray(ref_image_right)

                enhanced_rendered_center_left_pil = diffix3d_network.sample(
                    rendered_center_left_pil, height=112, width=544,
                    ref_image=ref_image_left_pil, prompt=cfg.prompt)
                enhanced_rendered_center_right_pil = diffix3d_network.sample(
                    rendered_center_right_pil, height=112, width=544,
                    ref_image=ref_image_right_pil, prompt=cfg.prompt)
                enhanced_rendered_last_left_pil = diffix3d_network.sample(
                    rendered_last_left_pil, height=112, width=544,
                    ref_image=ref_image_left_pil, prompt=cfg.prompt)
                enhanced_rendered_last_right_pil = diffix3d_network.sample(
                    rendered_last_right_pil, height=112, width=544,
                    ref_image=ref_image_right_pil, prompt=cfg.prompt)

                enhanced_rendered_center_left = np.array(enhanced_rendered_center_left_pil).astype(np.float32) / 255.0
                enhanced_rendered_center_right = np.array(enhanced_rendered_center_right_pil).astype(np.float32) / 255.0
                enhanced_rendered_last_left = np.array(enhanced_rendered_last_left_pil).astype(np.float32) / 255.0
                enhanced_rendered_last_right = np.array(enhanced_rendered_last_right_pil).astype(np.float32) / 255.0

                enhanced_rendered_center_left = torch.from_numpy(enhanced_rendered_center_left).to(rendered_center_frame.device)
                enhanced_rendered_center_right = torch.from_numpy(enhanced_rendered_center_right).to(rendered_center_frame.device)
                enhanced_rendered_last_left = torch.from_numpy(enhanced_rendered_last_left).to(rendered_center_frame.device)
                enhanced_rendered_last_right = torch.from_numpy(enhanced_rendered_last_right).to(rendered_center_frame.device)
                enhanced_rendered_center_left = enhanced_rendered_center_left.permute(2, 0, 1).unsqueeze(0)
                enhanced_rendered_center_right = enhanced_rendered_center_right.permute(2, 0, 1).unsqueeze(0)
                enhanced_rendered_last_left = enhanced_rendered_last_left.permute(2, 0, 1).unsqueeze(0)
                enhanced_rendered_last_right = enhanced_rendered_last_right.permute(2, 0, 1).unsqueeze(0)

                enhanced_rendered_center_frame = torch.cat([
                    enhanced_rendered_center_left, enhanced_rendered_center_right,
                    enhanced_rendered_last_left, enhanced_rendered_last_right
                ], dim=0).unsqueeze(0)
                rendered_center_frame = enhanced_rendered_center_frame

 
        input_batch_dict, output_batch_dict = self.prepare_input_multiview(batch=batch, view_num=6,
                                                                           matching_nums=4)

        if use_diffix3d:
            input_batch_dict["imgs"][:, 2:, :, :, :] = original_rendered_center_frame
            input_batch_dict["imgs"][:, 4:6, :, :, :] = rendered_center_frame[:, 2:4, :, :, :]
        else:
            input_batch_dict["imgs"][:, 2:, :, :, :] = rendered_center_frame

        img = input_batch_dict["imgs"]
        height, width = img.shape[-2:]
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img)
        gaussians_cv, gaussians_feat, pred_depths = self.costvolume_gs(input_batch_dict, cfg=cfg,
                                                                      images_feat=img_feats[0])

        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_cv_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_cv[b, :, 0] >= x_start) & (gaussians_cv[b, :, 0] <= x_end) & \
                           (gaussians_cv[b, :, 1] >= y_start) & (gaussians_cv[b, :, 1] <= y_end) & \
                           (gaussians_cv[b, :, 2] >= z_start) & (gaussians_cv[b, :, 2] <= z_end)
            gaussians_cv_mask_i = gaussians_cv[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_cv_mask.append(gaussians_cv_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)

        gaussians_volume = self.volume_gs(
            [img_feats[0]],
            input_batch_dict['extrinsics'],
            gaussians_cv_mask,
            gaussians_feat_mask,
            input_batch_dict["img_metas"])

        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
        bs = gaussians_all.shape[0]

        render_c2w = output_batch_dict["output_c2ws"]
        intrinsics = input_batch_dict['intrinsics'].clone()
        output_intrinsics = intrinsics[:, 0:1, :, :].repeat(1, render_c2w.shape[1], 1, 1)
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
        rendered_color_fuse = rendered_results_fuse['image']
        rendered_depth_fuse = rendered_results_fuse['depth']
        rendered_alpha_fuse = rendered_results_fuse['alpha']
        rendered_depth_fuse = rendered_depth_fuse.squeeze(2)
        rendered_alpha_fuse = rendered_alpha_fuse.squeeze(2)
        rendered_color_fuse = torch.clamp(rendered_color_fuse, min=0, max=1.0)
        rendered_depth_fuse = torch.clamp(rendered_depth_fuse, min=0, max=150)

        output_rgb = output_batch_dict['output_imgs']
        sparse_depth_gt = output_batch_dict['output_sparse_depth']

        rendered_images_fusion = interleave_left_right(rendered_color_fuse)
        rendered_depth_fusion = interleave_left_right_depth(rendered_depth_fuse)
        sparse_depth_gt = interleave_left_right_depth(sparse_depth_gt)
        rendered_images_gt = interleave_left_right(output_rgb)

        if cfg.use_diffix3d_postprocessing:
            enhanced_rendered_images_list = []
            for current_view_idx in range(rendered_images_fusion.shape[1]):
                current_rendered_image = rendered_images_fusion[0, current_view_idx, :, :, :].permute(1, 2, 0).cpu().numpy()
                current_rendered_image = (current_rendered_image * 255).astype(np.uint8)
                current_rendered_image_pil = Image.fromarray(current_rendered_image)
                current_rendered_image_pil = diffix3d_network.sample(
                    current_rendered_image_pil, height=112, width=544,
                    ref_image=ref_image_left_pil, prompt=cfg.prompt)
                current_rendered_image_np = np.array(current_rendered_image_pil).astype(np.float32) / 255.0
                current_rendered_image = torch.from_numpy(current_rendered_image_np).to(rendered_center_frame.device)
                current_rendered_image = current_rendered_image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                enhanced_rendered_images_list.append(current_rendered_image)
            rendered_images_fusion = torch.cat(enhanced_rendered_images_list, dim=1)

        rendered_images_first_stereo = rendered_images_fusion[:, -2:, :, :, :]
        gt_images_first_stereo = rendered_images_gt[:, -2:, :, :, :]
        renderded_depth_first_stereo = rendered_depth_fusion[:, -2:, :, :]
        gt_depth_first_stereo = sparse_depth_gt[:, -2:, :, :]

        rendered_images_last_stereo = rendered_images_fusion[:, -4:-2, :, :, :]
        gt_images_last_stereo = rendered_images_gt[:, -4:-2, :, :, :]
        renderded_depth_last_stereo = rendered_depth_fusion[:, -4:-2, :, :]
        gt_depth_last_stereo = sparse_depth_gt[:, -4:-2, :, :]

        rendered_images_center_stereo = rendered_images_fusion[:, -6:-4, :, :, :]
        gt_images_center_stereo = rendered_images_gt[:, -6:-4, :, :, :]
        renderded_depth_center_stereo = rendered_depth_fusion[:, -6:-4, :, :]
        gt_depth_center_stereo = sparse_depth_gt[:, -6:-4, :, :]

        rendered_images_all_stereo = rendered_images_fusion
        gt_images_all_stereo = rendered_images_gt
        renderded_depth_all_stereo = rendered_depth_fusion
        gt_depth_all_stereo = sparse_depth_gt

        first_rgb_eval_info = metrics_mean(pred=rendered_images_first_stereo, gt=gt_images_first_stereo)
        first_rgb_lpips = first_rgb_eval_info['lpips']
        first_rgb_ssim = first_rgb_eval_info['ssim']
        first_rgb_psnr = first_rgb_eval_info['psnr']

        center_rgb_eval_info = metrics_mean(pred=rendered_images_center_stereo, gt=gt_images_center_stereo)
        center_rgb_lpips = center_rgb_eval_info['lpips']
        center_rgb_ssim = center_rgb_eval_info['ssim']
        center_rgb_psnr = center_rgb_eval_info['psnr']

        last_rgb_eval_info = metrics_mean(pred=rendered_images_last_stereo, gt=gt_images_last_stereo)
        last_rgb_lpips = last_rgb_eval_info['lpips']
        last_rgb_ssim = last_rgb_eval_info['ssim']
        last_rgb_psnr = last_rgb_eval_info['psnr']

        all_rgb_eval_info = metrics_mean(pred=rendered_images_all_stereo, gt=gt_images_all_stereo)
        all_rgb_lpips = all_rgb_eval_info['lpips']
        all_rgb_ssim = all_rgb_eval_info['ssim']
        all_rgb_psnr = all_rgb_eval_info['psnr']

        first_view_depth_eval_info = depth_metrics_absrel_sqrel_rmse_log(
            pred=renderded_depth_first_stereo, gt=gt_depth_first_stereo)
        frist_view_abs_rel = first_view_depth_eval_info['AbsRel']
        frist_view_sq_rel = first_view_depth_eval_info['SqRel']
        frist_view_rmse_log = first_view_depth_eval_info['RMSE_log']

        center_view_depth_eval_info = depth_metrics_absrel_sqrel_rmse_log(
            pred=renderded_depth_center_stereo, gt=gt_depth_center_stereo)
        center_view_abs_rel = center_view_depth_eval_info['AbsRel']
        center_view_sq_rel = center_view_depth_eval_info['SqRel']
        center_view_rmse_log = center_view_depth_eval_info['RMSE_log']

        last_view_depth_eval_info = depth_metrics_absrel_sqrel_rmse_log(
            pred=renderded_depth_last_stereo, gt=gt_depth_last_stereo)
        last_view_abs_rel = last_view_depth_eval_info['AbsRel']
        last_view_sq_rel = last_view_depth_eval_info['SqRel']
        last_view_rmse_log = last_view_depth_eval_info['RMSE_log']

        all_view_depth_eval_info = depth_metrics_absrel_sqrel_rmse_log(
            pred=renderded_depth_all_stereo, gt=gt_depth_all_stereo)
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
            "RGB": evaluation_rgb_results_stat,
            "Depth": evaluation_depth_results_stat,
        }

        if vis:
            saved_folder_for_visualization = os.path.join(val_result_savedir, bin_token_name)
            os.makedirs(saved_folder_for_visualization, exist_ok=True)
            rendered_images_folder_path = os.path.join(saved_folder_for_visualization, 'rendered_images')
            rendered_depth_folder_path = os.path.join(saved_folder_for_visualization, 'rendered_depth')
            GT_images_folder_path = os.path.join(saved_folder_for_visualization, 'GT Images')
            GT_depth_folder_path = os.path.join(saved_folder_for_visualization, 'GT Depth')
            Rendered_Depth_Error_Folder_Path = os.path.join(saved_folder_for_visualization, "Rendered_Depth_Error")
            os.makedirs(rendered_images_folder_path, exist_ok=True)
            os.makedirs(rendered_depth_folder_path, exist_ok=True)
            os.makedirs(GT_images_folder_path, exist_ok=True)
            os.makedirs(GT_depth_folder_path, exist_ok=True)
            os.makedirs(Rendered_Depth_Error_Folder_Path, exist_ok=True)

            rendered_first_stereo = torch.cat((rendered_images_first_stereo[:, 0, :, :, :], rendered_images_first_stereo[:, 1, :, :, :]), dim=-1)
            skimage.io.imsave(os.path.join(rendered_images_folder_path, 'first_stereo.png'), (rendered_first_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            rendered_last_stereo = torch.cat((rendered_images_last_stereo[:, 0, :, :, :], rendered_images_last_stereo[:, 1, :, :, :]), dim=-1)
            skimage.io.imsave(os.path.join(rendered_images_folder_path, 'last_stereo.png'), (rendered_last_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            rendered_center_stereo = torch.cat((rendered_images_center_stereo[:, 0, :, :, :], rendered_images_center_stereo[:, 1, :, :, :]), dim=-1)
            skimage.io.imsave(os.path.join(rendered_images_folder_path, 'center_stereo.png'), (rendered_center_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

            gt_first_stereo = torch.cat((gt_images_first_stereo[:, 0, :, :, :], gt_images_first_stereo[:, 1, :, :, :]), dim=-1)
            skimage.io.imsave(os.path.join(GT_images_folder_path, 'first_stereo.png'), (gt_first_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            gt_last_stereo = torch.cat((gt_images_last_stereo[:, 0, :, :, :], gt_images_last_stereo[:, 1, :, :, :]), dim=-1)
            skimage.io.imsave(os.path.join(GT_images_folder_path, 'last_stereo.png'), (gt_last_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            gt_center_stereo = torch.cat((gt_images_center_stereo[:, 0, :, :, :], gt_images_center_stereo[:, 1, :, :, :]), dim=-1)
            skimage.io.imsave(os.path.join(GT_images_folder_path, 'center_stereo.png'), (gt_center_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

            rendered_depth_first_stereo = torch.cat((renderded_depth_first_stereo[:, 0, :, :], renderded_depth_first_stereo[:, 1, :, :]), dim=-1)
            rendered_depth_first_stereo_vis = convert_depth_to_disp(depth=rendered_depth_first_stereo.squeeze(0).cpu().numpy())
            skimage.io.imsave(os.path.join(rendered_depth_folder_path, 'first_stereo_depth.png'), rendered_depth_first_stereo_vis)
            rendered_depth_last_stereo = torch.cat((renderded_depth_last_stereo[:, 0, :, :], renderded_depth_last_stereo[:, 1, :, :]), dim=-1)
            rendered_depth_last_stereo_vis = convert_depth_to_disp(depth=rendered_depth_last_stereo.squeeze(0).cpu().numpy())
            skimage.io.imsave(os.path.join(rendered_depth_folder_path, 'last_stereo_depth.png'), rendered_depth_last_stereo_vis)
            rendered_depth_center_stereo = torch.cat((renderded_depth_center_stereo[:, 0, :, :], renderded_depth_center_stereo[:, 1, :, :]), dim=-1)
            rendered_depth_center_stereo_vis = convert_depth_to_disp(depth=rendered_depth_center_stereo.squeeze(0).cpu().numpy())
            skimage.io.imsave(os.path.join(rendered_depth_folder_path, 'center_stereo_depth.png'), rendered_depth_center_stereo_vis)

            gt_depth_first_stereo_cat = torch.cat((gt_depth_first_stereo[:, 0, :, :], gt_depth_first_stereo[:, 1, :, :]), dim=-1)
            gt_depth_first_stereo_vis = convert_depth_to_disp(depth=gt_depth_first_stereo_cat.squeeze(0).cpu().numpy())
            skimage.io.imsave(os.path.join(GT_depth_folder_path, 'first_stereo_depth.png'), gt_depth_first_stereo_vis)
            gt_depth_last_stereo_cat = torch.cat((gt_depth_last_stereo[:, 0, :, :], gt_depth_last_stereo[:, 1, :, :]), dim=-1)
            gt_depth_last_stereo_vis = convert_depth_to_disp(depth=gt_depth_last_stereo_cat.squeeze(0).cpu().numpy())
            skimage.io.imsave(os.path.join(GT_depth_folder_path, 'last_stereo_depth.png'), gt_depth_last_stereo_vis)
            gt_depth_center_stereo_cat = torch.cat((gt_depth_center_stereo[:, 0, :, :], gt_depth_center_stereo[:, 1, :, :]), dim=-1)
            gt_depth_center_stereo_vis = convert_depth_to_disp(depth=gt_depth_center_stereo_cat.squeeze(0).cpu().numpy())
            skimage.io.imsave(os.path.join(GT_depth_folder_path, 'center_stereo_depth.png'), gt_depth_center_stereo_vis)

            disp_error_img_first_stereo = disp_error_img(D_est_tensor=rendered_depth_first_stereo, D_gt_tensor=gt_depth_first_stereo_cat)
            skimage.io.imsave(os.path.join(Rendered_Depth_Error_Folder_Path, 'first_stereo_depth_error.png'), (disp_error_img_first_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            disp_error_img_last_stereo = disp_error_img(D_est_tensor=rendered_depth_last_stereo, D_gt_tensor=gt_depth_last_stereo_cat)
            skimage.io.imsave(os.path.join(Rendered_Depth_Error_Folder_Path, 'last_stereo_depth_error.png'), (disp_error_img_last_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            disp_error_img_center_stereo = disp_error_img(D_est_tensor=rendered_depth_center_stereo, D_gt_tensor=gt_depth_center_stereo_cat)
            skimage.io.imsave(os.path.join(Rendered_Depth_Error_Folder_Path, 'center_stereo_depth_error.png'), (disp_error_img_center_stereo.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        return evaluation_results_stat

    # render videos
    def bev_video_kitti360(self,batch,cfg=None,
                           rescale_h=3.0,rescale_w=1.0):
        
        view_num = 2
        matching_nums = 2
        
        # get the input and the reference input
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=view_num,
                                                                         matching_nums=matching_nums)
        
        img =input_batch_dict["imgs"] #[B,6,3,H,W]
        height,width = img.shape[-2:]
        bs = img.shape[0]
        
        bin_token_name = input_batch_dict['bin_token_name'][0][:-4]  
        current_resolution = [img.shape[-2], img.shape[-1]]
        
        
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
    
    def get_additional_bev_novel_views_non_progressive(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        view_num=2,
                                        matching_nums=2,
                                        cfg=None,
                                        vis=False,
                                        rescale_h=2.0,
                                        rescale_w=1.0):
        
        # get the input and the reference input
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=view_num,
                                                                         matching_nums=matching_nums)
        
        img =input_batch_dict["imgs"] #[B,6,3,H,W]
        height,width = img.shape[-2:]
        bs = img.shape[0]
        
        bin_token_name = input_batch_dict['bin_token_name'][0][:-4]  
        current_resolution = [img.shape[-2], img.shape[-1]]
        
        
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


            # rendered interploated images and depth views.
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

    # iteration twice
    def get_additional_bev_novel_views_progressive_iter_twice(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        start_images_views = 2,
                                        use_diffix3d=False,
                                        diffix3d_network=None,
                                        use_ref=False,
                                        cfg=None,
                                        vis=False,
                                        rescale_h=2.0,
                                        rescale_w=1.0):
        
        
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
            
            start_time = time.time()
            
            img =input_batch_dict["imgs"] #[B,6,3,H,W]
            height,width = img.shape[-2:]
            bs = img.shape[0]   
            
            current_resolution = [img.shape[-2], img.shape[-1]]
            
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
        
        render_c2w = interleave_render_c2w[:,-6:-4,:,:]
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-6:-4]# [B,6*3]
        render_fovys = output_batch_dict["output_fovys"][:,-6:-4] # [B,6*3]
        gt_center_frame = interleave_left_right(output_batch_dict["output_imgs"])
        gt_center_frame =gt_center_frame[:,-6:-4,:,:,:]

        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_center_frame = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_center_frame = torch.clamp(rendered_center_frame,min=0,max=1.0)
        
        end_time1 = time.time()

        
        
        if use_diffix3d:
            # logic here
            # enhance the center frame
            rendered_center_frame = rendered_center_frame
            rendered_center_left = rendered_center_frame[0,0,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_center_right = rendered_center_frame[0,1,:,:,:].permute(1,2,0).cpu().numpy()
            
            rendered_center_left = (rendered_center_left*255).astype(np.uint8)
            rendered_center_right = (rendered_center_right*255).astype(np.uint8)
            rendered_center_left_pil = Image.fromarray(rendered_center_left)
            rendered_center_right_pil = Image.fromarray(rendered_center_right)
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
            
            enhanced_rendered_center_left = np.array(enhanced_rendered_center_left_pil).astype(np.float32)/255.0
            enhanced_rendered_center_right = np.array(enhanced_rendered_center_right_pil).astype(np.float32)/255.0
            
            enhanced_rendered_center_left = torch.from_numpy(enhanced_rendered_center_left).to(rendered_center_frame.device)
            enhanced_rendered_center_right = torch.from_numpy(enhanced_rendered_center_right).to(rendered_center_frame.device)
            enhanced_rendered_center_left = enhanced_rendered_center_left.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_center_right = enhanced_rendered_center_right.permute(2,0,1).unsqueeze(0)
            
            enhanced_rendered_center_frame = torch.cat([enhanced_rendered_center_left,
                                                        enhanced_rendered_center_right],dim=0).unsqueeze(0)
            
            rendered_center_frame = enhanced_rendered_center_frame
        
            
        '''second time inference'''
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=4,
                                                                         matching_nums=3)
        
        input_batch_dict["imgs"][:,2:,:,:,:] = rendered_center_frame
        

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
        interleave_render_c2w = interleave_left_right_pose(render_c2w)
        render_c2w = interleave_render_c2w[:,-4:-2,:,:]
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-4:-2]
        render_fovys = output_batch_dict["output_fovys"][:,-4:-2]
        gt_last_frame = interleave_left_right(output_batch_dict["output_imgs"])
        gt_last_frame =gt_last_frame[:,-4:-2,:,:,:]
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_last_frame = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_last_frame = torch.clamp(rendered_last_frame,min=0,max=1.0)

        if use_diffix3d:
            # logic here
            # enhance the last frame
            rendered_last_frame = rendered_last_frame
            rendered_last_left = rendered_last_frame[0,0,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_last_right = rendered_last_frame[0,1,:,:,:].permute(1,2,0).cpu().numpy()
            
            rendered_last_left = (rendered_last_left*255).astype(np.uint8)
            rendered_last_right = (rendered_last_right*255).astype(np.uint8)
            rendered_last_left_pil = Image.fromarray(rendered_last_left)
            rendered_last_right_pil = Image.fromarray(rendered_last_right)
            width,height = rendered_last_left_pil.size
            
            # get the ref image
            ref_image_left = input_batch_dict["imgs"][0,0,:,:,:].permute(1,2,0).cpu().numpy()
            ref_image_left = (ref_image_left*255).astype(np.uint8)
            ref_image_left_pil = Image.fromarray(ref_image_left)
            ref_image_right = input_batch_dict["imgs"][0,1,:,:,:].permute(1,2,0).cpu().numpy()
            ref_image_right = (ref_image_right*255).astype(np.uint8)
            ref_image_right_pil = Image.fromarray(ref_image_right)

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
            
            enhanced_rendered_last_left = np.array(enhanced_rendered_last_left_pil).astype(np.float32)/255.0
            enhanced_rendered_last_right = np.array(enhanced_rendered_last_right_pil).astype(np.float32)/255.0
            
            enhanced_rendered_last_left = torch.from_numpy(enhanced_rendered_last_left).to(rendered_last_frame.device)
            enhanced_rendered_last_right = torch.from_numpy(enhanced_rendered_last_right).to(rendered_last_frame.device)
            enhanced_rendered_last_left = enhanced_rendered_last_left.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_last_right = enhanced_rendered_last_right.permute(2,0,1).unsqueeze(0)
            
            enhanced_rendered_last_frame = torch.cat([enhanced_rendered_last_left,
                                                        enhanced_rendered_last_right],dim=0).unsqueeze(0)
            
            rendered_last_frame = enhanced_rendered_last_frame
 
            
            
        
        '''third time inference'''
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=6,
                                                                         matching_nums=4)
        
        
        input_batch_dict["imgs"][:,2:4,:,:,:] = rendered_center_frame
        input_batch_dict["imgs"][:,4:,:,:,:] = rendered_last_frame 
        
        
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

    # iteration twice
    def get_additional_bev_novel_views_progressive_iter_once(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        start_images_views = 2,
                                        use_diffix3d=False,
                                        diffix3d_network=None,
                                        use_ref=False,
                                        cfg=None,
                                        vis=False,
                                        rescale_h=2.0,
                                        rescale_w=1.0
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

            current_resolution = [img.shape[-2], img.shape[-1]]
            
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
        
        render_c2w = interleave_render_c2w[:,-6:-2,:,:]
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-6:-2]# [B,6*3]
        render_fovys = output_batch_dict["output_fovys"][:,-6:-2] # [B,6*3]
        gt_center_frame = interleave_left_right(output_batch_dict["output_imgs"])
        gt_center_frame =gt_center_frame[:,-6:-2,:,:,:]
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_center_frame = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_center_frame = torch.clamp(rendered_center_frame,min=0,max=1.0)
        
        # enhanced by the diffusion model.
        if use_diffix3d:
            
            rendered_center_frame = rendered_center_frame
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
            



            
        ''' Second Time Inference for 3DGS Update'''
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=6,
                                                                         matching_nums=4)
        input_batch_dict["imgs"][:,2:,:,:,:] = rendered_center_frame
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
                                                  rescale_h=rescale_h,rescale_w=rescale_w,
                                                  diffix3d_network=diffix3d_network,
                                                  start_images_views=start_images_views,
                                                  use_diffix3d=use_diffix3d)
            
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

    # bev video encoding 
    def bev_video_kitti360(self,
                           batch,
                           cfg=None,
                           rescale_h=3.0,
                           rescale_w=1.0,
                           diffix3d_network=None,
                           start_images_views=2,
                           use_diffix3d=True):

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

            current_resolution = [img.shape[-2], img.shape[-1]]
            
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
        
        render_c2w = interleave_render_c2w[:,-6:-2,:,:]
        intrinsics = input_batch_dict['intrinsics']
        intrinsics = intrinsics.clone()     
        output_intrinsics = intrinsics[:,0:1,:,:].repeat(1,render_c2w.shape[1],1,1)
        render_fovxs = output_batch_dict["output_fovxs"][:,-6:-2]# [B,6*3]
        render_fovys = output_batch_dict["output_fovys"][:,-6:-2] # [B,6*3]
        gt_center_frame = interleave_left_right(output_batch_dict["output_imgs"])
        gt_center_frame =gt_center_frame[:,-6:-2,:,:,:]
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )  

        rendered_results_fuse = render_pkg_fuse
        rendered_center_frame = rendered_results_fuse['image'] # torch.Size([1, V, 3, 224, 832])
        rendered_center_frame = torch.clamp(rendered_center_frame,min=0,max=1.0)
        
        # enhanced by the diffusion model.
        if use_diffix3d:
            
            rendered_center_frame = rendered_center_frame
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
            

            # FIXME
            fusion_rendered_center_frame = rendered_center_frame * 0.64 + gt_center_frame * 0.36
            fusion_rendered_center_frame = torch.clamp(fusion_rendered_center_frame,min=0,max=1.0)
            rendered_center_frame = fusion_rendered_center_frame

            
        ''' Second Time Inference for 3DGS Update'''
        input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,view_num=6,
                                                                         matching_nums=4)
        input_batch_dict["imgs"][:,2:,:,:,:] = rendered_center_frame
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

    
    # FIXME: Please Delete in the Future This Version is just to show the 
    # the potential of the progressive inference.
    def oracle_upper_bound_ablation(self,
                                    batch,
                                    val_result_savedir,
                                    bin_token_list,
                                    start_images_views = 2,
                                    pseudo_ratio_index = [],
                                    use_diffix3d=False,
                                    diffix3d_network=None,
                                    use_ref=False,
                                    cfg=None,
                                    vis=False):
        

        bin_token_name = bin_token_list[0][:-4]
        
        # get the input and the reference input
        input_batch_dict,output_batch_dict =self.prepare_tripleview_by_ratio_index(
                                                batch=batch,
                                                pseudo_ratio_index=pseudo_ratio_index)
        

        img = input_batch_dict["imgs"] #[B,6,3,H,W]
        

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

    #FIXME: Please Delete in the Future, This Version is Just to Test the Difix3D Performance.
    def test_current_difix3d_performance(
                                        self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        psuedo_ratio=[],
                                        start_images_views = 2,
                                        use_diffix3d=False,
                                        diffix3d_network=None,
                                        use_ref=False,
                                        cfg=None,
                                        vis=False):
        
        # for the psnr and ssim evaluations for both raw and enhanced results.
        raw_results_stat = {
            "0": {},
            "0.125": {},
            "0.25": {},
            "0.33": {},
            "0.5": {},
            "0.66": {},
            "0.75": {},
            "1.0": {},
        }
        
        enhanced_results_stat = {
            "0": {},
            "0.125": {},
            "0.25": {},
            "0.33": {},
            "0.5": {},
            "0.66": {},
            "0.75": {},
            "1.0": {},
            
        }
        
        saved_images_dict = {            
        }
        
        
        
        view_num = 2
        matching_nums = 2
        # get the input and the reference input
        input_batch_dict,output_batch_dict = self.prepare_input_multiview(batch=batch,view_num=view_num,
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
        
        
        # change the ordered.
        rendered_images_fusion = interleave_left_right(rendered_color_fuse)
        rendered_depth_fusion = interleave_left_right_depth(rendered_depth_fuse)
        


        # GT Information For Supervision
        output_rgb = output_batch_dict['output_imgs']
        rgb_gt = output_rgb
        pseudo_depth_gt = output_batch_dict['output_depths_m']
        sparse_depth_gt = output_batch_dict['output_sparse_depth']
        valid_mask_01 = sparse_depth_gt>0
        valid_mask_01_float = valid_mask_01.float()
        
        # use this
        fusion_pseudo_with_sparse_gt = valid_mask_01_float * sparse_depth_gt + (1-valid_mask_01_float) * pseudo_depth_gt 

        
        rendered_images_gt = rgb_gt
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
        
        
        rest_rendered_images_all_stereo = rendered_images_all_stereo[:,:-6,:,:,:]
        rest_gt_images_all_stereo = gt_images_all_stereo[:,:-6,:,:,:]
        
        
        # for the selection of all the views
        candidates_rendered_images_all_stereo = torch.cat([rendered_images_first_stereo, 
                                                           rest_rendered_images_all_stereo,
                                                           rendered_images_last_stereo, 
                                                           ], dim=1)
        
        candidates_gt_images_all_stereo = torch.cat([gt_images_first_stereo, 
                                                     rest_gt_images_all_stereo,
                                                     gt_images_last_stereo, 
                                                     ], dim=1)
        
        
        left_ref = input_batch_dict["imgs"][0,0,:,:,:].permute(1,2,0).cpu().numpy()
        left_ref = (left_ref*255).astype(np.uint8)
        left_ref_pil = Image.fromarray(left_ref)
        
        right_ref = input_batch_dict["imgs"][0,1,:,:,:].permute(1,2,0).cpu().numpy()
        right_ref = (right_ref*255).astype(np.uint8)
        right_ref_pil = Image.fromarray(right_ref)
        
        
        # 0 ----> First
        # raw
        first_rgb_eval_info_raw = metrics_mean(pred=rendered_images_first_stereo,
                                           gt=gt_images_first_stereo)
        # enhance 
        raw_candiates_0_left = rendered_images_first_stereo[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_0_left= (raw_candiates_0_left*255).astype(np.uint8)
        raw_candiates_0_left_pil = Image.fromarray(raw_candiates_0_left)
        
        raw_candiates_0_right = rendered_images_first_stereo[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_0_right= (raw_candiates_0_right*255).astype(np.uint8)
        raw_candiates_0_right_pil = Image.fromarray(raw_candiates_0_right)
        
                

        with torch.no_grad():
            enhanced_candiates_0_left_pil = diffix3d_network.sample(
                raw_candiates_0_left_pil,
                height=112,
                width=544,
                ref_image=left_ref_pil,
                prompt=cfg.prompt
            )
            enhanced_candiates_0_right_pil = diffix3d_network.sample(
                raw_candiates_0_right_pil,
                height=112,
                width=544,
                ref_image=right_ref_pil,
                prompt=cfg.prompt
            )
        
        enhanced_candiates_0_left_tensor = convert_pil_to_tensor(enhanced_candiates_0_left_pil)
        enhanced_candiates_0_right_tensor = convert_pil_to_tensor(enhanced_candiates_0_right_pil)
        enhanced_candiates_0_stereo_tensor = torch.cat([enhanced_candiates_0_left_tensor, 
                                                        enhanced_candiates_0_right_tensor], 
                                                       dim=1)
        enhanced_candiates_0_stereo_tensor = enhanced_candiates_0_stereo_tensor.type_as(gt_images_first_stereo)
        
        first_rgb_eval_info_enhances = metrics_mean(pred=enhanced_candiates_0_stereo_tensor,
                                           gt=gt_images_first_stereo)
        
        first_rgb_lpips_raw = first_rgb_eval_info_raw['lpips'].data.item()
        first_rgb_ssim_raw = first_rgb_eval_info_raw['ssim'].data.item()
        first_rgb_psnr_raw = first_rgb_eval_info_raw['psnr'].data.item()
        
        first_rgb_lpips_enhances = first_rgb_eval_info_enhances['lpips'].data.item()
        first_rgb_ssim_enhances = first_rgb_eval_info_enhances['ssim'].data.item()
        first_rgb_psnr_enhances = first_rgb_eval_info_enhances['psnr'].data.item()
        
        raw_results_stat["0"]["lpips"] = first_rgb_lpips_raw
        raw_results_stat["0"]["ssim"] = first_rgb_ssim_raw
        raw_results_stat["0"]["psnr"] = first_rgb_psnr_raw
        enhanced_results_stat["0"]["lpips"] = first_rgb_lpips_enhances
        enhanced_results_stat["0"]["ssim"] = first_rgb_ssim_enhances
        enhanced_results_stat["0"]["psnr"] = first_rgb_psnr_enhances
        
        # saved images
        candidates_0_saved_images = torch.cat([rendered_images_first_stereo[0,0,:,:,:],
                                               enhanced_candiates_0_left_tensor[0,0,:,:,:].type_as(rendered_images_first_stereo[0,0,:,:,:]),
                                               gt_images_first_stereo[0,0,:,:,:]],
                                              dim=1)
        candidates_0_saved_images = candidates_0_saved_images.permute(1,2,0).cpu().numpy()
        candidates_0_saved_images = (candidates_0_saved_images*255).astype(np.uint8)        
        saved_images_dict["0"] = candidates_0_saved_images
        
        # 0.125
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.125)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_1_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_1_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_1_psuedo_raw = metrics_mean(pred=candidates_1_psuedo_images,
                                           gt=candidates_1_psuedo_gt)
        
        # enhanced
        raw_candiates_1_psuedo_left = candidates_1_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_1_psuedo_left= (raw_candiates_1_psuedo_left*255).astype(np.uint8)
        raw_candiates_1_psuedo_left_pil = Image.fromarray(raw_candiates_1_psuedo_left)
        
        raw_candiates_1_psuedo_right = candidates_1_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_1_psuedo_right= (raw_candiates_1_psuedo_right*255).astype(np.uint8)
        raw_candiates_1_psuedo_right_pil = Image.fromarray(raw_candiates_1_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_1_psuedo_left_pil = diffix3d_network.sample(
                raw_candiates_1_psuedo_left_pil,
                height=112,
                width=544,
                ref_image=left_ref_pil,
                prompt=cfg.prompt
            )
            enhanced_candiates_1_psuedo_right_pil = diffix3d_network.sample(
                raw_candiates_1_psuedo_right_pil,
                height=112,
                width=544,
                ref_image=right_ref_pil,
                prompt=cfg.prompt
            )
        
        enhanced_candiates_1_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_1_psuedo_left_pil)
        enhanced_candiates_1_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_1_psuedo_right_pil)
        enhanced_candiates_1_psuedo_stereo_tensor = torch.cat([enhanced_candiates_1_psuedo_left_tensor, 
                                                                enhanced_candiates_1_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_1_psuedo_stereo_tensor = enhanced_candiates_1_psuedo_stereo_tensor.type_as(candidates_1_psuedo_gt)
        eval_info_1_psuedo_enhances = metrics_mean(pred=enhanced_candiates_1_psuedo_stereo_tensor,
                                           gt=candidates_1_psuedo_gt)
        

        raw_results_stat["0.125"]["lpips"] = eval_info_1_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.125"]["ssim"] = eval_info_1_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.125"]["psnr"] = eval_info_1_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.125"]["lpips"] = eval_info_1_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.125"]["ssim"] = eval_info_1_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.125"]["psnr"] = eval_info_1_psuedo_enhances['psnr'].data.item()
        
    
        candidates_1_saved_images = torch.cat([candidates_1_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_1_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_1_psuedo_images[0,0,:,:,:]),
                                               candidates_1_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_1_saved_images = candidates_1_saved_images.permute(1,2,0).cpu().numpy()
        candidates_1_saved_images = (candidates_1_saved_images*255).astype(np.uint8)
        saved_images_dict["0.125"] = candidates_1_saved_images
        
    
    
        # 0.25
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.25)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_2_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_2_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_2_psuedo_raw = metrics_mean(pred=candidates_2_psuedo_images,
                                           gt=candidates_2_psuedo_gt)
        
        # enhanced
        raw_candiates_2_psuedo_left = candidates_2_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_2_psuedo_left= (raw_candiates_2_psuedo_left*255).astype(np.uint8)
        raw_candiates_2_psuedo_left_pil = Image.fromarray(raw_candiates_2_psuedo_left)
        
        raw_candiates_2_psuedo_right = candidates_2_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_2_psuedo_right= (raw_candiates_2_psuedo_right*255).astype(np.uint8)
        raw_candiates_2_psuedo_right_pil = Image.fromarray(raw_candiates_2_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_2_psuedo_left_pil = diffix3d_network.sample(
                raw_candiates_2_psuedo_left_pil,
                height=112,
                width=544,
                ref_image=left_ref_pil,
                prompt=cfg.prompt
            )
            enhanced_candiates_2_psuedo_right_pil = diffix3d_network.sample(
                raw_candiates_2_psuedo_right_pil,
                height=112,
                width=544,
                ref_image=right_ref_pil,
                prompt=cfg.prompt
            )
        
        enhanced_candiates_2_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_2_psuedo_left_pil)
        enhanced_candiates_2_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_2_psuedo_right_pil)
        enhanced_candiates_2_psuedo_stereo_tensor = torch.cat([enhanced_candiates_2_psuedo_left_tensor, 
                                                                enhanced_candiates_2_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_2_psuedo_stereo_tensor = enhanced_candiates_2_psuedo_stereo_tensor.type_as(candidates_2_psuedo_gt)
        eval_info_2_psuedo_enhances = metrics_mean(pred=enhanced_candiates_2_psuedo_stereo_tensor,
                                           gt=candidates_2_psuedo_gt)
        
        
        raw_results_stat["0.25"]["lpips"] = eval_info_2_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.25"]["ssim"] = eval_info_2_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.25"]["psnr"] = eval_info_2_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.25"]["lpips"] = eval_info_2_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.25"]["ssim"] = eval_info_2_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.25"]["psnr"] = eval_info_2_psuedo_enhances['psnr'].data.item()

        candidates_2_saved_images = torch.cat([candidates_2_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_2_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_2_psuedo_images[0,0,:,:,:]),
                                               candidates_2_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_2_saved_images = candidates_2_saved_images.permute(1,2,0).cpu().numpy()
        candidates_2_saved_images = (candidates_2_saved_images*255).astype(np.uint8)
        saved_images_dict["0.25"] = candidates_2_saved_images
        
        
        
        # 0.33
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.33)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_3_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_3_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_3_psuedo_raw = metrics_mean(pred=candidates_3_psuedo_images,
                                           gt=candidates_3_psuedo_gt)
        
        # enhanced
        raw_candiates_3_psuedo_left = candidates_3_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_3_psuedo_left= (raw_candiates_3_psuedo_left*255).astype(np.uint8)
        raw_candiates_3_psuedo_left_pil = Image.fromarray(raw_candiates_3_psuedo_left)
        
        raw_candiates_3_psuedo_right = candidates_3_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_3_psuedo_right= (raw_candiates_3_psuedo_right*255).astype(np.uint8)
        raw_candiates_3_psuedo_right_pil = Image.fromarray(raw_candiates_3_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_3_psuedo_left_pil = diffix3d_network.sample(
                raw_candiates_3_psuedo_left_pil,
                height=112,
                width=544,
                ref_image=left_ref_pil,
                prompt=cfg.prompt
            )
            enhanced_candiates_3_psuedo_right_pil = diffix3d_network.sample(
                raw_candiates_3_psuedo_right_pil,
                height=112,
                width=544,
                ref_image=right_ref_pil,
                prompt=cfg.prompt
            )
        
        enhanced_candiates_3_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_3_psuedo_left_pil)
        enhanced_candiates_3_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_3_psuedo_right_pil)
        enhanced_candiates_3_psuedo_stereo_tensor = torch.cat([enhanced_candiates_3_psuedo_left_tensor, 
                                                                enhanced_candiates_3_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_3_psuedo_stereo_tensor = enhanced_candiates_3_psuedo_stereo_tensor.type_as(candidates_2_psuedo_gt)
        eval_info_3_psuedo_enhances = metrics_mean(
                                            pred=enhanced_candiates_3_psuedo_stereo_tensor,
                                           gt=candidates_3_psuedo_gt)
        
        raw_results_stat["0.33"]["lpips"] = eval_info_3_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.33"]["ssim"] = eval_info_3_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.33"]["psnr"] = eval_info_3_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.33"]["lpips"] = eval_info_3_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.33"]["ssim"] = eval_info_3_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.33"]["psnr"] = eval_info_3_psuedo_enhances['psnr'].data.item()
        
        candidates_3_saved_images = torch.cat([candidates_3_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_3_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_3_psuedo_images[0,0,:,:,:]),
                                               candidates_3_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_3_saved_images = candidates_3_saved_images.permute(1,2,0).cpu().numpy()
        candidates_3_saved_images = (candidates_3_saved_images*255).astype(np.uint8)
        saved_images_dict["0.33"] = candidates_3_saved_images
        
    
        # 0.5 ----> Center
        # raw
        center_rgb_eval_info_raw = metrics_mean(pred=rendered_images_center_stereo,
                                           gt=gt_images_center_stereo)
        # enhance 
        raw_center_left = rendered_images_center_stereo[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_center_left = (raw_center_left*255).astype(np.uint8)
        raw_center_left_pil = Image.fromarray(raw_center_left)

        raw_center_right = rendered_images_center_stereo[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_center_right = (raw_center_right*255).astype(np.uint8)
        raw_center_right_pil = Image.fromarray(raw_center_right)


        with torch.no_grad():
            enhanced_center_left_pil = diffix3d_network.sample(
                raw_center_left_pil,
                height=112,
                width=544,
                ref_image=left_ref_pil,
                prompt=cfg.prompt
            )
            enhanced_center_right_pil = diffix3d_network.sample(
                raw_center_right_pil,
                height=112,
                width=544,
                ref_image=right_ref_pil,
                prompt=cfg.prompt
            )

        enhanced_center_left_tensor = convert_pil_to_tensor(enhanced_center_left_pil)
        enhanced_center_right_tensor = convert_pil_to_tensor(enhanced_center_right_pil)
        
        enhanced_center_tensor = torch.cat([enhanced_center_left_tensor, 
                                            enhanced_center_right_tensor], 
                                                       dim=1)
        
        enhanced_center_tensor = enhanced_center_tensor.type_as(gt_images_first_stereo)
        
        center_rgb_eval_info_enhances = metrics_mean(pred=enhanced_center_tensor,
                                           gt=gt_images_center_stereo)
        
        
        raw_results_stat["0.5"]["lpips"] = center_rgb_eval_info_raw['lpips'].data.item()
        raw_results_stat["0.5"]["ssim"] = center_rgb_eval_info_raw['ssim'].data.item()
        raw_results_stat["0.5"]["psnr"] = center_rgb_eval_info_raw['psnr'].data.item()
        enhanced_results_stat["0.5"]["lpips"] = center_rgb_eval_info_enhances['lpips'].data.item()
        enhanced_results_stat["0.5"]["ssim"] = center_rgb_eval_info_enhances['ssim'].data.item()
        enhanced_results_stat["0.5"]["psnr"] = center_rgb_eval_info_enhances['psnr'].data.item()
        
        candidates_center_saved_images = torch.cat([rendered_images_center_stereo[0,0,:,:,:],
                                               enhanced_center_left_tensor[0,0,:,:,:].type_as(rendered_images_center_stereo[0,0,:,:,:]),
                                               gt_images_center_stereo[0,0,:,:,:]],
                                              dim=1)
        candidates_center_saved_images = candidates_center_saved_images.permute(1,2,0).cpu().numpy()
        candidates_center_saved_images = (candidates_center_saved_images*255).astype(np.uint8)
        saved_images_dict["0.5"] = candidates_center_saved_images

        
        # 0.66 
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.66)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_4_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_4_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_4_psuedo_raw = metrics_mean(pred=candidates_4_psuedo_images,
                                           gt=candidates_4_psuedo_gt)
        
        # enhanced
        raw_candiates_4_psuedo_left = candidates_4_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_4_psuedo_left= (raw_candiates_4_psuedo_left*255).astype(np.uint8)
        raw_candiates_4_psuedo_left_pil = Image.fromarray(raw_candiates_4_psuedo_left)
        
        raw_candiates_4_psuedo_right = candidates_4_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_4_psuedo_right= (raw_candiates_4_psuedo_right*255).astype(np.uint8)
        raw_candiates_4_psuedo_right_pil = Image.fromarray(raw_candiates_4_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_4_psuedo_left_pil = diffix3d_network.sample(
                raw_candiates_4_psuedo_left_pil,
                height=112,
                width=544,
                ref_image=left_ref_pil,
                prompt=cfg.prompt
            )
            enhanced_candiates_4_psuedo_right_pil = diffix3d_network.sample(
                raw_candiates_4_psuedo_right_pil,
                height=112,
                width=544,
                ref_image=right_ref_pil,
                prompt=cfg.prompt
            )
        
        enhanced_candiates_4_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_4_psuedo_left_pil)
        enhanced_candiates_4_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_4_psuedo_right_pil)
        enhanced_candiates_4_psuedo_stereo_tensor = torch.cat([enhanced_candiates_4_psuedo_left_tensor, 
                                                                enhanced_candiates_4_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_4_psuedo_stereo_tensor = enhanced_candiates_4_psuedo_stereo_tensor.type_as(candidates_2_psuedo_gt)
        eval_info_4_psuedo_enhances = metrics_mean(
                                            pred=enhanced_candiates_4_psuedo_stereo_tensor,
                                           gt=candidates_4_psuedo_gt)
        
        raw_results_stat["0.66"]["lpips"] = eval_info_4_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.66"]["ssim"] = eval_info_4_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.66"]["psnr"] = eval_info_4_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.66"]["lpips"] = eval_info_4_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.66"]["ssim"] = eval_info_4_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.66"]["psnr"] = eval_info_4_psuedo_enhances['psnr'].data.item()
        
        
        candidates_4_saved_images = torch.cat([candidates_4_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_4_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_4_psuedo_images[0,0,:,:,:]),
                                               candidates_4_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_4_saved_images = candidates_4_saved_images.permute(1,2,0).cpu().numpy()
        candidates_4_saved_images = (candidates_4_saved_images*255).astype(np.uint8)
        saved_images_dict["0.66"] = candidates_4_saved_images
                
        # 0.75  
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.75)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_5_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_5_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_5_psuedo_raw = metrics_mean(pred=candidates_5_psuedo_images,
                                           gt=candidates_5_psuedo_gt)
        
        # enhanced
        raw_candiates_5_psuedo_left = candidates_5_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_5_psuedo_left= (raw_candiates_5_psuedo_left*255).astype(np.uint8)
        raw_candiates_5_psuedo_left_pil = Image.fromarray(raw_candiates_5_psuedo_left)
        
        raw_candiates_5_psuedo_right = candidates_5_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_5_psuedo_right= (raw_candiates_5_psuedo_right*255).astype(np.uint8)
        raw_candiates_5_psuedo_right_pil = Image.fromarray(raw_candiates_5_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_5_psuedo_left_pil = diffix3d_network.sample(
                raw_candiates_5_psuedo_left_pil,
                height=112,
                width=544,
                ref_image=left_ref_pil,
                prompt=cfg.prompt
            )
            enhanced_candiates_5_psuedo_right_pil = diffix3d_network.sample(
                raw_candiates_5_psuedo_right_pil,
                height=112,
                width=544,
                ref_image=right_ref_pil,
                prompt=cfg.prompt
            )
        
        enhanced_candiates_5_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_5_psuedo_left_pil)
        enhanced_candiates_5_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_5_psuedo_right_pil)
        enhanced_candiates_5_psuedo_stereo_tensor = torch.cat([enhanced_candiates_5_psuedo_left_tensor, 
                                                                enhanced_candiates_5_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_5_psuedo_stereo_tensor = enhanced_candiates_5_psuedo_stereo_tensor.type_as(candidates_2_psuedo_gt)
        eval_info_5_psuedo_enhances = metrics_mean(
                                            pred=enhanced_candiates_5_psuedo_stereo_tensor,
                                           gt=candidates_5_psuedo_gt)
    
        raw_results_stat["0.75"]["lpips"] = eval_info_5_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.75"]["ssim"] = eval_info_5_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.75"]["psnr"] = eval_info_5_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.75"]["lpips"] = eval_info_5_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.75"]["ssim"] = eval_info_5_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.75"]["psnr"] = eval_info_5_psuedo_enhances['psnr'].data.item()
        
        
        candidates_5_saved_images = torch.cat([candidates_5_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_5_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_5_psuedo_images[0,0,:,:,:]),
                                               candidates_5_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_5_saved_images = candidates_5_saved_images.permute(1,2,0).cpu().numpy()
        candidates_5_saved_images = (candidates_5_saved_images*255).astype(np.uint8)
        saved_images_dict["0.75"] = candidates_5_saved_images
        
        
    
        # 1.0 -----> Last
       # raw
        last_rgb_eval_info_raw = metrics_mean(pred=rendered_images_last_stereo,
                                           gt=gt_images_last_stereo)
        # enhance 
        raw_last_left = rendered_images_last_stereo[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_last_left = (raw_last_left*255).astype(np.uint8)
        raw_last_left_pil = Image.fromarray(raw_last_left)

        raw_last_right = rendered_images_last_stereo[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_last_right = (raw_last_right*255).astype(np.uint8)
        raw_last_right_pil = Image.fromarray(raw_last_right)


        with torch.no_grad():
            enhanced_last_left_pil = diffix3d_network.sample(
                raw_last_left_pil,
                height=112,
                width=544,
                ref_image=left_ref_pil,
                prompt=cfg.prompt
            )
            enhanced_last_right_pil = diffix3d_network.sample(
                raw_last_right_pil,
                height=112,
                width=544,
                ref_image=right_ref_pil,
                prompt=cfg.prompt
            )

        enhanced_last_left_tensor = convert_pil_to_tensor(enhanced_last_left_pil)
        enhanced_last_right_tensor = convert_pil_to_tensor(enhanced_last_right_pil)
        
        enhanced_last_tensor = torch.cat([enhanced_last_left_tensor, 
                                            enhanced_last_right_tensor], 
                                                       dim=1)
        
        enhanced_last_tensor = enhanced_last_tensor.type_as(gt_images_first_stereo)
        
        last_rgb_eval_info_enhances = metrics_mean(pred=enhanced_last_tensor,
                                           gt=gt_images_last_stereo)
        
        
        raw_results_stat["1.0"]["lpips"] = last_rgb_eval_info_raw['lpips'].data.item()
        raw_results_stat["1.0"]["ssim"] = last_rgb_eval_info_raw['ssim'].data.item()
        raw_results_stat["1.0"]["psnr"] = last_rgb_eval_info_raw['psnr'].data.item()
        enhanced_results_stat["1.0"]["lpips"] = last_rgb_eval_info_enhances['lpips'].data.item()
        enhanced_results_stat["1.0"]["ssim"] = last_rgb_eval_info_enhances['ssim'].data.item()
        enhanced_results_stat["1.0"]["psnr"] = last_rgb_eval_info_enhances['psnr'].data.item()
        
        
        candidates_last_saved_images = torch.cat([rendered_images_last_stereo[0,0,:,:,:],
                                               enhanced_last_left_tensor[0,0,:,:,:].type_as(rendered_images_last_stereo[0,0,:,:,:]),
                                               gt_images_last_stereo[0,0,:,:,:]],
                                              dim=1)
        candidates_last_saved_images = candidates_last_saved_images.permute(1,2,0).cpu().numpy()
        candidates_last_saved_images = (candidates_last_saved_images*255).astype(np.uint8)
        saved_images_dict["1.0"] = candidates_last_saved_images
        
        
        return raw_results_stat, enhanced_results_stat, saved_images_dict
    
    # FIXME: Please Delete in the Future, This Version is Just to Test the Difixi3D Performance.
    def test_official_difix3d_ref_performance(
                                        self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        psuedo_ratio=[],
                                        start_images_views = 2,
                                        use_diffix3d=False,
                                        diffix3d_network=None,
                                        use_ref=False,
                                        cfg=None,
                                        vis=False):
        
        # for the psnr and ssim evaluations for both raw and enhanced results.
        raw_results_stat = {
            "0": {},
            "0.125": {},
            "0.25": {},
            "0.33": {},
            "0.5": {},
            "0.66": {},
            "0.75": {},
            "1.0": {},
        }
        
        enhanced_results_stat = {
            "0": {},
            "0.125": {},
            "0.25": {},
            "0.33": {},
            "0.5": {},
            "0.66": {},
            "0.75": {},
            "1.0": {},
            
        }
        
        saved_images_dict = {            
        }
        
        
        
        view_num = 2
        matching_nums = 2
        # get the input and the reference input
        input_batch_dict,output_batch_dict = self.prepare_input_multiview(batch=batch,view_num=view_num,
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
        
        
        # change the ordered.
        rendered_images_fusion = interleave_left_right(rendered_color_fuse)
        rendered_depth_fusion = interleave_left_right_depth(rendered_depth_fuse)
        


        # GT Information For Supervision
        output_rgb = output_batch_dict['output_imgs']
        rgb_gt = output_rgb
        pseudo_depth_gt = output_batch_dict['output_depths_m']
        sparse_depth_gt = output_batch_dict['output_sparse_depth']
        valid_mask_01 = sparse_depth_gt>0
        valid_mask_01_float = valid_mask_01.float()
        
        # use this
        fusion_pseudo_with_sparse_gt = valid_mask_01_float * sparse_depth_gt + (1-valid_mask_01_float) * pseudo_depth_gt 

        
        rendered_images_gt = rgb_gt
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
        
        
        rest_rendered_images_all_stereo = rendered_images_all_stereo[:,:-6,:,:,:]
        rest_gt_images_all_stereo = gt_images_all_stereo[:,:-6,:,:,:]
        
        
        # for the selection of all the views
        candidates_rendered_images_all_stereo = torch.cat([rendered_images_first_stereo, 
                                                           rest_rendered_images_all_stereo,
                                                           rendered_images_last_stereo, 
                                                           ], dim=1)
        
        candidates_gt_images_all_stereo = torch.cat([gt_images_first_stereo, 
                                                     rest_gt_images_all_stereo,
                                                     gt_images_last_stereo, 
                                                     ], dim=1)
        
        
        left_ref = input_batch_dict["imgs"][0,0,:,:,:].permute(1,2,0).cpu().numpy()
        left_ref = (left_ref*255).astype(np.uint8)
        left_ref_pil = Image.fromarray(left_ref)
        
        right_ref = input_batch_dict["imgs"][0,1,:,:,:].permute(1,2,0).cpu().numpy()
        right_ref = (right_ref*255).astype(np.uint8)
        right_ref_pil = Image.fromarray(right_ref)
        
        
        # 0 ----> First
        # raw
        first_rgb_eval_info_raw = metrics_mean(pred=rendered_images_first_stereo,
                                           gt=gt_images_first_stereo)
        # enhance 
        raw_candiates_0_left = rendered_images_first_stereo[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_0_left= (raw_candiates_0_left*255).astype(np.uint8)
        raw_candiates_0_left_pil = Image.fromarray(raw_candiates_0_left)
        
        raw_candiates_0_right = rendered_images_first_stereo[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_0_right= (raw_candiates_0_right*255).astype(np.uint8)
        raw_candiates_0_right_pil = Image.fromarray(raw_candiates_0_right)
        
        


        with torch.no_grad():
            
            enhanced_candiates_0_left_pil = diffix3d_network(cfg.prompt, 
                                image=raw_candiates_0_left_pil, 
                                ref_image=left_ref_pil, 
                                num_inference_steps=1, 
             timesteps=[199], guidance_scale=0.0).images[0]
            
            enhanced_candiates_0_right_pil = diffix3d_network(cfg.prompt, 
                                image=raw_candiates_0_right_pil, 
                                ref_image=right_ref_pil, 
                                num_inference_steps=1, 
             timesteps=[199], guidance_scale=0.0).images[0]
            
        
        enhanced_candiates_0_left_tensor = convert_pil_to_tensor(enhanced_candiates_0_left_pil)
        enhanced_candiates_0_right_tensor = convert_pil_to_tensor(enhanced_candiates_0_right_pil)
        enhanced_candiates_0_stereo_tensor = torch.cat([enhanced_candiates_0_left_tensor, 
                                                        enhanced_candiates_0_right_tensor], 
                                                       dim=1)
        enhanced_candiates_0_stereo_tensor = enhanced_candiates_0_stereo_tensor.type_as(gt_images_first_stereo)
        
        first_rgb_eval_info_enhances = metrics_mean(pred=enhanced_candiates_0_stereo_tensor,
                                           gt=gt_images_first_stereo)
        
        first_rgb_lpips_raw = first_rgb_eval_info_raw['lpips'].data.item()
        first_rgb_ssim_raw = first_rgb_eval_info_raw['ssim'].data.item()
        first_rgb_psnr_raw = first_rgb_eval_info_raw['psnr'].data.item()
        
        first_rgb_lpips_enhances = first_rgb_eval_info_enhances['lpips'].data.item()
        first_rgb_ssim_enhances = first_rgb_eval_info_enhances['ssim'].data.item()
        first_rgb_psnr_enhances = first_rgb_eval_info_enhances['psnr'].data.item()
        
        raw_results_stat["0"]["lpips"] = first_rgb_lpips_raw
        raw_results_stat["0"]["ssim"] = first_rgb_ssim_raw
        raw_results_stat["0"]["psnr"] = first_rgb_psnr_raw
        enhanced_results_stat["0"]["lpips"] = first_rgb_lpips_enhances
        enhanced_results_stat["0"]["ssim"] = first_rgb_ssim_enhances
        enhanced_results_stat["0"]["psnr"] = first_rgb_psnr_enhances
        
        
        # saved images
        candidates_0_saved_images = torch.cat([rendered_images_first_stereo[0,0,:,:,:],
                                               enhanced_candiates_0_left_tensor[0,0,:,:,:].type_as(rendered_images_first_stereo[0,0,:,:,:]),
                                               gt_images_first_stereo[0,0,:,:,:]],
                                              dim=1)
        candidates_0_saved_images = candidates_0_saved_images.permute(1,2,0).cpu().numpy()
        candidates_0_saved_images = (candidates_0_saved_images*255).astype(np.uint8)        
        saved_images_dict["0"] = candidates_0_saved_images
        

        # 0.125
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.125)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_1_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_1_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_1_psuedo_raw = metrics_mean(pred=candidates_1_psuedo_images,
                                           gt=candidates_1_psuedo_gt)
        
        # enhanced
        raw_candiates_1_psuedo_left = candidates_1_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_1_psuedo_left= (raw_candiates_1_psuedo_left*255).astype(np.uint8)
        raw_candiates_1_psuedo_left_pil = Image.fromarray(raw_candiates_1_psuedo_left)
        
        raw_candiates_1_psuedo_right = candidates_1_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_1_psuedo_right= (raw_candiates_1_psuedo_right*255).astype(np.uint8)
        raw_candiates_1_psuedo_right_pil = Image.fromarray(raw_candiates_1_psuedo_right)
        
        
        with torch.no_grad():
                        
            enhanced_candiates_1_psuedo_left_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_1_psuedo_left_pil, 
                ref_image=left_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
            enhanced_candiates_1_psuedo_right_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_1_psuedo_right_pil, 
                ref_image=right_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
        
        enhanced_candiates_1_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_1_psuedo_left_pil)
        enhanced_candiates_1_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_1_psuedo_right_pil)
        enhanced_candiates_1_psuedo_stereo_tensor = torch.cat([enhanced_candiates_1_psuedo_left_tensor, 
                                                                enhanced_candiates_1_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_1_psuedo_stereo_tensor = enhanced_candiates_1_psuedo_stereo_tensor.type_as(candidates_1_psuedo_gt)
        eval_info_1_psuedo_enhances = metrics_mean(pred=enhanced_candiates_1_psuedo_stereo_tensor,
                                           gt=candidates_1_psuedo_gt)
        

        raw_results_stat["0.125"]["lpips"] = eval_info_1_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.125"]["ssim"] = eval_info_1_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.125"]["psnr"] = eval_info_1_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.125"]["lpips"] = eval_info_1_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.125"]["ssim"] = eval_info_1_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.125"]["psnr"] = eval_info_1_psuedo_enhances['psnr'].data.item()
        
    
        candidates_1_saved_images = torch.cat([candidates_1_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_1_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_1_psuedo_images[0,0,:,:,:]),
                                               candidates_1_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_1_saved_images = candidates_1_saved_images.permute(1,2,0).cpu().numpy()
        candidates_1_saved_images = (candidates_1_saved_images*255).astype(np.uint8)
        saved_images_dict["0.125"] = candidates_1_saved_images
        
    
        # 0.25
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.25)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_2_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_2_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_2_psuedo_raw = metrics_mean(pred=candidates_2_psuedo_images,
                                           gt=candidates_2_psuedo_gt)
        
        # enhanced
        raw_candiates_2_psuedo_left = candidates_2_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_2_psuedo_left= (raw_candiates_2_psuedo_left*255).astype(np.uint8)
        raw_candiates_2_psuedo_left_pil = Image.fromarray(raw_candiates_2_psuedo_left)
        
        raw_candiates_2_psuedo_right = candidates_2_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_2_psuedo_right= (raw_candiates_2_psuedo_right*255).astype(np.uint8)
        raw_candiates_2_psuedo_right_pil = Image.fromarray(raw_candiates_2_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_2_psuedo_left_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_2_psuedo_left_pil, 
                ref_image=left_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
            enhanced_candiates_2_psuedo_right_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_2_psuedo_right_pil, 
                ref_image=right_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
        
        
        enhanced_candiates_2_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_2_psuedo_left_pil)
        enhanced_candiates_2_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_2_psuedo_right_pil)
        enhanced_candiates_2_psuedo_stereo_tensor = torch.cat([enhanced_candiates_2_psuedo_left_tensor, 
                                                                enhanced_candiates_2_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_2_psuedo_stereo_tensor = enhanced_candiates_2_psuedo_stereo_tensor.type_as(candidates_2_psuedo_gt)
        eval_info_2_psuedo_enhances = metrics_mean(pred=enhanced_candiates_2_psuedo_stereo_tensor,
                                           gt=candidates_2_psuedo_gt)
        
        
        raw_results_stat["0.25"]["lpips"] = eval_info_2_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.25"]["ssim"] = eval_info_2_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.25"]["psnr"] = eval_info_2_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.25"]["lpips"] = eval_info_2_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.25"]["ssim"] = eval_info_2_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.25"]["psnr"] = eval_info_2_psuedo_enhances['psnr'].data.item()

        candidates_2_saved_images = torch.cat([candidates_2_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_2_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_2_psuedo_images[0,0,:,:,:]),
                                               candidates_2_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_2_saved_images = candidates_2_saved_images.permute(1,2,0).cpu().numpy()
        candidates_2_saved_images = (candidates_2_saved_images*255).astype(np.uint8)
        saved_images_dict["0.25"] = candidates_2_saved_images
        
        
        
        # 0.33
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.33)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_3_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_3_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_3_psuedo_raw = metrics_mean(pred=candidates_3_psuedo_images,
                                           gt=candidates_3_psuedo_gt)
        
        # enhanced
        raw_candiates_3_psuedo_left = candidates_3_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_3_psuedo_left= (raw_candiates_3_psuedo_left*255).astype(np.uint8)
        raw_candiates_3_psuedo_left_pil = Image.fromarray(raw_candiates_3_psuedo_left)
        
        raw_candiates_3_psuedo_right = candidates_3_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_3_psuedo_right= (raw_candiates_3_psuedo_right*255).astype(np.uint8)
        raw_candiates_3_psuedo_right_pil = Image.fromarray(raw_candiates_3_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_3_psuedo_left_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_3_psuedo_left_pil, 
                ref_image=left_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
            enhanced_candiates_3_psuedo_right_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_3_psuedo_right_pil, 
                ref_image=right_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
        
        enhanced_candiates_3_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_3_psuedo_left_pil)
        enhanced_candiates_3_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_3_psuedo_right_pil)
        enhanced_candiates_3_psuedo_stereo_tensor = torch.cat([enhanced_candiates_3_psuedo_left_tensor, 
                                                                enhanced_candiates_3_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_3_psuedo_stereo_tensor = enhanced_candiates_3_psuedo_stereo_tensor.type_as(candidates_2_psuedo_gt)
        eval_info_3_psuedo_enhances = metrics_mean(
                                            pred=enhanced_candiates_3_psuedo_stereo_tensor,
                                           gt=candidates_3_psuedo_gt)
        
        raw_results_stat["0.33"]["lpips"] = eval_info_3_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.33"]["ssim"] = eval_info_3_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.33"]["psnr"] = eval_info_3_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.33"]["lpips"] = eval_info_3_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.33"]["ssim"] = eval_info_3_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.33"]["psnr"] = eval_info_3_psuedo_enhances['psnr'].data.item()
        
        candidates_3_saved_images = torch.cat([candidates_3_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_3_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_3_psuedo_images[0,0,:,:,:]),
                                               candidates_3_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_3_saved_images = candidates_3_saved_images.permute(1,2,0).cpu().numpy()
        candidates_3_saved_images = (candidates_3_saved_images*255).astype(np.uint8)
        saved_images_dict["0.33"] = candidates_3_saved_images
        
    
        # 0.5 ----> Center
        # raw
        center_rgb_eval_info_raw = metrics_mean(pred=rendered_images_center_stereo,
                                           gt=gt_images_center_stereo)
        # enhance 
        raw_center_left = rendered_images_center_stereo[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_center_left = (raw_center_left*255).astype(np.uint8)
        raw_center_left_pil = Image.fromarray(raw_center_left)

        raw_center_right = rendered_images_center_stereo[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_center_right = (raw_center_right*255).astype(np.uint8)
        raw_center_right_pil = Image.fromarray(raw_center_right)


        with torch.no_grad():
            enhanced_center_left_pil = diffix3d_network(
                cfg.prompt,
                image=raw_center_left_pil, 
                ref_image=left_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]

            
            enhanced_center_right_pil = diffix3d_network(
                cfg.prompt,
                image=raw_center_right_pil, 
                ref_image=right_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            

        enhanced_center_left_tensor = convert_pil_to_tensor(enhanced_center_left_pil)
        enhanced_center_right_tensor = convert_pil_to_tensor(enhanced_center_right_pil)
        
        enhanced_center_tensor = torch.cat([enhanced_center_left_tensor, 
                                            enhanced_center_right_tensor], 
                                                       dim=1)
        
        enhanced_center_tensor = enhanced_center_tensor.type_as(gt_images_first_stereo)
        
        center_rgb_eval_info_enhances = metrics_mean(pred=enhanced_center_tensor,
                                           gt=gt_images_center_stereo)
        
        
        raw_results_stat["0.5"]["lpips"] = center_rgb_eval_info_raw['lpips'].data.item()
        raw_results_stat["0.5"]["ssim"] = center_rgb_eval_info_raw['ssim'].data.item()
        raw_results_stat["0.5"]["psnr"] = center_rgb_eval_info_raw['psnr'].data.item()
        enhanced_results_stat["0.5"]["lpips"] = center_rgb_eval_info_enhances['lpips'].data.item()
        enhanced_results_stat["0.5"]["ssim"] = center_rgb_eval_info_enhances['ssim'].data.item()
        enhanced_results_stat["0.5"]["psnr"] = center_rgb_eval_info_enhances['psnr'].data.item()
        
        candidates_center_saved_images = torch.cat([rendered_images_center_stereo[0,0,:,:,:],
                                               enhanced_center_left_tensor[0,0,:,:,:].type_as(rendered_images_center_stereo[0,0,:,:,:]),
                                               gt_images_center_stereo[0,0,:,:,:]],
                                              dim=1)
        candidates_center_saved_images = candidates_center_saved_images.permute(1,2,0).cpu().numpy()
        candidates_center_saved_images = (candidates_center_saved_images*255).astype(np.uint8)
        saved_images_dict["0.5"] = candidates_center_saved_images

        
        # 0.66 
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.66)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_4_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_4_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_4_psuedo_raw = metrics_mean(pred=candidates_4_psuedo_images,
                                           gt=candidates_4_psuedo_gt)
        
        # enhanced
        raw_candiates_4_psuedo_left = candidates_4_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_4_psuedo_left= (raw_candiates_4_psuedo_left*255).astype(np.uint8)
        raw_candiates_4_psuedo_left_pil = Image.fromarray(raw_candiates_4_psuedo_left)
        
        raw_candiates_4_psuedo_right = candidates_4_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_4_psuedo_right= (raw_candiates_4_psuedo_right*255).astype(np.uint8)
        raw_candiates_4_psuedo_right_pil = Image.fromarray(raw_candiates_4_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_4_psuedo_left_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_4_psuedo_left_pil, 
                ref_image=left_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
            enhanced_candiates_4_psuedo_right_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_4_psuedo_right_pil, 
                ref_image=right_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
        
        enhanced_candiates_4_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_4_psuedo_left_pil)
        enhanced_candiates_4_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_4_psuedo_right_pil)
        enhanced_candiates_4_psuedo_stereo_tensor = torch.cat([enhanced_candiates_4_psuedo_left_tensor, 
                                                                enhanced_candiates_4_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_4_psuedo_stereo_tensor = enhanced_candiates_4_psuedo_stereo_tensor.type_as(candidates_2_psuedo_gt)
        eval_info_4_psuedo_enhances = metrics_mean(
                                            pred=enhanced_candiates_4_psuedo_stereo_tensor,
                                           gt=candidates_4_psuedo_gt)
        
        raw_results_stat["0.66"]["lpips"] = eval_info_4_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.66"]["ssim"] = eval_info_4_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.66"]["psnr"] = eval_info_4_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.66"]["lpips"] = eval_info_4_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.66"]["ssim"] = eval_info_4_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.66"]["psnr"] = eval_info_4_psuedo_enhances['psnr'].data.item()
        
        
        candidates_4_saved_images = torch.cat([candidates_4_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_4_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_4_psuedo_images[0,0,:,:,:]),
                                               candidates_4_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_4_saved_images = candidates_4_saved_images.permute(1,2,0).cpu().numpy()
        candidates_4_saved_images = (candidates_4_saved_images*255).astype(np.uint8)
        saved_images_dict["0.66"] = candidates_4_saved_images
                
        # 0.75  
        nums_of_candidates_views = candidates_rendered_images_all_stereo.shape[1]
        nums_of_candidates_stereo_pairs = nums_of_candidates_views // 2
        stereo_index = int(nums_of_candidates_stereo_pairs * 0.75)
        left_index = stereo_index * 2
        right_index = stereo_index * 2 + 1
        
        # raw
        candidates_5_psuedo_images = candidates_rendered_images_all_stereo[:,left_index:right_index+1,:,:,:]
        candidates_5_psuedo_gt = candidates_gt_images_all_stereo[:,left_index:right_index+1,:,:,:]
        
        eval_info_5_psuedo_raw = metrics_mean(pred=candidates_5_psuedo_images,
                                           gt=candidates_5_psuedo_gt)
        
        # enhanced
        raw_candiates_5_psuedo_left = candidates_5_psuedo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_5_psuedo_left= (raw_candiates_5_psuedo_left*255).astype(np.uint8)
        raw_candiates_5_psuedo_left_pil = Image.fromarray(raw_candiates_5_psuedo_left)
        
        raw_candiates_5_psuedo_right = candidates_5_psuedo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_candiates_5_psuedo_right= (raw_candiates_5_psuedo_right*255).astype(np.uint8)
        raw_candiates_5_psuedo_right_pil = Image.fromarray(raw_candiates_5_psuedo_right)
        
        
        with torch.no_grad():
            enhanced_candiates_5_psuedo_left_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_5_psuedo_left_pil, 
                ref_image=left_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
            enhanced_candiates_5_psuedo_right_pil = diffix3d_network(
                cfg.prompt,
                image=raw_candiates_5_psuedo_right_pil, 
                ref_image=right_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
        
        enhanced_candiates_5_psuedo_left_tensor = convert_pil_to_tensor(enhanced_candiates_5_psuedo_left_pil)
        enhanced_candiates_5_psuedo_right_tensor = convert_pil_to_tensor(enhanced_candiates_5_psuedo_right_pil)
        enhanced_candiates_5_psuedo_stereo_tensor = torch.cat([enhanced_candiates_5_psuedo_left_tensor, 
                                                                enhanced_candiates_5_psuedo_right_tensor], 
                                                               dim=1)
        enhanced_candiates_5_psuedo_stereo_tensor = enhanced_candiates_5_psuedo_stereo_tensor.type_as(candidates_2_psuedo_gt)
        eval_info_5_psuedo_enhances = metrics_mean(
                                            pred=enhanced_candiates_5_psuedo_stereo_tensor,
                                           gt=candidates_5_psuedo_gt)
    
        raw_results_stat["0.75"]["lpips"] = eval_info_5_psuedo_raw['lpips'].data.item()
        raw_results_stat["0.75"]["ssim"] = eval_info_5_psuedo_raw['ssim'].data.item()
        raw_results_stat["0.75"]["psnr"] = eval_info_5_psuedo_raw['psnr'].data.item()
        enhanced_results_stat["0.75"]["lpips"] = eval_info_5_psuedo_enhances['lpips'].data.item()
        enhanced_results_stat["0.75"]["ssim"] = eval_info_5_psuedo_enhances['ssim'].data.item()
        enhanced_results_stat["0.75"]["psnr"] = eval_info_5_psuedo_enhances['psnr'].data.item()
        
        
        candidates_5_saved_images = torch.cat([candidates_5_psuedo_images[0,0,:,:,:],
                                               enhanced_candiates_5_psuedo_left_tensor[0,0,:,:,:].type_as(candidates_5_psuedo_images[0,0,:,:,:]),
                                               candidates_5_psuedo_gt[0,0,:,:,:]],
                                              dim=1)
        candidates_5_saved_images = candidates_5_saved_images.permute(1,2,0).cpu().numpy()
        candidates_5_saved_images = (candidates_5_saved_images*255).astype(np.uint8)
        saved_images_dict["0.75"] = candidates_5_saved_images
        
        
    
        # 1.0 -----> Last
       # raw
        last_rgb_eval_info_raw = metrics_mean(pred=rendered_images_last_stereo,
                                           gt=gt_images_last_stereo)
        # enhance 
        raw_last_left = rendered_images_last_stereo[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        raw_last_left = (raw_last_left*255).astype(np.uint8)
        raw_last_left_pil = Image.fromarray(raw_last_left)

        raw_last_right = rendered_images_last_stereo[0,1,:,:,:].permute(1,2,0).cpu().numpy()
        raw_last_right = (raw_last_right*255).astype(np.uint8)
        raw_last_right_pil = Image.fromarray(raw_last_right)


        with torch.no_grad():
            enhanced_last_left_pil = diffix3d_network(
                cfg.prompt,
                image=raw_last_left_pil, 
                ref_image=left_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]
            
            enhanced_last_right_pil = diffix3d_network(
                cfg.prompt,
                image=raw_last_right_pil, 
                ref_image=right_ref_pil, 
                num_inference_steps=1, 
                timesteps=[199], guidance_scale=0.0).images[0]


        enhanced_last_left_tensor = convert_pil_to_tensor(enhanced_last_left_pil)
        enhanced_last_right_tensor = convert_pil_to_tensor(enhanced_last_right_pil)
        
        enhanced_last_tensor = torch.cat([enhanced_last_left_tensor, 
                                            enhanced_last_right_tensor], 
                                                       dim=1)
        
        enhanced_last_tensor = enhanced_last_tensor.type_as(gt_images_first_stereo)
        
        last_rgb_eval_info_enhances = metrics_mean(pred=enhanced_last_tensor,
                                           gt=gt_images_last_stereo)
        
        
        raw_results_stat["1.0"]["lpips"] = last_rgb_eval_info_raw['lpips'].data.item()
        raw_results_stat["1.0"]["ssim"] = last_rgb_eval_info_raw['ssim'].data.item()
        raw_results_stat["1.0"]["psnr"] = last_rgb_eval_info_raw['psnr'].data.item()
        enhanced_results_stat["1.0"]["lpips"] = last_rgb_eval_info_enhances['lpips'].data.item()
        enhanced_results_stat["1.0"]["ssim"] = last_rgb_eval_info_enhances['ssim'].data.item()
        enhanced_results_stat["1.0"]["psnr"] = last_rgb_eval_info_enhances['psnr'].data.item()
        
        
        candidates_last_saved_images = torch.cat([rendered_images_last_stereo[0,0,:,:,:],
                                               enhanced_last_left_tensor[0,0,:,:,:].type_as(rendered_images_last_stereo[0,0,:,:,:]),
                                               gt_images_last_stereo[0,0,:,:,:]],
                                              dim=1)
        candidates_last_saved_images = candidates_last_saved_images.permute(1,2,0).cpu().numpy()
        candidates_last_saved_images = (candidates_last_saved_images*255).astype(np.uint8)
        saved_images_dict["1.0"] = candidates_last_saved_images
        
        
        return raw_results_stat, enhanced_results_stat, saved_images_dict
    
    # FIXME： Please Delete in the future, this version is just to select the
    # best selection from finetuned diffix3d and the stereosplat 
    def stereosplat_plus_gt_pose_once_progressive_inference_with_difix3d(self,
                                        batch,
                                        val_result_savedir,
                                        bin_token_list,
                                        start_images_views = 2,
                                        pseudo_ratio_index = [],
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
            input_batch_dict,output_batch_dict =self.prepare_input_multiview(batch=batch,
                                                                        view_num=view_num,
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
        
        #  rendere all the images/ intrinsics and extrinsics
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
        
        output_rgb = output_batch_dict['output_imgs'] # This is the GT Images
        sparse_depth_gt = output_batch_dict['output_sparse_depth']  


        '''Do the visualization and the evaluation here'''
        rendered_images_fusion = interleave_left_right(rendered_color_fuse)
        rendered_depth_fusion = interleave_left_right_depth(rendered_depth_fuse)
        sparse_depth_gt = interleave_left_right_depth(sparse_depth_gt)
        rendered_images_gt = interleave_left_right(output_rgb)



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
        
        
        
        # get the gt input vies
        input_batch_dict,output_batch_dict =self.prepare_tripleview_by_ratio_index(
                                                batch=batch,
                                                pseudo_ratio_index=pseudo_ratio_index)
        
    
        if use_diffix3d:
            
            # reference image here
            left_image_ref = input_batch_dict["imgs"][0,0,:,:,:]
            right_image_ref = input_batch_dict["imgs"][0,1,:,:,:]
            left_image_ref = left_image_ref.permute(1,2,0).cpu().numpy()
            right_image_ref = right_image_ref.permute(1,2,0).cpu().numpy()
            left_image_ref = (left_image_ref*255).astype(np.uint8)
            right_image_ref = (right_image_ref*255).astype(np.uint8)
            left_image_ref_pil = Image.fromarray(left_image_ref)
            right_image_ref_pil = Image.fromarray(right_image_ref)
            
            
            # default using center and the last frame as the input.
            if pseudo_ratio_index[0]==0.5 and pseudo_ratio_index[1]==1.0:
                # implemented with the finetuned diffix3d model here
                raw_center_left = rendered_images_center_stereo[0,0,:,:,:].permute(1,2,0).cpu().numpy()
                raw_center_left = (raw_center_left*255).astype(np.uint8)
                raw_center_left_pil = Image.fromarray(raw_center_left)
                
                raw_center_right = rendered_images_center_stereo[0,1,:,:,:].permute(1,2,0).cpu().numpy()
                raw_center_right = (raw_center_right*255).astype(np.uint8)
                raw_center_right_pil = Image.fromarray(raw_center_right)
                
                raw_last_left = rendered_images_last_stereo[0,0,:,:,:].permute(1,2,0).cpu().numpy()
                raw_last_left = (raw_last_left*255).astype(np.uint8)
                raw_last_left_pil = Image.fromarray(raw_last_left)
                
                raw_last_right = rendered_images_last_stereo[0,1,:,:,:].permute(1,2,0).cpu().numpy()
                raw_last_right = (raw_last_right*255).astype(np.uint8)
                raw_last_right_pil = Image.fromarray(raw_last_right)
                
                
                with torch.no_grad():
                    
                    enhanced_center_left_pil = diffix3d_network.sample(
                        raw_center_left_pil,
                        height=112,
                        width=544,
                        ref_image=left_image_ref_pil,
                        prompt=cfg.prompt
                    )
                    enhanced_center_right_pil = diffix3d_network.sample(
                        raw_center_right_pil,
                        height=112,
                        width=544,
                        ref_image=right_image_ref_pil,
                        prompt=cfg.prompt
                    )
                    enhanced_last_left_pil = diffix3d_network.sample(
                        raw_last_left_pil,
                        height=112,
                        width=544,
                        ref_image=left_image_ref_pil,
                        prompt=cfg.prompt
                    )
                    enhanced_last_right_pil = diffix3d_network.sample(
                        raw_last_right_pil,
                        height=112,
                        width=544,
                        ref_image=right_image_ref_pil,
                        prompt=cfg.prompt
                    )
                    
                    
                    enhance_center_left_tensor = convert_pil_to_tensor(enhanced_center_left_pil)
                    enhance_center_right_tensor = convert_pil_to_tensor(enhanced_center_right_pil)
                    enhance_last_left_tensor = convert_pil_to_tensor(enhanced_last_left_pil)
                    enhance_last_right_tensor = convert_pil_to_tensor(enhanced_last_right_pil)
                    enhanced_center_and_last_stereo_frames = torch.cat([enhance_center_left_tensor, 
                                                                        enhance_center_right_tensor, 
                                                                        enhance_last_left_tensor, 
                                                                        enhance_last_right_tensor], dim=1)
                    
                    input_batch_dict["imgs"][:,2:,:,:,:] = enhanced_center_and_last_stereo_frames

            
            else:
                # imnplement with the finetuned difix3d model here.
                raw_rendered_rest_images = rendered_images_fusion[:,:-6,:,:,:]
                raw_rendered_rest_images = torch.cat([rendered_images_first_stereo, 
                                                      raw_rendered_rest_images,
                                                      rendered_images_last_stereo], dim=1)
                
                raw_rendered_rest_stereo_pair_nums = raw_rendered_rest_images.shape[1] // 2
                
                raw_second_frame_left_id = int(raw_rendered_rest_stereo_pair_nums * pseudo_ratio_index[0]) * 2
                raw_second_frame_right_id = raw_second_frame_left_id + 1
                
                raw_third_frame_left_id = int(raw_rendered_rest_stereo_pair_nums * pseudo_ratio_index[1]) * 2
                raw_third_frame_right_id = raw_third_frame_left_id + 1
                
                
                # replaced with the new images
                raw_second_frames_stereo_images = raw_rendered_rest_images[:,raw_second_frame_left_id:raw_second_frame_right_id+1,:,:,:]
                raw_third_frames_stereo_images = raw_rendered_rest_images[:,raw_third_frame_left_id:raw_third_frame_right_id+1,:,:,:]               
                
                
                
                raw_second_frame_left = raw_second_frames_stereo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
                raw_second_frame_left = (raw_second_frame_left*255).astype(np.uint8)
                raw_second_frame_left_pil = Image.fromarray(raw_second_frame_left)
                
                raw_second_frame_right = raw_second_frames_stereo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
                raw_second_frame_right = (raw_second_frame_right*255).astype(np.uint8)
                raw_second_frame_right_pil = Image.fromarray(raw_second_frame_right)
                
                raw_third_frame_left = raw_third_frames_stereo_images[0,0,:,:,:].permute(1,2,0).cpu().numpy()
                raw_third_frame_left = (raw_third_frame_left*255).astype(np.uint8)
                raw_third_frame_left_pil = Image.fromarray(raw_third_frame_left)
                
                raw_third_frame_right = raw_third_frames_stereo_images[0,1,:,:,:].permute(1,2,0).cpu().numpy()
                raw_third_frame_right = (raw_third_frame_right*255).astype(np.uint8)
                raw_third_frame_right_pil = Image.fromarray(raw_third_frame_right)
                
                
                with torch.no_grad():
                    enhanced_second_left_pil = diffix3d_network.sample(
                        raw_second_frame_left_pil,
                        height=112,
                        width=544,
                        ref_image=left_image_ref_pil,
                        prompt=cfg.prompt
                    )
                    enhanced_second_right_pil = diffix3d_network.sample(
                        raw_second_frame_right_pil,
                        height=112,
                        width=544,
                        ref_image=right_image_ref_pil,
                        prompt=cfg.prompt
                    )
                    enhanced_third_left_pil = diffix3d_network.sample(
                        raw_third_frame_left_pil,
                        height=112,
                        width=544,
                        ref_image=left_image_ref_pil,
                        prompt=cfg.prompt
                    )
                    enhanced_third_right_pil = diffix3d_network.sample(
                        raw_third_frame_right_pil,
                        height=112,
                        width=544,
                        ref_image=right_image_ref_pil,
                        prompt=cfg.prompt
                    )
                    
                    enhance_second_left_tensor = convert_pil_to_tensor(enhanced_second_left_pil)
                    enhance_second_right_tensor = convert_pil_to_tensor(enhanced_second_right_pil)
                    enhance_third_left_tensor = convert_pil_to_tensor(enhanced_third_left_pil)
                    enhance_third_right_tensor = convert_pil_to_tensor(enhanced_third_right_pil)
                    enhanced_second_and_third_stereo_frames = torch.cat([enhance_second_left_tensor, 
                                                                        enhance_second_right_tensor, 
                                                                        enhance_third_left_tensor, 
                                                                        enhance_third_right_tensor], dim=1)
        
                input_batch_dict["imgs"][:,2:,:,:,:] = enhanced_second_and_third_stereo_frames

        
        else:
            if pseudo_ratio_index[0]==0.5 and pseudo_ratio_index[1]==1.0:
                
                raw_center_and_last_stereo_frames = torch.cat([rendered_images_center_stereo, 
                                                               rendered_images_last_stereo], dim=1)
                input_batch_dict["imgs"][:,2:,:,:,:] = raw_center_and_last_stereo_frames
            
            else:
                
                raw_rendered_rest_images = rendered_images_fusion[:,:-6,:,:,:]
                raw_rendered_rest_images = torch.cat([rendered_images_first_stereo, 
                                                      raw_rendered_rest_images,
                                                      rendered_images_last_stereo], dim=1)
                
                raw_rendered_rest_stereo_pair_nums = raw_rendered_rest_images.shape[1] // 2
                
                raw_second_frame_left_id = int(raw_rendered_rest_stereo_pair_nums * pseudo_ratio_index[0]) * 2
                raw_second_frame_right_id = raw_second_frame_left_id + 1
                
                raw_third_frame_left_id = int(raw_rendered_rest_stereo_pair_nums * pseudo_ratio_index[1]) * 2
                raw_third_frame_right_id = raw_third_frame_left_id + 1
                
                
                # replaced with the new images
                raw_second_frames_stereo_images = raw_rendered_rest_images[:,raw_second_frame_left_id:raw_second_frame_right_id+1,:,:,:]
                raw_third_frames_stereo_images = raw_rendered_rest_images[:,raw_third_frame_left_id:raw_third_frame_right_id+1,:,:,:]               
                

                
                input_batch_dict["imgs"][:,2:4,:,:,:] = raw_second_frames_stereo_images
                input_batch_dict["imgs"][:,4:,:,:,:] = raw_third_frames_stereo_images
                
                
                

        # second time inference here for progressive infernece.
        img = input_batch_dict["imgs"]
        height, width = img.shape[-2:]
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img)
        gaussians_cv, gaussians_feat, pred_depths = self.costvolume_gs(input_batch_dict, cfg=cfg,
                                                                      images_feat=img_feats[0])

        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_cv_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_cv[b, :, 0] >= x_start) & (gaussians_cv[b, :, 0] <= x_end) & \
                           (gaussians_cv[b, :, 1] >= y_start) & (gaussians_cv[b, :, 1] <= y_end) & \
                           (gaussians_cv[b, :, 2] >= z_start) & (gaussians_cv[b, :, 2] <= z_end)
            gaussians_cv_mask_i = gaussians_cv[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_cv_mask.append(gaussians_cv_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)

        gaussians_volume = self.volume_gs(
            [img_feats[0]],
            input_batch_dict['extrinsics'],
            gaussians_cv_mask,
            gaussians_feat_mask,
            input_batch_dict["img_metas"])

        gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
        gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
        bs = gaussians_all.shape[0]

        render_c2w = output_batch_dict["output_c2ws"]
        intrinsics = input_batch_dict['intrinsics'].clone()
        output_intrinsics = intrinsics[:, 0:1, :, :].repeat(1, render_c2w.shape[1], 1, 1)
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
        rendered_color_fuse = rendered_results_fuse['image']
        rendered_depth_fuse = rendered_results_fuse['depth']
        rendered_alpha_fuse = rendered_results_fuse['alpha']
        rendered_depth_fuse = rendered_depth_fuse.squeeze(2)
        rendered_alpha_fuse = rendered_alpha_fuse.squeeze(2)
        rendered_color_fuse = torch.clamp(rendered_color_fuse, min=0, max=1.0)
        rendered_depth_fuse = torch.clamp(rendered_depth_fuse, min=0, max=150)

        output_rgb = output_batch_dict['output_imgs']
        sparse_depth_gt = output_batch_dict['output_sparse_depth']

        rendered_images_fusion = interleave_left_right(rendered_color_fuse)
        rendered_depth_fusion = interleave_left_right_depth(rendered_depth_fuse)
        sparse_depth_gt = interleave_left_right_depth(sparse_depth_gt)
        rendered_images_gt = interleave_left_right(output_rgb)
        
        

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

    # FIXME: Please Delete in the Future, this verison is just for baseline preserving fusion debugging purpose.
    def stereosplat_plus_baseline_preserving_fusion_oracle(self,
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
        

        ''' Novel View Generation  for the 3DG (Base) here'''
        rendered_images_base = interleave_left_right(rendered_color_fuse)
        rendered_depth_base = interleave_left_right_depth(rendered_depth_fuse)
        rendered_alpha_base = interleave_left_right_depth(rendered_alpha_fuse)
        
        sparse_depth_gt = interleave_left_right_depth(sparse_depth_gt)
        rendered_images_gt = interleave_left_right(output_rgb)
        
        rendered_images_based_center_last = rendered_images_base[:,-6:-2,:,:,:]
        rendered_images_based_center = rendered_images_based_center_last[:,-4:-2,:,:,:]
        rendered_images_based_last = rendered_images_based_center_last[:,-2:,:,:,:]
        
        

        
        
    
        
        if use_diffix3d:
            # logic here
            # enhance the center frame
            rendered_images_based_center_last = rendered_images_based_center_last
            rendered_center_left = rendered_images_based_center_last[0,0,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_center_right = rendered_images_based_center_last[0,1,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_last_left = rendered_images_based_center_last[0,2,:,:,:].permute(1,2,0).cpu().numpy()
            rendered_last_right = rendered_images_based_center_last[0,3,:,:,:].permute(1,2,0).cpu().numpy()
            
            
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
            
            enhanced_rendered_center_left = torch.from_numpy(enhanced_rendered_center_left).to(rendered_images_based_center_last.device)
            enhanced_rendered_center_right = torch.from_numpy(enhanced_rendered_center_right).to(rendered_images_based_center_last.device)
            enhanced_rendered_last_left = torch.from_numpy(enhanced_rendered_last_left).to(rendered_images_based_center_last.device)
            enhanced_rendered_last_right = torch.from_numpy(enhanced_rendered_last_right).to(rendered_images_based_center_last.device)
            enhanced_rendered_center_left = enhanced_rendered_center_left.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_center_right = enhanced_rendered_center_right.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_last_left = enhanced_rendered_last_left.permute(2,0,1).unsqueeze(0)
            enhanced_rendered_last_right = enhanced_rendered_last_right.permute(2,0,1).unsqueeze(0)
            
            enhanced_rendered_images_based_center_last = torch.cat([enhanced_rendered_center_left,
                                                        enhanced_rendered_center_right,
                                                        enhanced_rendered_last_left,
                                                        enhanced_rendered_last_right],dim=0).unsqueeze(0)
            
            rendered_images_based_center_last = enhanced_rendered_images_based_center_last
        
        
        input_batch_dict,output_batch_dict = self.prepare_input_multiview(batch=batch,view_num=6,
                                                                         matching_nums=4)
        input_batch_dict["imgs"][:,2:,:,:,:] = rendered_images_based_center_last
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
        rendered_images_plus = interleave_left_right(rendered_color_fuse)
        rendered_depth_plus = interleave_left_right_depth(rendered_depth_fuse)
        rendered_alpha_plus = interleave_left_right_depth(rendered_alpha_fuse)
        
        
        
        all_rgb_eval_info_base = metrics_mean(pred = rendered_images_base,
                                           gt = rendered_images_gt)
        all_rgb_eval_info_base_depth = depth_metrics_absrel_sqrel_rmse_log(
                                                        pred = rendered_depth_base,
                                                         gt = sparse_depth_gt)
        

        all_rgb_eval_info_plus = metrics_mean(pred = rendered_images_plus,
                                           gt = rendered_images_gt)
        all_rgb_eval_info_plus_depth = depth_metrics_absrel_sqrel_rmse_log(
                                                        pred = rendered_depth_plus,
                                                         gt = sparse_depth_gt)

        # oracle G_base and G_plus fusion version
        base_mask,plus_mask, fusion_image = fuse_rgb_by_gt_error(
            rgb1 = rendered_images_base,
            rgb2 = rendered_images_plus,
            gt = rendered_images_gt
        )
        
        
        fusion_depth = rendered_depth_base * base_mask.squeeze(2) + rendered_depth_plus * plus_mask.squeeze(2)
    
        all_rgb_eval_info_fusion = metrics_mean(pred = fusion_image,
                                           gt = rendered_images_gt)
        all_depth_eval_info_fusion = depth_metrics_absrel_sqrel_rmse_log(
                                                        pred = fusion_depth,
                                                         gt = sparse_depth_gt)
        
        

        eval_base_dict = {
            "psnr": all_rgb_eval_info_base['psnr'].data.item(),
            "ssim": all_rgb_eval_info_base['ssim'].data.item(),
            "lpips": all_rgb_eval_info_base['lpips'].data.item(),
            "absrel": all_rgb_eval_info_base_depth['AbsRel'].data.item(),
            "sqrel": all_rgb_eval_info_base_depth['SqRel'].data.item(),
            "rmse_log": all_rgb_eval_info_base_depth['RMSE_log'].data.item(),
        }
        eval_plus_dict = {
            "psnr": all_rgb_eval_info_plus['psnr'].data.item(),
            "ssim": all_rgb_eval_info_plus['ssim'].data.item(),
            "lpips": all_rgb_eval_info_plus['lpips'].data.item(),
            "absrel": all_rgb_eval_info_plus_depth['AbsRel'].data.item(),
            "sqrel": all_rgb_eval_info_plus_depth['SqRel'].data.item(),
            "rmse_log": all_rgb_eval_info_plus_depth['RMSE_log'].data.item(),
        }
        
        eval_fusion_dict = {
            "psnr": all_rgb_eval_info_fusion['psnr'].data.item(),
            "ssim": all_rgb_eval_info_fusion['ssim'].data.item(),
            "lpips": all_rgb_eval_info_fusion['lpips'].data.item(),
            "absrel": all_depth_eval_info_fusion['AbsRel'].data.item(),
            "sqrel": all_depth_eval_info_fusion['SqRel'].data.item(),
            "rmse_log": all_depth_eval_info_fusion['RMSE_log'].data.item(),
        }
        
        final_eval_dict ={
            "G_base": eval_base_dict,
            "G_plus": eval_plus_dict,
            "G_fusion": eval_fusion_dict,
        }
        
        
        
        # doing the visualiation here
         # - Visualize 
         # (1)Images:  base images/ plus images / fusion images/ gt images
         # (2) fusion_images
         # (3) GT_Base_Mask
         # (4) GT_Plus_Mask
         # (5) Raw Base Alpha Map
         # (6) Raw Plus Alpha Map
         # (7) Comparasion Base Alpha Map (0,1)
         # (8) Comparsion Plus Alpha Map (0,1)
         
         
        if vis:
            saved_folder_for_visualization = os.path.join(val_result_savedir,bin_token_name)
            os.makedirs(saved_folder_for_visualization,exist_ok=True)
            
            rendered_images_folder = os.path.join(saved_folder_for_visualization,"Images")
            rendered_depths_folder = os.path.join(saved_folder_for_visualization,"Rendered_Depths")
            
            GT_Base_Mask_folder = os.path.join(saved_folder_for_visualization,"GT_Base_Mask")
            GT_Plus_Mask_folder = os.path.join(saved_folder_for_visualization,"GT_Plus_Mask")
            Raw_Base_Alpha_Map_folder = os.path.join(saved_folder_for_visualization,"Raw_Base_Alpha_Map")
            Raw_Plus_Alpha_Map_folder = os.path.join(saved_folder_for_visualization,"Raw_Plus_Alpha_Map")
            Comparasion_Base_Alpha_Map_folder = os.path.join(saved_folder_for_visualization,"Comparasion_Base_Alpha_Map")
            Comparasion_Plus_Alpha_Map_folder = os.path.join(saved_folder_for_visualization,"Comparasion_Plus_Alpha_Map")
            
            os.makedirs(rendered_images_folder,exist_ok=True)
            os.makedirs(rendered_depths_folder,exist_ok=True)
            os.makedirs(GT_Base_Mask_folder,exist_ok=True)
            os.makedirs(GT_Plus_Mask_folder,exist_ok=True)
            os.makedirs(Raw_Base_Alpha_Map_folder,exist_ok=True)
            os.makedirs(Raw_Plus_Alpha_Map_folder,exist_ok=True)
            os.makedirs(Comparasion_Base_Alpha_Map_folder,exist_ok=True)
            os.makedirs(Comparasion_Plus_Alpha_Map_folder,exist_ok=True)
            
            
            # saved images here
            ##########################################################################################
            rendered_base_plus_fusion_gt_tensor = torch.cat([rendered_images_base,
                                                             rendered_images_plus,
                                                             fusion_image,
                                                             rendered_images_gt],
                                                            dim=-2)
            
            rendered_base_plus_fusion_gt_tensor_first_stereo = rendered_base_plus_fusion_gt_tensor[:,-2:,:,:,:]
            rendered_base_plus_fusion_gt_tensor_last_stereo = rendered_base_plus_fusion_gt_tensor[:,-4:-2,:,:,:]
            rendered_base_plus_fusion_gt_tensor_center_stereo = rendered_base_plus_fusion_gt_tensor[:,-6:-4,:,:,:]
            
            rendered_base_plus_fusion_gt_tensor_first_stereo = torch.cat([rendered_base_plus_fusion_gt_tensor_first_stereo[0][0],
                                                                          rendered_base_plus_fusion_gt_tensor_first_stereo[0][1]],
                                                                         dim=-1).permute(1,2,0).cpu().numpy()
            
            rendered_base_plus_fusion_gt_tensor_first_stereo_vis = (rendered_base_plus_fusion_gt_tensor_first_stereo*255.0).astype(np.uint8)
            skimage.io.imsave(os.path.join(rendered_images_folder,"first_stereo.png"),rendered_base_plus_fusion_gt_tensor_first_stereo_vis)
            
            rendered_base_plus_fusion_gt_tensor_last_stereo = torch.cat([rendered_base_plus_fusion_gt_tensor_last_stereo[0][0],
                                                                         rendered_base_plus_fusion_gt_tensor_last_stereo[0][1]],
                                                                         dim=-1).permute(1,2,0).cpu().numpy()
            rendered_base_plus_fusion_gt_tensor_last_stereo_vis = (rendered_base_plus_fusion_gt_tensor_last_stereo*255.0).astype(np.uint8)
            skimage.io.imsave(os.path.join(rendered_images_folder,"last_stereo.png"),rendered_base_plus_fusion_gt_tensor_last_stereo_vis)
            
            rendered_base_plus_fusion_gt_tensor_center_stereo = torch.cat([rendered_base_plus_fusion_gt_tensor_center_stereo[0][0],
                                                                          rendered_base_plus_fusion_gt_tensor_center_stereo[0][1]],
                                                                         dim=-1).permute(1,2,0).cpu().numpy()
            rendered_base_plus_fusion_gt_tensor_center_stereo_vis = (rendered_base_plus_fusion_gt_tensor_center_stereo*255.0).astype(np.uint8)
            skimage.io.imsave(os.path.join(rendered_images_folder,"center_stereo.png"),rendered_base_plus_fusion_gt_tensor_center_stereo_vis)
            
            rendered_images_readme_content_txt = "from top to down: base images/plus images/fusion images/gt images"
            saved_readme_txt_path = os.path.join(rendered_images_folder,"readme.txt")
            
            if not os.path.exists(saved_readme_txt_path):
                with open(saved_readme_txt_path,"w") as f:
                    f.write(rendered_images_readme_content_txt)
                    
            ##########################################################################################        
            # save the depths here  
            rendered_depth_base_plus_fusion_gt_tensor = torch.cat([rendered_depth_base,
                                                                   rendered_depth_plus,
                                                                   fusion_depth,
                                                                   sparse_depth_gt],
                                                                  dim=-2)
            
            rendered_depth_base_plus_fusion_gt_tensor_first_stereo = rendered_depth_base_plus_fusion_gt_tensor[:,-2:,:,:]
            rendered_depth_base_plus_fusion_gt_tensor_last_stereo = rendered_depth_base_plus_fusion_gt_tensor[:,-4:-2,:,:]
            rendered_depth_base_plus_fusion_gt_tensor_center_stereo = rendered_depth_base_plus_fusion_gt_tensor[:,-6:-4,:,:]
            
            
            rendered_depth_base_plus_fusion_gt_tensor_first_stereo_vis = torch.cat([rendered_depth_base_plus_fusion_gt_tensor_first_stereo[0][0],
                                                                                    rendered_depth_base_plus_fusion_gt_tensor_first_stereo[0][1]],
                                                                                   dim=-1).cpu().numpy()
            
            
            rendered_depth_base_plus_fusion_gt_tensor_last_stereo_vis = torch.cat([rendered_depth_base_plus_fusion_gt_tensor_last_stereo[0][0],
                                                                                    rendered_depth_base_plus_fusion_gt_tensor_last_stereo[0][1]],
                                                                                   dim=-1).cpu().numpy()
            
            rendered_depth_base_plus_fusion_gt_tensor_center_stereo_vis = torch.cat([rendered_depth_base_plus_fusion_gt_tensor_center_stereo[0][0],
                                                                                    rendered_depth_base_plus_fusion_gt_tensor_center_stereo[0][1]],
                                                                                   dim=-1).cpu().numpy()
            
            
            rendered_depth_base_plus_fusion_gt_tensor_first_stereo_vis = convert_depth_to_disp(depth=rendered_depth_base_plus_fusion_gt_tensor_first_stereo_vis)
            rendered_depth_base_plus_fusion_gt_tensor_last_stereo_vis = convert_depth_to_disp(depth=rendered_depth_base_plus_fusion_gt_tensor_last_stereo_vis)
            rendered_depth_base_plus_fusion_gt_tensor_center_stereo_vis = convert_depth_to_disp(depth=rendered_depth_base_plus_fusion_gt_tensor_center_stereo_vis)
            
            
            skimage.io.imsave(os.path.join(rendered_depths_folder,"first_stereo.png"),rendered_depth_base_plus_fusion_gt_tensor_first_stereo_vis)
            skimage.io.imsave(os.path.join(rendered_depths_folder,"last_stereo.png"),rendered_depth_base_plus_fusion_gt_tensor_last_stereo_vis)
            skimage.io.imsave(os.path.join(rendered_depths_folder,"center_stereo.png"),rendered_depth_base_plus_fusion_gt_tensor_center_stereo_vis)
            
            rendered_depths_readme_content_txt = "from top to down: base depths/ plus depths/ fusion depths/ gt depths"
            saved_readme_txt_path = os.path.join(rendered_depths_folder,"readme.txt")
            if not os.path.exists(saved_readme_txt_path):
                with open(saved_readme_txt_path,"w") as f:
                    f.write(rendered_depths_readme_content_txt)
            ########################################################################################## 
            
            ############################### Save the GT Mask here ########################################################
            gt_mask_base_first_frame_stereo = base_mask[0,-2:,0,:,:]
            gt_mask_base_last_frame_stereo = base_mask[0,-4:-2,0,:,:]
            gt_mask_base_center_frame_stereo = base_mask[0,-6:-4,0,:,:]
            
            gt_mask_plus_first_frame_stereo = plus_mask[0,-2:,0,:,:]
            gt_mask_plus_last_frame_stereo = plus_mask[0,-4:-2,0,:,:]
            gt_mask_plus_center_frame_stereo = plus_mask[0,-6:-4,0,:,:]
            
            
            gt_mask_base_first_frame_stereo_left = gt_mask_base_first_frame_stereo[0].cpu().numpy()
            gt_mask_base_first_frame_stereo_left = convert_a_numpy_to_uint8(gt_mask_base_first_frame_stereo_left)
            gt_mask_base_first_frame_stereo_right = gt_mask_base_first_frame_stereo[1].cpu().numpy()
            gt_mask_base_first_frame_stereo_right = convert_a_numpy_to_uint8(gt_mask_base_first_frame_stereo_right)
            gt_mask_base_last_framme_stereo_left = gt_mask_base_last_frame_stereo[0].cpu().numpy()
            gt_mask_base_last_framme_stereo_left = convert_a_numpy_to_uint8(gt_mask_base_last_framme_stereo_left)
            gt_mask_base_last_framme_stereo_right = gt_mask_base_last_frame_stereo[1].cpu().numpy()
            gt_mask_base_last_framme_stereo_right = convert_a_numpy_to_uint8(gt_mask_base_last_framme_stereo_right)
            
            gt_mask_base_center_frame_stereo_left = gt_mask_base_center_frame_stereo[0].cpu().numpy()
            gt_mask_base_center_frame_stereo_left = convert_a_numpy_to_uint8(gt_mask_base_center_frame_stereo_left)
            gt_mask_base_center_frame_stereo_right = gt_mask_base_center_frame_stereo[1].cpu().numpy()
            gt_mask_base_center_frame_stereo_right = convert_a_numpy_to_uint8(gt_mask_base_center_frame_stereo_right)
            
            gt_mask_plus_first_frame_stereo_left = gt_mask_plus_first_frame_stereo[0].cpu().numpy()
            gt_mask_plus_first_frame_stereo_left = convert_a_numpy_to_uint8(gt_mask_plus_first_frame_stereo_left)
            gt_mask_plus_first_frame_stereo_right = gt_mask_plus_first_frame_stereo[1].cpu().numpy()
            gt_mask_plus_first_frame_stereo_right = convert_a_numpy_to_uint8(gt_mask_plus_first_frame_stereo_right)
            
            gt_mask_plus_last_frame_stereo_left = gt_mask_plus_last_frame_stereo[0].cpu().numpy()
            gt_mask_plus_last_frame_stereo_left = convert_a_numpy_to_uint8(gt_mask_plus_last_frame_stereo_left)
            gt_mask_plus_last_frame_stereo_right = gt_mask_plus_last_frame_stereo[1].cpu().numpy()
            gt_mask_plus_last_frame_stereo_right = convert_a_numpy_to_uint8(gt_mask_plus_last_frame_stereo_right)
            
            gt_mask_plus_center_frame_stereo_left = gt_mask_plus_center_frame_stereo[0].cpu().numpy()
            gt_mask_plus_center_frame_stereo_left = convert_a_numpy_to_uint8(gt_mask_plus_center_frame_stereo_left)
            gt_mask_plus_center_frame_stereo_right = gt_mask_plus_center_frame_stereo[1].cpu().numpy()
            gt_mask_plus_center_frame_stereo_right = convert_a_numpy_to_uint8(gt_mask_plus_center_frame_stereo_right)
            
            
            
            skimage.io.imsave(os.path.join(GT_Base_Mask_folder,"first_stereo_left.png"),gt_mask_base_first_frame_stereo_left)
            skimage.io.imsave(os.path.join(GT_Base_Mask_folder,"first_stereo_right.png"),gt_mask_base_first_frame_stereo_right)
            skimage.io.imsave(os.path.join(GT_Base_Mask_folder,"last_stereo_left.png"),gt_mask_base_last_framme_stereo_left)
            skimage.io.imsave(os.path.join(GT_Base_Mask_folder,"last_stereo_right.png"),gt_mask_base_last_framme_stereo_right)
            skimage.io.imsave(os.path.join(GT_Base_Mask_folder,"center_stereo_left.png"),gt_mask_base_center_frame_stereo_left)
            skimage.io.imsave(os.path.join(GT_Base_Mask_folder,"center_stereo_right.png"),gt_mask_base_center_frame_stereo_right)
            
            skimage.io.imsave(os.path.join(GT_Plus_Mask_folder,"first_stereo_left.png"),gt_mask_plus_first_frame_stereo_left)
            skimage.io.imsave(os.path.join(GT_Plus_Mask_folder,"first_stereo_right.png"),gt_mask_plus_first_frame_stereo_right)
            skimage.io.imsave(os.path.join(GT_Plus_Mask_folder,"last_stereo_left.png"),gt_mask_plus_last_frame_stereo_left)
            skimage.io.imsave(os.path.join(GT_Plus_Mask_folder,"last_stereo_right.png"),gt_mask_plus_last_frame_stereo_right)
            skimage.io.imsave(os.path.join(GT_Plus_Mask_folder,"center_stereo_left.png"),gt_mask_plus_center_frame_stereo_left)
            skimage.io.imsave(os.path.join(GT_Plus_Mask_folder,"center_stereo_right.png"),gt_mask_plus_center_frame_stereo_right)
            
            
            
            ################################# Save the Raw Alpha Map here ########################################################
            
            rendered_alpha_base_first_stereo = rendered_alpha_base[:,-2:,:,:]
            rendered_alpha_base_first_stereo_left = rendered_alpha_base_first_stereo[0][0].cpu().numpy()
            rendered_alpha_base_first_stereo_right = rendered_alpha_base_first_stereo[0][1].cpu().numpy()
            rendered_alpha_base_last_stereo = rendered_alpha_base[:,-4:-2,:,:]
            rendered_alpha_base_last_stereo_left = rendered_alpha_base_last_stereo[0][0].cpu().numpy()
            rendered_alpha_base_last_stereo_right = rendered_alpha_base_last_stereo[0][1].cpu().numpy()
            rendered_alpha_base_center_stereo = rendered_alpha_base[:,-6:-4,:,:]
            rendered_alpha_base_center_stereo_left = rendered_alpha_base_center_stereo[0][0].cpu().numpy()
            rendered_alpha_base_center_stereo_right = rendered_alpha_base_center_stereo[0][1].cpu().numpy()
            
            rendered_alpha_plus_first_stereo = rendered_alpha_plus[:,-2:,:,:]   
            rendered_alpha_plus_first_stereo_left = rendered_alpha_plus_first_stereo[0][0].cpu().numpy()
            rendered_alpha_plus_first_stereo_right = rendered_alpha_plus_first_stereo[0][1].cpu().numpy()
            rendered_alpha_plus_last_stereo = rendered_alpha_plus[:,-4:-2,:,:]
            rendered_alpha_plus_last_stereo_left = rendered_alpha_plus_last_stereo[0][0].cpu().numpy()
            rendered_alpha_plus_last_stereo_right = rendered_alpha_plus_last_stereo[0][1].cpu().numpy()
            rendered_alpha_plus_center_stereo = rendered_alpha_plus[:,-6:-4,:,:]
            rendered_alpha_plus_center_stereo_left = rendered_alpha_plus_center_stereo[0][0].cpu().numpy()
            rendered_alpha_plus_center_stereo_right = rendered_alpha_plus_center_stereo[0][1].cpu().numpy()
            
            

            visualize_mask_heatmap(
                            rendered_alpha_base_first_stereo_left,
                            save_path=os.path.join(Raw_Base_Alpha_Map_folder, "first_stereo_left_heatmap.png"),
                            title="rendered alpha base first stereo left",
                            colorbar_label="alpha",
                        )
            
            visualize_mask_heatmap(
                            rendered_alpha_base_first_stereo_right,
                            save_path=os.path.join(Raw_Base_Alpha_Map_folder, "first_stereo_right_heatmap.png"),
                            title="rendered alpha base first stereo right",
                            colorbar_label="alpha",
                        )
            
            visualize_mask_heatmap(
                
                rendered_alpha_base_last_stereo_left,
                save_path=os.path.join(Raw_Base_Alpha_Map_folder, "last_stereo_left_heatmap.png"),
                title="rendered alpha base last stereo left",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(
                
                rendered_alpha_base_last_stereo_right,
                save_path=os.path.join(Raw_Base_Alpha_Map_folder, "last_stereo_right_heatmap.png"),
                title="rendered alpha base last stereo right",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(
                
                rendered_alpha_base_center_stereo_left,
                save_path=os.path.join(Raw_Base_Alpha_Map_folder, "center_stereo_left_heatmap.png"),
                title="rendered alpha base center stereo left",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(
                
                rendered_alpha_base_center_stereo_right,
                save_path=os.path.join(Raw_Base_Alpha_Map_folder, "center_stereo_right_heatmap.png"),
                title="rendered alpha base center stereo right",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(
                
                rendered_alpha_plus_first_stereo_left,
                save_path=os.path.join(Raw_Plus_Alpha_Map_folder, "first_stereo_left_heatmap.png"),
                title="rendered alpha plus first stereo left",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(
                rendered_alpha_plus_first_stereo_right,
                save_path=os.path.join(Raw_Plus_Alpha_Map_folder, "first_stereo_right_heatmap.png"),
                title="rendered alpha plus first stereo right",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(
                rendered_alpha_plus_last_stereo_left,
                save_path=os.path.join(Raw_Plus_Alpha_Map_folder, "last_stereo_left_heatmap.png"),
                title="rendered alpha plus last stereo left",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(
                rendered_alpha_plus_last_stereo_right,
                save_path=os.path.join(Raw_Plus_Alpha_Map_folder, "last_stereo_right_heatmap.png"),
                title="rendered alpha plus last stereo right",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(
                rendered_alpha_plus_center_stereo_left,
                save_path=os.path.join(Raw_Plus_Alpha_Map_folder, "center_stereo_left_heatmap.png"),
                title="rendered alpha plus center stereo left",
                colorbar_label="alpha",
            )
            
            visualize_mask_heatmap(                
                rendered_alpha_plus_center_stereo_right,
                save_path=os.path.join(Raw_Plus_Alpha_Map_folder, "center_stereo_right_heatmap.png"),
                title="rendered alpha plus center stereo right",
                colorbar_label="alpha",
                )
            
            
            # ################################ Save the Comparative Alpha Map Here #######################################
            comparsion_mask = (rendered_alpha_plus >= rendered_alpha_base).to(
                dtype=rendered_alpha_plus.dtype
            )
            
            comparsion_mask_first_stereo = comparsion_mask[:,-2:,:,:]
            comparsion_mask_last_stereo = comparsion_mask[:,-4:-2,:,:]
            comparsion_mask_center_stereo = comparsion_mask[:,-6:-4,:,:]
            
            comparsion_mask_first_stereo_left = comparsion_mask_first_stereo[0][0].cpu().numpy()
            comparsion_mask_first_stereo_right = comparsion_mask_first_stereo[0][1].cpu().numpy()
            comparsion_mask_last_stereo_left = comparsion_mask_last_stereo[0][0].cpu().numpy()
            comparsion_mask_last_stereo_right = comparsion_mask_last_stereo[0][1].cpu().numpy()
            comparsion_mask_center_stereo_left = comparsion_mask_center_stereo[0][0].cpu().numpy()
            comparsion_mask_center_stereo_right = comparsion_mask_center_stereo[0][1].cpu().numpy()
            
            comparsion_mask_first_stereo_left = convert_a_numpy_to_uint8(comparsion_mask_first_stereo_left)
            comparsion_mask_first_stereo_right = convert_a_numpy_to_uint8(comparsion_mask_first_stereo_right)
            comparsion_mask_last_stereo_left = convert_a_numpy_to_uint8(comparsion_mask_last_stereo_left)
            comparsion_mask_last_stereo_right = convert_a_numpy_to_uint8(comparsion_mask_last_stereo_right)
            comparsion_mask_center_stereo_left = convert_a_numpy_to_uint8(comparsion_mask_center_stereo_left)
            comparsion_mask_center_stereo_right = convert_a_numpy_to_uint8(comparsion_mask_center_stereo_right)
            
            skimage.io.imsave(os.path.join(Comparasion_Plus_Alpha_Map_folder, "first_stereo_left.png"),comparsion_mask_first_stereo_left)
            skimage.io.imsave(os.path.join(Comparasion_Plus_Alpha_Map_folder, "first_stereo_right.png"),comparsion_mask_first_stereo_right)
            skimage.io.imsave(os.path.join(Comparasion_Plus_Alpha_Map_folder, "last_stereo_left.png"),comparsion_mask_last_stereo_left)
            skimage.io.imsave(os.path.join(Comparasion_Plus_Alpha_Map_folder, "last_stereo_right.png"),comparsion_mask_last_stereo_right)
            skimage.io.imsave(os.path.join(Comparasion_Plus_Alpha_Map_folder, "center_stereo_left.png"),comparsion_mask_center_stereo_left)
            skimage.io.imsave(os.path.join(Comparasion_Plus_Alpha_Map_folder, "center_stereo_right.png"),comparsion_mask_center_stereo_right)
        

        
        return final_eval_dict
        
    

@torch.no_grad()
def convert_pil_to_tensor(pil_image):
    img = torch.from_numpy(np.array(pil_image)).type(torch.float32)/255.0
    img = img.permute(2,0,1)
    img = img.unsqueeze(0).unsqueeze(0)
    return img
        

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


def get_mean(list):
    return sum(list)*1.0/len(list)

def saved_into_json(data_dict,path):
    import json
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)


def fuse_rgb_by_gt_error(
    rgb1: torch.Tensor,
    rgb2: torch.Tensor,
    gt: torch.Tensor,
    metric: str = "l1",
):
    """
    Compare rgb1 and rgb2 against GT pixel-wisely, and fuse the better one.

    Args:
        rgb1: Tensor of shape [1, V, 3, H, W]
        rgb2: Tensor of shape [1, V, 3, H, W]
        gt:   Tensor of shape [1, V, 3, H, W]
        metric: "l1" or "l2"

    Returns:
        mask1:     Tensor of shape [1, V, 1, H, W], 1 where rgb1 is better
        mask2:     Tensor of shape [1, V, 1, H, W], 1 where rgb2 is better
        fused_rgb: Tensor of shape [1, V, 3, H, W]
    """
    if rgb1.shape != rgb2.shape or rgb1.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: rgb1={rgb1.shape}, rgb2={rgb2.shape}, gt={gt.shape}"
        )
    if rgb1.ndim != 5 or rgb1.shape[0] != 1 or rgb1.shape[2] != 3:
        raise ValueError(
            f"Expected input shape [1, V, 3, H, W], but got {rgb1.shape}"
        )

    if metric not in ["l1", "l2"]:
        raise ValueError(f"Unsupported metric: {metric}. Use 'l1' or 'l2'.")

    # pixel-wise error against GT, reduced over channel dimension
    if metric == "l1":
        err1 = torch.mean(torch.abs(rgb1 - gt), dim=2, keepdim=True)  # [1,V,1,H,W]
        err2 = torch.mean(torch.abs(rgb2 - gt), dim=2, keepdim=True)
    else:  # l2
        err1 = torch.mean((rgb1 - gt) ** 2, dim=2, keepdim=True)      # [1,V,1,H,W]
        err2 = torch.mean((rgb2 - gt) ** 2, dim=2, keepdim=True)

    # mask1: rgb1 is better or equal
    mask1 = (err1 <= err2).to(rgb1.dtype)  # [1,V,1,H,W]
    mask2 = 1.0 - mask1

    # expand mask to RGB channels
    mask1_rgb = mask1.expand(-1, -1, 3, -1, -1)
    mask2_rgb = mask2.expand(-1, -1, 3, -1, -1)

    fused_rgb = rgb1 * mask1_rgb + rgb2 * mask2_rgb

    return mask1, mask2, fused_rgb



def convert_a_numpy_to_uint8(numpy_array):
    return (numpy_array*255.0).astype(np.uint8)



if __name__=="__main__":
    pass