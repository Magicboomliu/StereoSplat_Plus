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
import skimage.io
from .metrics import convert_depth_to_disp,compute_psnr_ssim
import matplotlib.pyplot as plt
import math
from .gaussian import GaussianRenderer
from .losses import Custom_Depth_Loss
from .utils.interpolation import interpolate_extrinsics
from tqdm import tqdm
from .gs_fuse import transform_g2_to_g1
from .utilsdir.gaussain_fusion import fuse_gaussians_by_voxel_with_depth_batched_vectorized,fuse_gaussians_by_voxel_with_depth_scatter_batched


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

class VolumeFusion(BaseModule):
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

    def forward(self,batch,mode='train',iter=0,cfg=None):
        # get inpout_batch_dict
        
        if 'pair_images' in cfg.keys():
            input_batch_dict,output_batch_dict = self.prepare_input_batch_data(batch=batch)
        else:
            input_batch_dict,output_batch_dict = self.prepare_input_batch_data(batch=batch)
        

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

    # def validation_step_with_token_names(self, batch, val_result_savedir,cfg=None):
        
    #     bin_token_name = batch['bin_token'][0][:-4]
    #     # loss and loss terms
    #     with torch.no_grad():
    #         loss, loss_terms,rendered_fusion_list,\
    #             rendered_volume_list,rendered_cv_results_list, \
    #                 predicted_input_depth,input_sparse_gt_depth,\
    #                     output_rgb,sparse_depth_gt,input_images = self.forward(batch,mode='val',
    #                                                         cfg=cfg)

    #     batch_data_for_eval = {
    #         "output_gt_rgb": output_rgb,
    #         "output_gt_sparse_depth": sparse_depth_gt,
    #         "input_images": input_images,
    #         "input_gt_sparse_gt": input_sparse_gt_depth,
    #         "predicted_input_depth": predicted_input_depth,
    #         "rendered_fusion": rendered_fusion_list,
    #         "rendered_cost_volume":rendered_cv_results_list,
    #         "rendered_volume":rendered_volume_list,
    #         "bin_token_name": bin_token_name
    #     }
            
            
            
    #     # saved into the val_result_dir: the visualiation results
        
    #     # rendered RGBs
    #     # rendered Depths
    #     # GT RGBs
    #     # GT Depths
    #     # Estimated Depths
        
    #     metrics_rendered_rgb_list,metrics_rendered_depth_list,metrics_estimated_depth_list,final_rendered_rgb_list,final_rendered_depth_list,final_gt_rgb_list,final_gt_depth_list = self.save_val_results_with_tokens(batch_data_for_eval,val_result_savedir,cfg=cfg)
        
    #     return metrics_rendered_rgb_list,metrics_rendered_depth_list,metrics_estimated_depth_list,final_rendered_rgb_list,final_rendered_depth_list,final_gt_rgb_list,final_gt_depth_list

    # def save_val_results_with_tokens(self,batch_data_for_eval,saved_dir,cfg):
        
    #     '''input batch data for evaluation'''
        
    #     rendered_fusion = batch_data_for_eval['rendered_fusion']
    #     rendered_volume = batch_data_for_eval['rendered_volume']
    #     rendered_cv = batch_data_for_eval['rendered_cost_volume']
        
    #     rendered_list = [rendered_fusion,rendered_volume,rendered_cv]
    #     names = ['fusion','volume','cost_volume']
        
    #     metrics_rendered_rgb_list = []
    #     metrics_rendered_depth_list = []
    #     metrics_estimated_depth_list = []
        
    #     # from "fusion"----> "volume" -----> "cost volume"
        
    #     final_rendered_rgb_list = []
    #     final_rendered_depth_list = []
    #     final_gt_rgb_list = []
    #     final_gt_depth_list = []
        
        
    #     for idx, rendered_output in enumerate(rendered_list):
            
    #         current_name = names[idx]
    #         output_rgb_meter_dict = dict()
    #         # get the psnr and ssim for the output view
    #         output_rendered_rgb = rendered_output[0] #torch.Size([1, 6, 3, 224, 832]) # 0 is the rgb
    #         output_gt_rgb = batch_data_for_eval['output_gt_rgb'] #torch.Size([1, 6, 3, 224, 832])
            
    #         # rendered center
    #         center_frame_left_est =  output_rendered_rgb[:,0,:,:,:]
    #         center_frame_right_est = output_rendered_rgb[:,2,:,:,:]
    #         last_frame_left_est =  output_rendered_rgb[:,1,:,:,:]
    #         last_frame_right_est =  output_rendered_rgb[:,3,:,:,:]
    #         first_frame_left_est = output_rendered_rgb[:,4,:,:,:]
    #         first_frame_right_est = output_rendered_rgb[:,5,:,:,:]
            
    #         center_frame_left_gt =  output_gt_rgb[:,0,:,:,:]
    #         center_frame_right_gt = output_gt_rgb[:,2,:,:,:]
    #         last_frame_left_gt =  output_gt_rgb[:,1,:,:,:]
    #         last_frame_right_gt =  output_gt_rgb[:,3,:,:,:]
    #         first_frame_left_gt = output_gt_rgb[:,4,:,:,:]
    #         first_frame_right_gt = output_gt_rgb[:,5,:,:,:]
            
            
    #         cl_psnr,cl_ssim = compute_psnr_ssim(pred=center_frame_left_est,target=center_frame_left_gt)
    #         cr_psnr,cr_ssim = compute_psnr_ssim(pred=center_frame_right_est,target=center_frame_right_gt)
    #         ll_psnr,ll_ssim = compute_psnr_ssim(pred=last_frame_left_est,target=last_frame_left_gt)
    #         lr_psnr,lr_ssim = compute_psnr_ssim(pred=last_frame_right_est,target=last_frame_right_gt)
    #         fl_psnr,fl_ssim = compute_psnr_ssim(pred=first_frame_left_est,target=first_frame_left_gt)
    #         fr_psnr,fr_ssim = compute_psnr_ssim(pred=first_frame_right_est,target=first_frame_right_gt)
            
    #         output_rgb_meter_dict['center_view'] = dict()
    #         output_rgb_meter_dict['center_view']['left'] = dict()
    #         output_rgb_meter_dict['center_view']['left']['psnr'] = cl_psnr.data.item()
    #         output_rgb_meter_dict['center_view']['left']['ssim'] = cl_ssim.data.item()

    #         output_rgb_meter_dict['center_view']['right'] = dict()
    #         output_rgb_meter_dict['center_view']['right']['psnr'] = cr_psnr.data.item()
    #         output_rgb_meter_dict['center_view']['right']['ssim'] = cr_ssim.data.item()
            

    #         output_rgb_meter_dict['last_view'] = dict()
    #         output_rgb_meter_dict['last_view']['left'] = dict()
    #         output_rgb_meter_dict['last_view']['left']['psnr'] = ll_psnr.data.item()
    #         output_rgb_meter_dict['last_view']['left']['ssim'] = ll_ssim.data.item()

    #         output_rgb_meter_dict['last_view']['right'] = dict()
    #         output_rgb_meter_dict['last_view']['right']['psnr'] = lr_psnr.data.item()
    #         output_rgb_meter_dict['last_view']['right']['ssim'] = lr_ssim.data.item()


    #         output_rgb_meter_dict['first_view'] = dict()
    #         output_rgb_meter_dict['first_view']['left'] = dict()
    #         output_rgb_meter_dict['first_view']['left']['psnr'] = fl_psnr.data.item()
    #         output_rgb_meter_dict['first_view']['left']['ssim'] = fl_ssim.data.item()

    #         output_rgb_meter_dict['first_view']['right'] = dict()
    #         output_rgb_meter_dict['first_view']['right']['psnr'] = fr_psnr.data.item()
    #         output_rgb_meter_dict['first_view']['right']['ssim'] = fr_ssim.data.item()

            
    #         # get the MAE and the MSE of the output view
    #         output_depth_meter_dict = dict()
    #         output_rendered_depth = rendered_output[1] #torch.Size([1, 6, 3, 224, 832]) # 1 is the depth
    #         output_gt_depth = batch_data_for_eval['output_gt_sparse_depth'] #torch.Size([1, 6, 3, 224, 832])
            

    #         center_frame_left_est_depth =  output_rendered_depth[:,0,:,:]
    #         center_frame_right_est_depth = output_rendered_depth[:,2,:,:]
    #         last_frame_left_est_depth =  output_rendered_depth[:,1,:,:]
    #         last_frame_right_est_depth =  output_rendered_depth[:,3,:,:]
    #         first_frame_left_est_depth = output_rendered_depth[:,4,:,:]
    #         first_frame_right_est_depth = output_rendered_depth[:,5,:,:]

    #         center_frame_left_gt_depth =  output_gt_depth[:,0,:,:]
    #         center_frame_right_gt_depth = output_gt_depth[:,2,:,:]
    #         last_frame_left_gt_depth =  output_gt_depth[:,1,:,:]
    #         last_frame_right_gt_depth =  output_gt_depth[:,3,:,:]
    #         first_frame_left_gt_depth = output_gt_depth[:,4,:,:]
    #         first_frame_right_gt_depth = output_gt_depth[:,5,:,:]

    #         cl_mae,cl_mse = compute_depth_mae_mse(depth_pred=center_frame_left_est_depth,
    #                             depth_gt=center_frame_left_gt_depth)
            
    #         cr_mae,cr_mse = compute_depth_mae_mse(depth_pred=center_frame_right_est_depth,
    #                             depth_gt=center_frame_right_gt_depth)
            
    #         ll_mae,ll_mse = compute_depth_mae_mse(depth_pred=last_frame_left_est_depth,
    #                             depth_gt=last_frame_left_gt_depth)
            
    #         lr_mae,lr_mse = compute_depth_mae_mse(depth_pred=last_frame_right_est_depth,
    #                             depth_gt=last_frame_right_gt_depth)
            
    #         fl_mae,fl_mse = compute_depth_mae_mse(depth_pred=first_frame_left_est_depth,
    #                             depth_gt=first_frame_left_gt_depth)

    #         fr_mae,fr_mse = compute_depth_mae_mse(depth_pred=first_frame_right_est_depth,
    #                             depth_gt=first_frame_right_gt_depth)

    #         output_depth_meter_dict['center_view'] = dict()
    #         output_depth_meter_dict['center_view']['left'] = dict()
    #         output_depth_meter_dict['center_view']['left']['mae'] = cl_mae.data.item()
    #         output_depth_meter_dict['center_view']['left']['mse'] = cl_mse.data.item()

    #         output_depth_meter_dict['center_view']['right'] = dict()
    #         output_depth_meter_dict['center_view']['right']['mae'] = cr_mae.data.item()
    #         output_depth_meter_dict['center_view']['right']['mse'] = cr_mse.data.item()
            

    #         output_depth_meter_dict['last_view'] = dict()
    #         output_depth_meter_dict['last_view']['left'] = dict()
    #         output_depth_meter_dict['last_view']['left']['mae'] = ll_mae.data.item()
    #         output_depth_meter_dict['last_view']['left']['mse'] = ll_mse.data.item()

    #         output_depth_meter_dict['last_view']['right'] = dict()
    #         output_depth_meter_dict['last_view']['right']['mae'] = lr_mae.data.item()
    #         output_depth_meter_dict['last_view']['right']['mse'] = lr_mse.data.item()


    #         output_depth_meter_dict['first_view'] = dict()
    #         output_depth_meter_dict['first_view']['left'] = dict()
    #         output_depth_meter_dict['first_view']['left']['mae'] = fl_mae.data.item()
    #         output_depth_meter_dict['first_view']['left']['mse'] = fl_mse.data.item()

    #         output_depth_meter_dict['first_view']['right'] = dict()
    #         output_depth_meter_dict['first_view']['right']['mae'] = fr_mae.data.item()
    #         output_depth_meter_dict['first_view']['right']['mse'] = fr_mse.data.item()

            
    #         # get the MAE and the MSE of the input view (sterep)
    #         input_depth_meter_dict = dict()
    #         input_depth_estimation = batch_data_for_eval['predicted_input_depth'] #torch.Size([1, 2, 224, 832])
    #         input_gt_depth = batch_data_for_eval['input_gt_sparse_gt'] #torch.Size([1, 2, 224, 832])
            
    #         input_depth_estimation_left = input_depth_estimation[:,0,:,:]
    #         input_depth_estimation_right = input_depth_estimation[:,1,:,:]
            
    #         input_gt_depth_sparse_left = input_gt_depth[:,0,:,:]
    #         input_gt_depth_sparse_right = input_gt_depth[:,1,:,:]
            
            
    #         input_l_mae,input_l_mse =  compute_depth_mae_mse(depth_pred=input_depth_estimation_left,
    #                             depth_gt=input_gt_depth_sparse_left)
            
    #         input_r_mae, input_r_mse = compute_depth_mae_mse(depth_pred=input_depth_estimation_right,
    #                             depth_gt=input_gt_depth_sparse_right)
            
            
    #         input_depth_meter_dict['input_depth'] = dict()
    #         input_depth_meter_dict['input_depth']['left'] = dict()
    #         input_depth_meter_dict['input_depth']['left']['mae'] = input_l_mae.data.item()
    #         input_depth_meter_dict['input_depth']['left']['mse'] = input_l_mse.data.item()
            
    #         input_depth_meter_dict['input_depth']['right'] = dict()
    #         input_depth_meter_dict['input_depth']['right']['mae'] = input_r_mae.data.item()
    #         input_depth_meter_dict['input_depth']['right']['mse'] = input_r_mse.data.item()
            

    #         metrics_rendered_rgb_list.append(output_rgb_meter_dict)
    #         metrics_rendered_depth_list.append(output_depth_meter_dict)
    #         metrics_estimated_depth_list.append(input_depth_meter_dict)
            
            
    #         final_rendered_rgb_list.append(rendered_output[0])
    #         final_rendered_depth_list.append(rendered_output[1])
            
    #         final_gt_rgb_list.append(output_gt_rgb)
    #         final_gt_depth_list.append(output_gt_depth)
            
            
            
    #     return metrics_rendered_rgb_list,metrics_rendered_depth_list,metrics_estimated_depth_list,final_rendered_rgb_list,final_rendered_depth_list,final_gt_rgb_list,final_gt_depth_list

    # def forward_kitti360_videos(self,batch,saved_dir,cfg=None):
        
    #     bin_token_name = batch['bin_token']
        
    #     # get inpout_batch_dict
    #     input_batch_dict,output_batch_dict = self.prepare_input_batch_data(batch=batch)

    #     img =input_batch_dict["imgs"] #[B,6,3,H,W]
    #     height,width = img.shape[-2:]
    #     bs = img.shape[0]
        
    #     input_pseudo_depth = input_batch_dict['pseudo_depths']
    #     input_sparse_gt_depth = input_batch_dict['sparse_depths']
    #     img_feats = self.extract_img_feat(img=img) # feature list----> 4 layers
                
    #     # perform the cost volume-based 
    #     # perform the cost volume-based 
    #     gaussians_cv,gaussians_feat,pred_depths = self.costvolume_gs(input_batch_dict,cfg=cfg,
    #                                                       images_feat=img_feats[0])
        



    #     # volume-gs prediction
    #     pc_range = self.dataset_params.pc_range
    #     x_start, y_start, z_start, x_end, y_end, z_end = pc_range
    #     # batch-wise saved the gaussain-pixel and the feature-pixel
    #     gaussians_cv_mask, gaussians_feat_mask = [], []
    #     for b in range(bs):
    #         mask_pixel_i = (gaussians_cv[b, :, 0] >= x_start) & (gaussians_cv[b, :, 0] <= x_end) & \
    #                     (gaussians_cv[b, :, 1] >= y_start) & (gaussians_cv[b, :, 1] <= y_end) & \
    #                     (gaussians_cv[b, :, 2] >= z_start) & (gaussians_cv[b, :, 2] <= z_end)
    #         # get the valid gaussains in the pixel splat
    #         gaussians_cv_mask_i = gaussians_cv[b][mask_pixel_i]
    #         # get the valid feature in the pixel splat
    #         gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
    #         gaussians_cv_mask.append(gaussians_cv_mask_i)
    #         gaussians_feat_mask.append(gaussians_feat_mask_i)
        
        

        

    #     gaussians_volume = self.volume_gs(
    #             [img_feats[0]],
    #             gaussians_cv_mask,
    #             gaussians_feat_mask,
    #             input_batch_dict["img_metas"])
    

    #     # Make Sure the estimate gaussains are valid
    #     gaussians_cv = sanitize_gaussians_tensor(gaussians_cv)
    #     gaussians_volume = sanitize_gaussians_tensor(gaussians_volume)
        
    #     gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
    #     bs = gaussians_all.shape[0] # batch size is 2


    #     data_dict = output_batch_dict

    #     c2w_lf_left = data_dict["output_c2ws"][:, 1]
    #     c2w_lf_right = data_dict["output_c2ws"][:, 3]
    #     c2w_ff_left = data_dict["output_c2ws"][:, 4]
    #     c2w_ff_right = data_dict["output_c2ws"][:, 5]
    #     c2w_cf_left = data_dict["output_c2ws"][:, 0] #(1,2,4,4)
    #     c2w_cf_right = data_dict["output_c2ws"][:, 2] #(1,2,4,4)
        
    #     ''' 
    #         Movement 0:  Center Left Rotation 45 Degree
    #         Movement 1:  Center Rotation Back
    #     '''
    
    #     # left backward 3---------rotation 45 ------rotation back 
    #     theta = -math.pi / 4  # 
    #     rot_0 = torch.tensor([
    #         [math.cos(theta), -math.sin(theta), 0],
    #         [math.sin(theta),  math.cos(theta), 0],
    #         [0,                0,               1]
    #     ], dtype=torch.float32).to(c2w_cf_left.device)

    #     c2w_cf_left_rot2_right = c2w_cf_left.clone()
    #     c2w_cf_left_rot2_right[...,:3,:3] = rot_0@c2w_cf_left_rot2_right[...,:3,:3]

    #     c2w_lf_left_rot2_right = c2w_lf_left.clone()
    #     c2w_lf_left_rot2_right[...,:3,:3] = rot_0 @ c2w_lf_left_rot2_right[...,:3,:3]
        
    #     c2w_ff_left_rot2_right = c2w_ff_left.clone()
    #     c2w_ff_left_rot2_right[...,:3,:3] = rot_0 @ c2w_ff_left_rot2_right[...,:3,:3]

    #     ''' 
    #         Movement 2:  Center Left Cam to Center Right
    #         Movement 3:  Center Right Rot Inside
    #         Movement 4: Rotation Back
    #     '''
        
    #     # right backward 3---------rotation 45 ------rotation back:  short +1
    #     theta = math.pi / 4  # 
    #     rot_1 = torch.tensor([
    #         [math.cos(theta), -math.sin(theta), 0],
    #         [math.sin(theta),  math.cos(theta), 0],
    #         [0,                0,               1]
    #     ], dtype=torch.float32).to(c2w_cf_left.device)
        
    #     c2w_cf_right_rot2_left = c2w_cf_right.clone()
    #     c2w_cf_right_rot2_left[...,:3,:3] = rot_1 @ c2w_cf_right_rot2_left[...,:3,:3]
        
    #     c2w_lf_right_rot2_left = c2w_lf_right.clone()
    #     c2w_lf_right_rot2_left[...,:3,:3] = rot_1 @ c2w_lf_right_rot2_left[...,:3,:3]
        
    #     c2w_ff_right_rot2_left = c2w_ff_right.clone() 
    #     c2w_ff_right_rot2_left[...,:3,:3] = rot_1 @ c2w_ff_right_rot2_left[...,:3,:3]
        
        
    #     ''' Movement 5: from right to left '''
    #     '''Movement 6: From Center left to Last Left'''
    #     num_frames_short = 60
    #     num_frames_long = 60
        
    #     t_short = torch.linspace(0, 1, num_frames_short, dtype=torch.float32, device=self.device)
    #     t_long = torch.linspace(0, 1 - 1 / (num_frames_long + 1), num_frames_long, dtype=torch.float32, device=self.device)
    #     # center left rot
    #     movement_0 = interpolate_extrinsics(c2w_cf_left,c2w_cf_left_rot2_right,t_short)
    #     # center left rot back
    #     movement_1 = interpolate_extrinsics(c2w_cf_left_rot2_right,c2w_cf_left,t_short)
    #     # center left to right
    #     movement_2 = interpolate_extrinsics(c2w_cf_left,c2w_cf_right,t_short)
    #     # center right rot
    #     movement_3 = interpolate_extrinsics(c2w_cf_right,c2w_cf_right_rot2_left,t_short)
    #     # center right rot back
    #     movement_4 = interpolate_extrinsics(c2w_cf_right_rot2_left,c2w_cf_right,t_short)
    #     # center right to left
    #     movement_5 = interpolate_extrinsics(c2w_cf_right,c2w_cf_left,t_short)
    #     # center left to last left
    #     movement_6 = interpolate_extrinsics(c2w_cf_left,c2w_lf_left,t_short)
    #     # last left to rot
    #     movement_7 = interpolate_extrinsics(c2w_lf_left,c2w_lf_left_rot2_right,t_short)
    #     # last left rot back
    #     movement_8 = interpolate_extrinsics(c2w_lf_left_rot2_right,c2w_lf_left,t_short)
    #     # last left to last right
    #     movement_9 = interpolate_extrinsics(c2w_lf_left,c2w_lf_right,t_short)
        
    #     # last right rot
    #     movement_10 = interpolate_extrinsics(c2w_lf_right,c2w_lf_right_rot2_left,t_short)
    #     movement_11 = interpolate_extrinsics(c2w_lf_right_rot2_left, c2w_lf_right,t_short)
    #     movement_12 = interpolate_extrinsics(c2w_lf_right,c2w_lf_left ,t_short)
    #     movement_13 = interpolate_extrinsics(c2w_lf_left,c2w_ff_left,t_short)
    #     movement_14 = interpolate_extrinsics(c2w_ff_left,c2w_ff_left_rot2_right,t_short)
    #     movement_15 = interpolate_extrinsics(c2w_ff_left_rot2_right,c2w_ff_left,t_short)
    #     movement_16 = interpolate_extrinsics(c2w_ff_left,c2w_ff_right,t_short)
    #     movement_17 = interpolate_extrinsics(c2w_ff_right,c2w_ff_right_rot2_left,t_short)
    #     movement_18 = interpolate_extrinsics(c2w_ff_right_rot2_left,c2w_ff_right,t_short)
        
    #     c2w_interp = torch.cat([movement_0, movement_1, movement_2,
    #                             movement_3, movement_4,movement_5,
    #                             movement_6,movement_7,
    #                             movement_8,movement_9,
    #                             movement_10,movement_11,
    #                             movement_12,movement_13,
    #                             movement_14,movement_15,
    #                             movement_16,movement_17,
    #                             movement_18
    #                             ], dim=1)  


    #     N_Chunks = 10
    #     interval = int(c2w_interp.shape[1]//N_Chunks)
        
    #     rendered_rgb_list = []
    #     rendered_depth_list = []      


    #     for idx in tqdm(range(N_Chunks)):
            
    #         current_c2w_interp = c2w_interp[:,idx*interval:(idx+1)*interval,:]
            
    #         current_fovxs_interp = output_batch_dict["output_fovxs"][:, -6:-5].repeat(1, current_c2w_interp.shape[1])   # [4,960] --> Center
    #         current_fovys_interp =output_batch_dict["output_fovys"][:, -6:-5].repeat(1, current_c2w_interp.shape[1]) 

    #         render_pkg_cv = self.renderer.render(
    #             gaussians=gaussians_all,
    #             c2w=current_c2w_interp,
    #             fovx=current_fovxs_interp ,
    #             fovy=current_fovys_interp,
    #             rays_o=None,
    #             rays_d=None
    #         )
    #         rendered_results = render_pkg_cv

    #         rendered_color = rendered_results['image'] # torch.Size([1, V, 3, 224, 832])
    #         rendered_depth = rendered_results['depth'] # torch.Size([1, V, 1, 224, 832])
    #         rendered_alpha = rendered_results['alpha'] # torch.Size([1, V, 1, 224, 832])
    #         rendered_depth = rendered_depth.squeeze(2)
    #         rendered_alpha = rendered_alpha.squeeze(2)
            
    #         rendered_color = torch.clamp(rendered_color,min=0,max=1.0)
    #         rendered_depth = torch.clamp(rendered_depth,min=0,max=150)

    #         rendered_rgb_list.append(rendered_color)
    #         rendered_depth_list.append(rendered_depth)
            

    #     rendered_rgb_final = torch.cat(rendered_rgb_list,dim=1)
    #     rendered_depth_final = torch.cat(rendered_depth_list,dim=1)
        
    #     preds = {"img":rendered_rgb_final,"depth":rendered_depth_final}
        
    #     return preds,bin_token_name

    def validation_complete_with_bin_tokens(self,
                                            batch,
                                            val_result_savedir,
                                            bin_token_list,
                                            saved_label=False,
                                            cfg=None
                                            ):
        
        input_batch_dict,output_batch_dict = self.prepare_data_complete(batch=batch)
        
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
        
        
        gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
        bs = gaussians_all.shape[0] # batch size is 2


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
            saved_bin_token_name =input_batch_dict['bin_token_name'][0][:-4]
            current_saved_bin_folder = os.path.join(val_result_savedir,saved_bin_token_name)
            os.makedirs(current_saved_bin_folder,exist_ok=True)
        

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
            
            
            if saved_label:
                
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
                

            
            
        return rendered_left_images_list,rendered_right_images_list,rendered_left_depth_list,rendered_right_depth_list, \
            left_psnr_list,left_ssim_list,right_psnr_list,right_ssim_list,left_depth_mae_list,left_depth_mse_list,right_depth_mae_list,right_depth_mse_list

    def forward_kitti360_videos(self,batch,cfg):
        bin_token_name = batch['bin_token']
        
        input_batch_dict,output_batch_dict = self.prepare_data_complete(batch=batch)
        
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
        
        
        gaussians_all = torch.cat([gaussians_cv, gaussians_volume], dim=1)
        bs = gaussians_all.shape[0] # batch size is 2
        

        '''Output C2W'''
        c2w_lf_left = output_batch_dict['output_list'][2]["output_c2ws"][:,0,:,:]
        c2w_lf_right = output_batch_dict['output_list'][2]["output_c2ws"][:,1,:,:]
        c2w_ff_left = output_batch_dict['output_list'][0]["output_c2ws"][:,0,:,:]
        c2w_ff_right = output_batch_dict['output_list'][0]["output_c2ws"][:,1,:,:]
        c2w_cf_left = output_batch_dict['output_list'][1]["output_c2ws"][:,0,:,:]
        c2w_cf_right = output_batch_dict['output_list'][1]["output_c2ws"][:,1,:,:]

        
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
        fovxs_interp =output_batch_dict['output_list'][0]["output_fovxs"][:, -2:-1].repeat(1, num_frames_all)
        fovys_interp =output_batch_dict['output_list'][0]["output_fovys"][:, -2:-1].repeat(1, num_frames_all)
        
        
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

        # cur_bs,cur_view,cur_h,cur_w = pred_depths_list[0].shape        
        # pred_depths_for_concat0 = pred_depths_list[0].reshape(cur_bs,cur_view*cur_h*cur_w,1)
        # pred_depths_for_concat1 = pred_depths_list[1].reshape(cur_bs,cur_view*cur_h*cur_w,1)
        # pred_depths_for_concat2 = pred_depths_list[2].reshape(cur_bs,cur_view*cur_h*cur_w,1)
        # g0_aug = torch.cat((g0,pred_depths_for_concat0),dim=-1)
        # g1_aug = torch.cat((g1_trans,pred_depths_for_concat1),dim=-1)
        # g2_aug = torch.cat((g2_trans,pred_depths_for_concat2),dim=-1)
        # fused_gaussain0 = fuse_gaussians_by_voxel_with_depth_scatter_batched(
        #                 gaussians1_b=g0_aug,
        #                 gaussians2_b=g1_aug,
        #                 point_cloud_range=cfg.point_cloud_range,
        #                 voxel_size=0.5
        #             )
        # fused_gaussain1 = fuse_gaussians_by_voxel_with_depth_scatter_batched(
        #                 gaussians1_b=fused_gaussain0,
        #                 gaussians2_b=g2_aug,
        #                 point_cloud_range=cfg.point_cloud_range,
        #                 voxel_size=0.5
        #             )
        
        
       
        
        # fused_padded = fuse_gaussians_by_voxel_with_depth_scatter_batched(gaussians1_b=g0,
        #                                                    gaussians2_b=g2_trans)

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

def get_mean(list):
    return sum(list)*1.0/len(list)


def saved_into_json(data_dict,path):
    import json
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)

if __name__=="__main__":
    pass