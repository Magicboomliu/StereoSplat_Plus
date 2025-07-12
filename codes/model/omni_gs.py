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
from .mertics import compute_psnr_ssim,kitti_colormap,convert_depth_to_disp
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
    
    def validation_complete_with_bin_tokens(self,
                                            batch,
                                            val_result_savedir,
                                            bin_token_list,
                                            saved_label=False
                                            ):
        
        data_dict = self.prepare_data_complete(batch=batch)
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
        output_info_list_all = data_dict['output_list']
        
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
            saved_bin_token_name =data_dict["bin_token"][0][:-4]
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


    def forward_kitti360_videos(self,batch):
        
        data_dict = self.prepare_data_complete(batch=batch)
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
        

    
        '''Output C2W'''
        c2w_lf_left = data_dict['output_list'][2]["output_c2ws"][:,0,:,:]
        c2w_lf_right = data_dict['output_list'][2]["output_c2ws"][:,1,:,:]
        c2w_ff_left = data_dict['output_list'][0]["output_c2ws"][:,0,:,:]
        c2w_ff_right = data_dict['output_list'][0]["output_c2ws"][:,1,:,:]
        c2w_cf_left = data_dict['output_list'][1]["output_c2ws"][:,0,:,:]
        c2w_cf_right = data_dict['output_list'][1]["output_c2ws"][:,1,:,:]

        
        
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
        fovxs_interp = data_dict['output_list'][0]["output_fovxs"][:, -2:-1].repeat(1, num_frames_all)
        fovys_interp = data_dict['output_list'][0]["output_fovys"][:, -2:-1].repeat(1, num_frames_all)
        
        
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

