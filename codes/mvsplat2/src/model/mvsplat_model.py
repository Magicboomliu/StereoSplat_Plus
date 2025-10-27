import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from collections import OrderedDict
from einops import rearrange
from .geometry.projection import sample_image_grid
from .model_types import Gaussians


import moviepy.editor as mpy
import wandb
from einops import pack, rearrange, repeat, einsum
from jaxtyping import Float
from tqdm import tqdm
import json
import os
import math
from PIL import Image
import torchvision.transforms as T
import skimage.io


from .encoder2.backbone import BackboneMultiview
from .encoder2.costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .encoder2.common.gaussian_adapter import GaussianAdapter,GaussianAdapterCfg
from .decoder2.my_decoder_cuda import DecoderSplattingCUDA

# Tools Here
from .metrics import compute_psnr_ssim,compute_depth_mae_mse,convert_depth_to_disp,kitti_colormap,save_dict_to_json
from .rgb_loss import LPIPS
from .utils import maybe_resize
from .utils import interpolate_extrinsics

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

class GaussainEncoder(nn.Module):
    def __init__(self, 
                 cfg:dict,
                 ):
        super().__init__()
        self.cfg = cfg
        
        # build the image extractor backbone here
        self.backbone = BackboneMultiview(
                feature_channels=cfg.feature_channels,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_cost_volume,
                use_epipolar_trans=cfg.use_epipolar_trans)

        ckpt_path = cfg.unimatch_weights_path
        
        if ckpt_path is not None or ckpt_path != "None":
            unimatch_pretrained_ckpt = torch.load(ckpt_path)["model"]
            updated_state_dict = OrderedDict(
                {
                    k: v
                    for k, v in unimatch_pretrained_ckpt.items()
                    if k in self.backbone.state_dict()
                }
            )
            is_strict_loading = cfg.wo_backbone_cross_attn
            self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)
            print("Loaded the Unimatch Pretrained Model Successfully!")
                

        # Gaussain Adpater
        gs_adapter_cfg = GaussianAdapterCfg(
            gaussian_scale_min=cfg.gs_adapter_cfg.gaussian_scale_min,
            gaussian_scale_max=cfg.gs_adapter_cfg.gaussian_scale_max,
            sh_degree=cfg.gs_adapter_cfg.sh_degree
        )
        
        self.gaussian_adapter = GaussianAdapter(gs_adapter_cfg)
        
        
        # Depth Predictor
        self.depth_predictor = DepthPredictorMultiView(
                    feature_channels=cfg.d_feature,
                    upscale_factor=cfg.downscale_factor,
                    num_depth_candidates=cfg.num_depth_candidates,
                    costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
                    costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
                    costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
                    gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
                    gaussians_per_pixel=cfg.gaussians_per_pixel,
                    num_views=cfg.num_context_views,
                    depth_unet_feat_dim=cfg.depth_unet_feat_dim,
                    depth_unet_attn_res=cfg.depth_unet_attn_res,
                    depth_unet_channel_mult=cfg.depth_unet_channel_mult,
                    wo_depth_refine=cfg.wo_depth_refine,
                    wo_cost_volume=cfg.wo_cost_volume,
                    wo_cost_volume_refine=cfg.wo_cost_volume_refine
        )


    def map_pdf_to_opacity(
        self,
        pdf,
        global_step: int,
        cfg=None,
    ):
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))


    def forward(self, 
                input_images,
                intrinsics,
                extrinsics,
                near,
                far,
                input_nn_matrix,
                input_bin_tokens_name,
                global_step=0,
                cfg=None):
        
        device = input_images.device
        b, v, _, h, w = input_images.shape
        
        # inference the backbone here
        trans_features, cnn_features = self.backbone(
                input_images,
                attn_splits=cfg.multiview_trans_attn_split,
                return_cnn_features=True,
                epipolar_kwargs=None,
            )

        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(input_images, "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = input_bin_tokens_name

        gpp = cfg.gaussians_per_pixel
        
        deterministic=False
        
        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
        )

        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities,global_step=global_step,cfg=cfg) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1
        final_gaussains = Gaussians(
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    opacity_multiplier * gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
    
        return final_gaussains



        
        

class MVSplatModel(nn.Module):
    def __init__(self,
                 encoder: nn.Module):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = DecoderSplattingCUDA()

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
        depth_max_value = cfg.max_depth # 1000
        depth_min_value = cfg.min_depth # 0.01    
        
        
        # inputs information
        input_images = input_batch_dict['imgs'] # [B,V,3,H,W]
        intrinsics = input_batch_dict['intrinsics'] # [B,V,3,3]
        input_extrinsics = input_batch_dict['extrinsics'] # [B,V,4,4]
        input_nn_matrix = input_batch_dict['nn_matrix'] #[B,V,K]
        bs = input_images.shape[0]
        input_pseudo_depth = input_batch_dict['pseudo_depths']
        input_sparse_gt_depth = input_batch_dict['sparse_depths']
        
        input_bin_tokens_name = input_batch_dict['bin_token_name']


        mask = input_sparse_gt_depth > 0
        mask = mask.float()
        input_nn_matrix = input_nn_matrix.long()
        
        
        current_batch_size = input_images.shape[0]
        current_nums_of_views = input_images.shape[1]
        
        near = torch.full((current_batch_size, current_nums_of_views), depth_min_value, dtype=self.dtype, device=self.device)
        far = torch.full((current_batch_size, current_nums_of_views), depth_max_value, dtype=self.dtype, device=self.device)
        
        height, width = input_images.shape[3:]

        # intrinsics[:,:,0,2] = intrinsics[:,:,0,2] + 13
        # intrinsics[:,:,1,2] = intrinsics[:,:,1,2] - 30
        intrinsics = intrinsics.clone()
        # Normalized the instrinsics -----> Maybe not neccssary
        intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/width
        intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/height
        
        # estimate the raw gaussains here
        estimated_gausssains_raw = self.encoder(input_images,
                     intrinsics,
                     input_extrinsics,
                     near,
                     far,
                     input_nn_matrix,
                     input_bin_tokens_name,
                     global_step=iter,
                     cfg=cfg)
        
        
        # rendered extrinsics and intrinsics here 
        output_extrinsics = output_batch_dict['output_c2ws']
        output_intrinsics = intrinsics.clone()
        output_intrinsics = output_intrinsics[:,0:1,:,:].repeat(1,output_extrinsics.shape[1],1,1)

        output_near = torch.full((current_batch_size, output_extrinsics.shape[1]), depth_min_value, dtype=self.dtype, device=self.device)
        output_far = torch.full((current_batch_size, output_extrinsics.shape[1]), depth_max_value, dtype=self.dtype, device=self.device)
        
        
        rendered_color,rendered_depth = self.decoder.forward(
                estimated_gausssains_raw,
                output_extrinsics,
                output_intrinsics,
                output_near,
                output_far,
                (height, width),
                depth_mode='depth',
            )
        
        rendered_color = torch.clamp(rendered_color,min=0,max=1.0)
        rendered_depth = torch.clamp(rendered_depth,min=0,max=150)
        
        if mode=='train' or mode=='val':
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
             
            if mode=='train':
                return loss, loss_terms,rendered_color,rendered_depth,estimated_gausssains_raw
            
            elif mode=='val':
                return loss, loss_terms,rendered_color,rendered_depth,estimated_gausssains_raw,input_sparse_gt_depth,output_rgb,sparse_depth_gt,input_images
            
        elif mode=='test':
            return rendered_color,rendered_depth,estimated_gausssains_raw

        else:
            raise NotImplementedError


    def validation_step(self, batch, 
                        val_result_savedir,
                        cfg=None):
        
        bin_token_name = batch['bin_token'][0][:-4]
        
        
        with torch.no_grad():
            loss, loss_terms,rendered_color,rendered_depth,estimated_gausssains_raw,input_sparse_gt_depth,output_rgb,sparse_depth_gt,input_images = self.forward(batch,mode='val',
                                                                                cfg=cfg)

        batch_data_for_eval = {
            "output_gt_rgb": output_rgb,
            "output_gt_sparse_depth": sparse_depth_gt,
            "input_images": input_images,
            "input_gt_sparse_gt": input_sparse_gt_depth,
            "rendered_rgb": rendered_color,
            "rendered_depth": rendered_depth,
            "estimated_raw_gs": estimated_gausssains_raw,
            "bin_token_name": bin_token_name
        }
        
        # rendered RGBs
        # rendered Depths
        # GT RGBs
        # GT Depths
        # Estimated Depths
        
        output_rgb_meter_dict,output_depth_meter_dict = self.save_val_results(batch_data_for_eval,val_result_savedir,cfg=cfg)
        
        return output_rgb_meter_dict,output_depth_meter_dict
        
        
        
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


        return output_rgb_meter_dict,output_depth_meter_dict


if __name__ == "__main__":
    
    
    # create the demo input for debugging,
    from demo_input import create_mvsplat_demo_input
    batch_example, context, target = create_mvsplat_demo_input(image_height=112, 
                                                               image_width=544,
                                                               num_context_views=2,
                                                               num_target_views=6,
                                                               batch_size=1,
                                                               device='cuda:0')
    '''   MVSplat Encoder Configuration .....  '''
    # hyperparamters for the backbone
    feature_channels = 128
    downscale_factor = 4
    multiview_trans_attn_split = 2
    no_cross_attn = False
    use_epipolar_trans = False
    unimatch_weights_path = "/home/zliu/Project2025/mvsplat/checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth"
    wo_depth_refine = False         # Table 3: base
    wo_cost_volume = False          # Table 3: w/o cost volume
    wo_backbone_cross_attn= False  # Table 3: w/o cross-view attention
    wo_cost_volume_refine= False   # Table 3: w/o U-Net
    use_epipolar_trans=False
    
    
    # hyperparamters for the depth predictor.
    gaussians_per_pixel=1
    near = 0.1    # 0.1米
    far = 1000.0  # 1000米
    num_context_views=2
    depth_unet_feat_dim=32
    depth_unet_attn_res= [16]
    depth_unet_channel_mult= [1, 1, 1]
    shim_patch_size= 4

    # futher build the depth predictor 
    d_feature=128
    num_depth_candidates=192
    costvolume_unet_feat_dim=128
    costvolume_unet_channel_mult= [1,1,1]
    costvolume_unet_attn_res =[]
    num_surfaces = 1
    
    # infernece config
    global_step=100
    
    # get the backbone
    from encoder2.backbone import BackboneMultiview
    # get the depth predictor
    from encoder2.costvolume.depth_predictor_multiview import DepthPredictorMultiView
    # get the gaussain aadaper
    from encoder2.common.gaussian_adapter import GaussianAdapter,GaussianAdapterCfg
    
    # multiview transformer backbone
    backbone = BackboneMultiview(
            feature_channels=128,
            downscale_factor=4,
            no_cross_attn=wo_cost_volume,
            use_epipolar_trans=use_epipolar_trans)
    backbone.to(device='cuda:0')
    
    ckpt_path = unimatch_weights_path
    unimatch_pretrained_model = torch.load(ckpt_path)["model"]
    updated_state_dict = OrderedDict(
        {
            k: v
            for k, v in unimatch_pretrained_model.items()
            if k in backbone.state_dict()
        }
    )
    is_strict_loading = wo_backbone_cross_attn
    backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)


    # build the gaussain adapter
    gs_adapter_cfg = GaussianAdapterCfg(
        gaussian_scale_min=0.5,
        gaussian_scale_max=15.0,
        sh_degree=4
    )
    gaussian_adapter = GaussianAdapter(gs_adapter_cfg)
    gaussian_adapter.to(device='cuda:0')
    # debug here
    class OpacityMappingCfg:
        initial=0.0
        final=0.0
        warm_up=1.0
    opacity_mapping_cfg = OpacityMappingCfg()

    def map_pdf_to_opacity(
        pdf,
        global_step: int,
    ):
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = opacity_mapping_cfg
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))


    # define the depth predictor here
    depth_predictor = DepthPredictorMultiView(
        feature_channels=d_feature,
        upscale_factor=downscale_factor,
        num_depth_candidates=num_depth_candidates,
        costvolume_unet_feat_dim=costvolume_unet_feat_dim,
        costvolume_unet_channel_mult=tuple(costvolume_unet_channel_mult),
        costvolume_unet_attn_res=tuple(costvolume_unet_attn_res),
        gaussian_raw_channels=num_surfaces * (gaussian_adapter.d_in + 2),
        gaussians_per_pixel=gaussians_per_pixel,
        num_views=num_context_views,
        depth_unet_feat_dim=depth_unet_feat_dim,
        depth_unet_attn_res=depth_unet_attn_res,
        depth_unet_channel_mult=depth_unet_channel_mult,
        wo_depth_refine=wo_depth_refine,
        wo_cost_volume=wo_cost_volume,
        wo_cost_volume_refine=wo_cost_volume_refine,
    )
    depth_predictor.to(device='cuda:0')
    # inference here
    device = context["image"].device
    b, v, _, h, w = context["image"].shape

    # inference the backbone here
    trans_features, cnn_features = backbone(
            context["image"],
            attn_splits=multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=None,
        )

    in_feats = trans_features
    extra_info = {}
    extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
    extra_info["scene_names"] = batch_example['scene']

    gpp = gaussians_per_pixel
    
    deterministic=False
    
    depths, densities, raw_gaussians = depth_predictor(
        in_feats,
        context["intrinsics"],
        context["extrinsics"],
        context["near"],
        context["far"],
        gaussians_per_pixel=gpp,
        deterministic=deterministic,
        extra_info=extra_info,
        cnn_features=cnn_features,
    )
    
    # Convert the features and depths into Gaussians.
    xy_ray, _ = sample_image_grid((h, w), device)
    xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
    gaussians = rearrange(
        raw_gaussians,
        "... (srf c) -> ... srf c",
        srf=num_surfaces,
    )
    offset_xy = gaussians[..., :2].sigmoid()
    pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
    xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
    gpp = gaussians_per_pixel
    gaussians = gaussian_adapter.forward(
        rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
        rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
        rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
        depths,
        map_pdf_to_opacity(densities,global_step) / gpp,
        rearrange(
            gaussians[..., 2:],
            "b v r srf c -> b v r srf () c",
        ),
        (h, w),
    )

    # Optionally apply a per-pixel opacity.
    opacity_multiplier = 1
    final_gaussains = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

    

    
    
    '''  MVSplat Decoder and Renderer Configuration .....  '''
    
    from decoder2.my_decoder_cuda import DecoderSplattingCUDA
    
    gs_decoder = DecoderSplattingCUDA()
    
    gs_decoder.to(device='cuda:0')
    

    color,depth = gs_decoder.forward(
            final_gaussains,
            batch_example["target"]["extrinsics"],
            batch_example["target"]["intrinsics"],
            batch_example["target"]["near"],
            batch_example["target"]["far"],
            (h, w),
            depth_mode='depth',
        )

    print(color.shape)
    print(depth.shape)
    
    
    # Compute the loss function here # 