from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from typing import Literal
import functools
import moviepy.editor as mpy
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
import numpy as np
import os
import skimage.io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet
from jaxtyping import Float
from torch import Tensor

from .encoder2.epipolar.my_epipolar_transformer import EpipolarTransformer
from .encoder2.epipolar.depth_predictor_monocular import DepthPredictorMonocular
from .encoder2.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg

from .decoder2.my_decoder_cuda import DecoderSplattingCUDA
from .geometry.projection import sample_image_grid

# Tools Here
from .metrics import compute_psnr_ssim,compute_depth_mae_mse,convert_depth_to_disp,kitti_colormap,save_dict_to_json
from .rgb_loss import LPIPS
from .utils import maybe_resize
from .utils import interpolate_extrinsics

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]

@dataclass
class BackboneDinoCfg:
    name: str = "dino"
    model: str = "dino_vitb8"
    d_out: int = 512
    
@dataclass
class BackboneResnetCfg:
    name: str = "resnet"  # 添加类型注解
    model: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"
    ] = "dino_resnet50"
    num_layers: int = 4
    use_first_pool: bool = False
    d_out: int = 512

@dataclass
class EncoderEpipolarCfg:
    name: Literal["epipolar"]
    d_feature: int
    num_monocular_samples: int
    num_surfaces: int
    predict_opacity: bool

# backbone resnet
class BackboneResnet(nn.Module):
    model: ResNet
    def __init__(self, cfg: BackboneResnetCfg, d_in: int) -> None:
        super().__init__()  # nn.Module 不接受参数
        self.cfg = cfg

        assert d_in == 3

        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )

        if cfg.model == "dino_resnet50":
            self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        else:
            self.model = getattr(torchvision.models, cfg.model)(norm_layer=norm_layer)

        # Set up projections
        self.projections = nn.ModuleDict({})
        for index in range(1, cfg.num_layers):
            key = f"layer{index}"
            block = getattr(self.model, key)
            conv_index = 1
            try:
                while True:
                    d_layer_out = getattr(block[-1], f"conv{conv_index}").out_channels
                    conv_index += 1
            except AttributeError:
                pass
            self.projections[key] = nn.Conv2d(d_layer_out, cfg.d_out, 1)

        # Add a projection for the first layer.
        self.projections["layer0"] = nn.Conv2d(
            self.model.conv1.out_channels, cfg.d_out, 1
        )

    def forward(
        self,
        image,
    ):
        # Merge the batch dimensions.
        b, v, _, h, w = image.shape
        x = rearrange(image, "b v c h w -> (b v) c h w")

        # Run the images through the resnet.
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        features = [self.projections["layer0"](x)]

        # Propagate the input through the resnet's layers.
        for index in range(1, self.cfg.num_layers):
            key = f"layer{index}"
            if index == 0 and self.cfg.use_first_pool:
                x = self.model.maxpool(x)
            x = getattr(self.model, key)(x)
            features.append(self.projections[key](x))

        # Upscale the features.
        features = [
            F.interpolate(f, (h, w), mode="bilinear", align_corners=True)
            for f in features
        ]
        features = torch.stack(features).sum(dim=0)

        # Separate batch dimensions.
        return rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

# backbone dino con
class BackboneDino(nn.Module):
    def __init__(self, d_in=3):
        super().__init__()
        
        # dino configuration.
        dino_cfg = BackboneDinoCfg()  # 使用默认值
        self.cfg = dino_cfg  # 保存配置以便访问 patch_size
        self.dino = torch.hub.load("facebookresearch/dino:main", dino_cfg.model)  # 传入字符串
         
        self.d_out = dino_cfg.d_out
         
        # 方式1: 使用默认值（推荐）
        resnet_cfg = BackboneResnetCfg()
        self.resnet_backbone = BackboneResnet(resnet_cfg, d_in)

        self.global_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, dino_cfg.d_out),
        )
        self.local_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, dino_cfg.d_out),
        )
        
    @property
    def patch_size(self) -> int:
        """从模型名称中提取 patch size，例如 'dino_vitb8' -> 8"""
        return int("".join(filter(str.isdigit, self.cfg.model)))

    def forward(self, image):
        resnet_features = self.resnet_backbone(image)

        # Compute features from the DINO-pretrained ViT.
        b, v, _, h, w = image.shape
        
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        tokens = rearrange(image, "b v c h w -> (b v) c h w")
        tokens = self.dino.get_intermediate_layers(tokens)[0]
        global_token = self.global_token_mlp(tokens[:, 0])
        local_tokens = self.local_token_mlp(tokens[:, 1:])

        # Repeat the global token to match the image shape.
        global_token = repeat(global_token, "(b v) c -> b v c h w", b=b, v=v, h=h, w=w)

        # Repeat the local tokens to match the image shape.
        local_tokens = repeat(
            local_tokens,
            "(b v) (h w) c -> b v c (h hps) (w wps)",
            b=b,
            v=v,
            h=h // self.patch_size,
            hps=self.patch_size,
            w=w // self.patch_size,
            wps=self.patch_size,
        )

        return resnet_features + local_tokens + global_token

class GaussainEncoder(nn.Module):
    def __init__(self,backbone,cfg=None):
        super().__init__()
        
        d_feature = 128

        # backbone configuration.
        self.backbone = backbone
        # backbone projection.
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out,d_feature),
        )

        self.epipolar_transformer = EpipolarTransformer(
            num_context_views=2,
            num_samples=32,
        )

        num_monocular_samples = 32
        num_surfaces = 1
        use_transmittance = False
        
        self.depth_predictor = DepthPredictorMonocular(
            d_feature,
            num_monocular_samples,
            num_surfaces,
            use_transmittance,
        )
        
        gaussian_adapter_cfg = GaussianAdapterCfg(
            gaussian_scale_min=0.5,
            gaussian_scale_max=15.0,
            sh_degree=4,
        )
        self.gaussian_adapter = GaussianAdapter(gaussian_adapter_cfg)

        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                d_feature,
                num_surfaces * (2 + self.gaussian_adapter.d_in),
            ),
        )

        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, d_feature, 7, 1, 3),
            nn.ReLU(),
        )

    def map_pdf_to_opacity(
        self,
        pdf,
        global_step):
        # Figure out the exponent.
        initial = 0.0
        final = 0.0
        warm_up = 1
        
        x = initial + min(global_step / warm_up, 1) * (final - initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    
    def forward(self,
                image,
                extrinsics,
                intrinsics,
                near,
                far,
                global_step=0):
        
        deterministic = False
        gaussians_per_pixel = 3
        num_surfaces = 1
        
        device = image.device
        b, v, _, h, w = image.shape
        # Encode the context images.
        features = self.backbone(image)
        features = rearrange(features, "b v c h w -> b v h w c")

        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w")
        
        features, sampling = self.epipolar_transformer(
                features,
                extrinsics,
                intrinsics,
                near,
                far,
            )

        # Add the high-resolution skip connection.
        skip = rearrange(image, "b v c h w -> (b v) c h w")
        skip = self.high_resolution_skip(skip)
        features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)

        # Sample depths from the resulting features.
        features = rearrange(features, "b v c h w -> b v (h w) c")
        
        depths, densities = self.depth_predictor.forward(
            features,
            near,
            far,
            deterministic,
            1 if deterministic else gaussians_per_pixel,
        )

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            self.to_gaussians(features),
            "... (srf c) -> ... srf c",
            srf=num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        
        gpp = gaussians_per_pixel

        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
            (h, w),
        )


        opacity_multiplier = 1
        

        return Gaussians(
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



class PixelSplatModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.gaussain_encoder = GaussainEncoder(backbone=BackboneDino(d_in=3))
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
    
                    
    def forward(self, batch, mode="train", iter=0, cfg=None):

        input_batch_dict,output_batch_dict = self.prepare_input_batch_data(batch=batch)

        return_depth = cfg.return_depth
        iter_end = cfg.max_train_steps 
        depth_max_value = cfg.max_depth # 1000
        depth_min_value = cfg.min_depth # 0.01    

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

        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/width
        intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/height
        
        
        # gaussain encode
        estimated_gausssains_raw = self.gaussain_encoder(input_images, 
                                          input_extrinsics,
                                          intrinsics, 
                                          near, 
                                          far, 
                                          global_step=iter)


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
    
    device = "cuda:0"
    global_step = 0
    
    # networks input configruations.
    from model_input import create_mvsplat_demo_input
    batch_example, context, target = create_mvsplat_demo_input(device="cuda:0",
                              dtype=torch.float32,
                              image_height=112,
                              image_width=544,
                              batch_size=1,
                              num_context_views=2,
                              num_target_views=6)
    
    
    pixel_splat_model = PixelSplatModel()
    pixel_splat_model.to(device)
    
    rendered_color,rendered_depth = pixel_splat_model(context["image"],
                     context["extrinsics"],
                     context["intrinsics"],
                     context["near"],
                     context["far"],
                     global_step=global_step)
    
    print(rendered_color.shape)
    print(rendered_depth.shape)
    quit()
    
    

    # backbone_dino = BackboneDino(d_in=3)    
    # gaussain_encoder = GaussainEncoder(backbone=backbone_dino)
    
    # gaussain_encoder.to(device)
    
    # est_gaussians = gaussain_encoder(context["image"],
    #                  context["extrinsics"],
    #                  context["intrinsics"],
    #                  context["near"],
    #                  context["far"],
    #                  global_step=global_step)
    
    # print(est_gaussians.means.shape)
    # print(est_gaussians.covariances.shape)
    # print(est_gaussians.harmonics.shape)
    # print(est_gaussians.opacities.shape)
    # quit()

    
