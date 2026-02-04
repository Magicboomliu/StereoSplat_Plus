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
from .rgb_loss import LPIPS
from .utils import maybe_resize
from .utils import interpolate_extrinsics

from .depth_error_vis import disp_error_img,depths_to_colors
from .metrics import compute_psnr_ssim,compute_depth_mae_mse,convert_depth_to_disp,kitti_colormap,save_dict_to_json,compute_stereo_psnr_ssim,compute_all_stereo_psnr_ssim
import math
import os.path as osp
from tqdm import tqdm
import random
import copy
import lpips
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn


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


    # """
    # Computes MAE and MSE between predicted and GT depth maps, with optional valid range filtering.

    # Args:
    #     depth_pred (torch.Tensor): Predicted depth, shape [B, V, H, W]
    #     depth_gt (torch.Tensor): Ground truth depth, shape [B, V, H, W]
    #     valid_min (float): minimum valid GT depth
    #     valid_max (float): maximum valid GT depth

    # Returns:
    #     mae (torch.Tensor): scalar mean absolute error
    #     mse (torch.Tensor): scalar mean squared error
    # """
    # assert depth_pred.shape == depth_gt.shape, "Shape mismatch between prediction and GT"

    # # Create valid mask (only use pixels with valid GT depth)
    # valid_mask = (depth_gt > valid_min) & (depth_gt < valid_max)

    # # Compute errors
    # abs_error = torch.abs(depth_pred - depth_gt)
    # sq_error = (depth_pred - depth_gt) ** 2

    # # Apply mask
    # abs_error = abs_error[valid_mask]
    # sq_error = sq_error[valid_mask]

    # # Final metrics
    # mae = abs_error.mean()
    # mse = sq_error.mean()

    # return mae, mse

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
                                          global_step=126000)


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
                    
            current_batch_size = output_intrinsics.shape[0]
            current_nums_of_views = output_intrinsics.shape[1]
            near = torch.full((current_batch_size, current_nums_of_views), depth_min_value, dtype=self.dtype, device=self.device)
            far = torch.full((current_batch_size, current_nums_of_views), depth_max_value, dtype=self.dtype, device=self.device)
            
            
            rendered_color,rendered_depth = self.decoder.forward(
                    estimated_gausssains_raw,
                    current_c2w_interp,
                    output_intrinsics,
                    near,
                    far,
                    (height, width),
                    depth_mode='depth',
                )
            
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
                                          global_step=126000)


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


    def bev_video_kitti360(self,batch,cfg=None,
                           rescale_h=3.0,rescale_w=1.0):
        
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
        
        current_resolution = [input_images.shape[-2], input_images.shape[-1]]

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
                                          global_step=126000)

        # rendered extrinsics and intrinsics here 
        output_extrinsics = output_batch_dict['output_c2ws']
        output_intrinsics = intrinsics.clone()
        output_intrinsics = output_intrinsics[:,0:1,:,:].repeat(1,output_extrinsics.shape[1],1,1)
        output_near = torch.full((current_batch_size, output_extrinsics.shape[1]), depth_min_value, dtype=self.dtype, device=self.device)
        output_far = torch.full((current_batch_size, output_extrinsics.shape[1]), depth_max_value, dtype=self.dtype, device=self.device)
        
        

        rendered_resolution = [int(current_resolution[0]*rescale_h), int(current_resolution[1]*rescale_w)]
        
        
        render_c2w = output_extrinsics.clone()
        
        num_of_views_all = render_c2w.shape[1]
        rest_of_views_all = num_of_views_all - 2
        rest_besides_first_c2w = render_c2w[:,:-2,:,:]
        
        half_rest_of_views_all = rest_of_views_all // 2
        center_view_c2w_left_index = half_rest_of_views_all -2
        last_view_c2w_left_index = half_rest_of_views_all -1
        
        center_view_c2w_right_index = rest_of_views_all - 2
        last_view_c2w_right_index = rest_of_views_all - 1
        

        
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
            

            recovered_intrinsic = output_intrinsics[0,0,:,:].clone()
            recovered_intrinsic[0] = recovered_intrinsic[0]*1.0 * width
            recovered_intrinsic[1] = recovered_intrinsic[1]*1.0 *height
            
            recovered_intrinsic[0,2] = recovered_intrinsic[0,2] * rescale_w
            recovered_intrinsic[1,2] = recovered_intrinsic[1,2] * rescale_h
            
            recovered_intrinsics = recovered_intrinsic.unsqueeze(0).unsqueeze(0).repeat(1,rendered_bev_novel_views_c2w.shape[1],1,1)
            recovered_intrinsics[:, :, 0] = recovered_intrinsics[:, :, 0]*1.0/rendered_resolution[1]
            recovered_intrinsics[:, :, 1] = recovered_intrinsics[:, :, 1]*1.0/rendered_resolution[0]
            
            z_far_batch = output_far[:,0:1].repeat(1,rendered_bev_novel_views_c2w.shape[1])
            z_near_batch = output_near[:,0:1].repeat(1,rendered_bev_novel_views_c2w.shape[1])
        

            rendered_color,rendered_depth = self.decoder.forward(
                    estimated_gausssains_raw,
                    rendered_bev_novel_views_c2w,
                    recovered_intrinsics,
                    z_near_batch,
                    z_far_batch,
                    (rendered_resolution[0], rendered_resolution[1]),
                    depth_mode='depth',
                )
            
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
        
        current_resolution = [input_images.shape[-2], input_images.shape[-1]]

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
                                          global_step=126000)


        # rendered extrinsics and intrinsics here 
        output_extrinsics = output_batch_dict['output_c2ws']
        output_intrinsics = intrinsics.clone()
        output_intrinsics = output_intrinsics[:,0:1,:,:].repeat(1,output_extrinsics.shape[1],1,1)
        output_near = torch.full((current_batch_size, output_extrinsics.shape[1]), depth_min_value, dtype=self.dtype, device=self.device)
        output_far = torch.full((current_batch_size, output_extrinsics.shape[1]), depth_max_value, dtype=self.dtype, device=self.device)
        
        

        rendered_resolution = [int(current_resolution[0]*rescale_h), int(current_resolution[1]*rescale_w)]
        
        
        render_c2w = output_extrinsics.clone()
        
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
        
        recovered_intrinsic = output_intrinsics[0,0,:,:].clone()
        recovered_intrinsic[0] = recovered_intrinsic[0]*1.0 * width
        recovered_intrinsic[1] = recovered_intrinsic[1]*1.0 *height
        
        recovered_intrinsic[0,2] = recovered_intrinsic[0,2] * rescale_w
        recovered_intrinsic[1,2] = recovered_intrinsic[1,2] * rescale_h
        
        recovered_intrinsics = recovered_intrinsic.unsqueeze(0).unsqueeze(0).repeat(1,rendered_bev_novel_views_c2w.shape[1],1,1)
        recovered_intrinsics[:, :, 0] = recovered_intrinsics[:, :, 0]*1.0/rendered_resolution[1]
        recovered_intrinsics[:, :, 1] = recovered_intrinsics[:, :, 1]*1.0/rendered_resolution[0]
        
        z_far_batch = output_far[:,0:1].repeat(1,rendered_bev_novel_views_c2w.shape[1])
        z_near_batch = output_near[:,0:1].repeat(1,rendered_bev_novel_views_c2w.shape[1])
        
        
        rendered_color,rendered_depth = self.decoder.forward(
                estimated_gausssains_raw,
                rendered_bev_novel_views_c2w,
                recovered_intrinsics,
                z_near_batch,
                z_far_batch,
                (rendered_resolution[0], rendered_resolution[1]),
                depth_mode='depth',
            )
        
        rendered_color = torch.clamp(rendered_color,min=0,max=1.0)
        rendered_depth = torch.clamp(rendered_depth,min=0,max=150)


        rendered_color_fuse = rendered_color 
        rendered_depth_fuse = rendered_depth
        

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

    
