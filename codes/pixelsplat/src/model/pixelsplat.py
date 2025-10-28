from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from typing import Literal
import functools
import moviepy.editor as mpy
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet
from jaxtyping import Float
from torch import Tensor

from encoder2.epipolar.my_epipolar_transformer import EpipolarTransformer
from encoder2.epipolar.depth_predictor_monocular import DepthPredictorMonocular
from encoder2.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg

from decoder2.my_decoder_cuda import DecoderSplattingCUDA

from geometry.projection import sample_image_grid


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


class PixelSplatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussain_encoder = GaussainEncoder(backbone=BackboneDino(d_in=3))
        self.decoder = DecoderSplattingCUDA()
        
    def forward(self, image, extrinsics, intrinsics, near, far, global_step=0):
        height,width = image.shape[-2:]

        gaussians = self.gaussain_encoder(image, extrinsics, intrinsics, near, far, global_step)

        rendered_color,rendered_depth = self.decoder.forward(gaussians, 
                                                             extrinsics, 
                                                             intrinsics, 
                                                             near, 
                                                             far, 
                                                             (height,width),                                                          
                                                             global_step)
        return rendered_color,rendered_depth


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

    
