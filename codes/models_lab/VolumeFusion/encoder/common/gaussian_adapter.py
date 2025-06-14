from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ..geometry.projection import get_world_rays
from ..misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"] # 3D Means
    covariances: Float[Tensor, "*batch 3 3"] # Covariances
    scales: Float[Tensor, "*batch 3"]   # Scales
    rotations: Float[Tensor, "*batch 4"] # Rotations in World
    harmonics: Float[Tensor, "*batch 3 _"] # SH Coffients
    opacities: Float[Tensor, " *batch"]    # Opcaities


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg 

    def __init__(self,
                 gaussian_scale_min,
                 gaussian_scale_max,
                 sh_degree,
                 **kwargs
                        ):
        super().__init__()
        self.cfg = GaussianAdapterCfg(gaussian_scale_max=gaussian_scale_max,
                                      gaussian_scale_min=gaussian_scale_min,
                                      sh_degree=sh_degree
                                      )

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        # here the sh mask is a learnable
        '''
        sh_mask 是一个 buffer（不会被更新，但会保存在模型状态中），作用是初始化时
        保留低阶 SH（Spherical Harmonics）系数，衰减高阶项，
        避免一开始就产生复杂的 view-dependent 颜色影响。
        
        '''
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        '''
        
        SH 系数总共为(sh+1)^w ，这里将非 DC 分量设为较小值。 比如当 sh_degree = 2，则总共是 9 维，
        degree=1 对应索引 1~3, degree=2 对应 4~8，这些都乘以较小因子。
        # initalization SH Masks
        '''
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
    
    def forward_for_world(self,
                          opacities,
                          mean3D,
                          raw_gaussians,
                          scale_max,
                          eps: float = 1e-8,
                          ):
        
        opacities =opacities.unsqueeze(-1) #torch.Size([1, 192, 192, 16, 3, 1, 1])
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        opacities = opacities.clamp(min=0.001, max=0.999)  # 避免0或1导致渲染异常
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        scale_x = F.sigmoid(scales[..., :1]) * scale_max[0]
        scale_y = F.sigmoid(scales[..., 1:2]) * scale_max[1]
        scale_z = F.sigmoid(scales[..., 2:3]) *scale_max[2]

        scales = torch.cat([scale_x,scale_y,scale_z],dim=-1)
        
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh * self.sh_mask # torch.Size([1, 192, 192, 16, 3, 3, 9])
    
        covariances = build_covariance(scales, rotations)
        
        opacities = opacities.squeeze(-1).squeeze(-1)

        # print(mean3D.shape)
        # print(covariances.shape)
        # print(opacities.shape)
        # # print(sh.shape)
        # quit()
        # quit()
        

        return Gaussians(
            means=mean3D,
            covariances=covariances,
            harmonics=sh, # 将球谐系数从相机坐标系旋转到世界坐标系。
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        
        '''
        Scale:3维,
        rotation 四元数 4维,
        sh(球谐,3色通道 x sh维数)
        
        '''
        # print(extrinsics.shape) #(B,V,1,1,1,4,4)
        # print(intrinsics.shape) #(B,V,1,1,1,3,3)
        
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        # print(scales.shape) # torch.Size([1, 2, 186368, 1, 1, 3])
        # print(rotations.shape) # torch.Size([1, 2, 186368, 1, 1, 4])
        # print(sh.shape) # torch.Size([1, 2, 186368, 1, 1, 27])
        # quit()

        #  softplus 激活 + 截断（保证 scale 有效、非负）
        # softplus(x - 4) 让初始 scale 更小更平滑。
        # https://blog.csdn.net/hy592070616/article/details/120623303
        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        assert input_images is not None
        
        
        opacities = opacities.clamp(min=0.001, max=0.999)  # 避免0或1导致渲染异常

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [B, V, N, 1, 1, 3, 25]
        # reshape SH，乘以 mask（低阶系数保留，高阶压缩）
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # 如果有输入图像，就用 RGB 作为 SH 的第 0 阶初始化
        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        # 生成高斯协方差矩阵（以 scale 和 rotation 为输入）
        # Local covariances
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        # to world covariances
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        # 生成每个像素或射线的 3D 起点和方向
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        
        # get the means
        means = origins + directions * depths[..., None]
        
        

        
        
        
        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]), # 将球谐系数从相机坐标系旋转到世界坐标系。
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh
    
    @property
    def d_in_for_volume(self) ->int:
        return 3*self.d_sh + 3 + 3+ 4+ 1


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
