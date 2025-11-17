import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)
from .utils.ops import get_cam_info_gaussian
from .utils.typing import *


C0 = 0.28209479177387814

def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

class Depth2Normal(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delzdelxkernel = torch.tensor(
            [
                [0.00000, 0.00000, 0.00000],
                [-1.00000, 0.00000, 1.00000],
                [0.00000, 0.00000, 0.00000],
            ]
        )
        self.delzdelykernel = torch.tensor(
            [
                [0.00000, -1.00000, 0.00000],
                [0.00000, 0.00000, 0.00000],
                [0.0000, 1.00000, 0.00000],
            ]
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        B, C, H, W = x.shape
        delzdelxkernel = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
        delzdelx = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelxkernel, padding=1
        ).reshape(B, C, H, W)
        delzdelykernel = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
        delzdely = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelykernel, padding=1
        ).reshape(B, C, H, W)
        normal = -torch.cross(delzdelx, delzdely, dim=1)
        return normal


class GaussianRenderer:
    def __init__(
        self, 
        device,
        resolution: list = [512, 512],
        znear: float = 0.1,
        zfar: float = 100.0, 
        renderer_type: str = "vanilla", # only support "vanilla"
        **kwargs,
    ):  
        self.renderer_type = renderer_type

        self.resolution = resolution
        self.znear = znear
        self.zfar = zfar
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # convert the depth into surface normal
        self.normal_module = Depth2Normal().to(device)

        self.setup_functions()


    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    '''
    Input a couple pf 3D gaussains, Gaussians blobs
    return: RGB + Alpha + Depth
    '''
    def render(
        self, 
        gaussians: Float[Tensor, "B N F"], # 一堆高斯 blob，[B, N, 14]，每个点14个属性（位置、颜色、透明度、旋转、尺度）
        c2w: Float[Tensor, "B V 4 4"], # c2w---> B, 6, 4,4, default is the openGL
        fovx: Float[Tensor, "B V"] = None, # B, 6 
        fovy: Float[Tensor, "B V"] = None,
        rays_o: Float[Tensor, "B V H W 3"] = None,
        rays_d: Float[Tensor, "B V H W 3"] = None,
        bg_color: Float[Tensor, "... 3"] = None, 
        scale_modifier: float = 1.,
    ):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        # at least one of fovx and fovy is not none
        # 要求至少要有一个 FOV，否则不知道怎么建相机视锥。
        assert fovx is not None or fovy is not None
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx

        device = gaussians.device
        B, V = c2w.shape[:2]

        # ---------------- sanitize fovx/fovy ---------------- #
        def safe_fov(tensor):
            tensor = torch.nan_to_num(tensor, nan=0.5, posinf=0.5, neginf=0.5)  # default 0.5 rad ≈ 57 deg
            return tensor.clamp(min=1e-3, max=math.pi - 1e-3)
        if fovx is None and fovy is None:
            raise ValueError("At least one of fovx or fovy must be provided.")
        if fovx is None:
            fovx = safe_fov(fovy.clone())
        if fovy is None:
            fovy = safe_fov(fovx.clone())
        fovx = safe_fov(fovx)
        fovy = safe_fov(fovy)

        # ---------------- sanitize c2w ---------------- #
        c2w = torch.nan_to_num(c2w, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------------- sanitize bg_color ---------------- #
        if bg_color is not None:
            bg_color = torch.nan_to_num(bg_color, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            bg_color = self.bg_color  # use default (usually [0, 0, 0])

        # ---------------- sanitize rays ---------------- #
        if rays_o is not None:
            rays_o = torch.nan_to_num(rays_o, nan=0.0, posinf=0.0, neginf=0.0)
        if rays_d is not None:
            rays_d = torch.nan_to_num(rays_d, nan=0.0, posinf=0.0, neginf=0.0)


        # loop of loop...
        # 为了最后收集每个 batch、每个相机的渲染结果
        images = []
        alphas = []
        depths = []
        
        # batch size
        for b in range(B):
            # 3D 坐标（means）
            means3D = gaussians[b, :, 0:3].contiguous().float()
            rgbs = gaussians[b, :, 3:6].contiguous().float() # [N, 3]
            opacity = gaussians[b, :, 6:7].contiguous().float() #
            rotations = gaussians[b, :, 7:11].contiguous().float() #旋转四元数
            scales = gaussians[b, :, 11:].contiguous().float() # 尺度

            means3D = torch.nan_to_num(means3D, nan=0.0, posinf=1.0, neginf=0.0)
            rgbs = torch.nan_to_num(rgbs, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            opacity = torch.nan_to_num(opacity, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            rotations = torch.nan_to_num(rotations, nan=0.0, posinf=0.0, neginf=0.0)
            norm = torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-6)
            rotations = rotations / norm
            scales = torch.nan_to_num(scales, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-4, max=10.0)
                

            
            
            # means2D 预留一个空的屏幕2D位置（后面rasterizer会用到
            means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device=device) #(N,3)

            # 再进入循环：对每个相机处理
            for v in range(V):
                # 当前batch的第v个相机的FOV、位姿。
                fovx_ = fovx[b, v].clone() 
                fovy_ = fovy[b, v].clone()
                c2w_ = c2w[b, v].clone()
                
                # get the w2c and projection matrix: from gaussains to images
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w_, fovx=fovx_, fovy=fovy_, znear=self.znear, zfar=self.zfar
                )
                
                # 但我们在投影变换（比如 perspective projection）里，通常是以相机的朝向为中心，处理中心到边缘的半角度。
                # render novel views
                tan_half_fovx = torch.tan(fovx_ * 0.5)
                tan_half_fovy = torch.tan(fovy_ * 0.5)
                
                # 初始化一个高斯点 Rasterizer Settings
                if self.renderer_type == "vanilla":
                    raster_settings = GaussianRasterizationSettings(
                        image_height=self.resolution[0],
                        image_width=self.resolution[1],
                        tanfovx=tan_half_fovx, # set the half the FOV
                        tanfovy=tan_half_fovy,
                        bg=self.bg_color if bg_color is None else bg_color,
                        scale_modifier=scale_modifier,
                        viewmatrix=w2c, # world to cam matrix
                        projmatrix=proj,
                        sh_degree=0,
                        campos=cam_p, # center cam
                        prefiltered=False,
                        debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                else:
                    raise NotImplementedError

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                if self.renderer_type == "vanilla":
                    # 高斯点半径 (radii)（用不上）
                    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                    )
                    rendered_normal = None
                else:
                    raise NotImplementedError

                # assert not has_nan_or_inf(rendered_image),"Rendered Images Contains NAN or INF.............."
                rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)
                images.append(rendered_image)
                alphas.append(rendered_alpha)
                depths.append(rendered_depth)

        images = torch.stack(images, dim=0).view(B, V, 3, self.resolution[0], self.resolution[1])
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
        depths = torch.stack(depths, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
            "depth": depths
        }

    # save the gaussains attributes
    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
    
    
    def render_customized_resolution(
        self, 
        gaussians: Float[Tensor, "B N F"], # 一堆高斯 blob，[B, N, 14]，每个点14个属性（位置、颜色、透明度、旋转、尺度）
        c2w: Float[Tensor, "B V 4 4"], # c2w---> B, 6, 4,4, default is the openGL
        fovx: Float[Tensor, "B V"] = None, # B, 6 
        fovy: Float[Tensor, "B V"] = None,
        rays_o: Float[Tensor, "B V H W 3"] = None,
        rays_d: Float[Tensor, "B V H W 3"] = None,
        bg_color: Float[Tensor, "... 3"] = None, 
        scale_modifier: float = 1.,
        new_resolution: list = [512, 512],
    ):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        # at least one of fovx and fovy is not none
        # 要求至少要有一个 FOV，否则不知道怎么建相机视锥。
        assert fovx is not None or fovy is not None
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx

        device = gaussians.device
        B, V = c2w.shape[:2]

        # ---------------- sanitize fovx/fovy ---------------- #
        def safe_fov(tensor):
            tensor = torch.nan_to_num(tensor, nan=0.5, posinf=0.5, neginf=0.5)  # default 0.5 rad ≈ 57 deg
            return tensor.clamp(min=1e-3, max=math.pi - 1e-3)
        if fovx is None and fovy is None:
            raise ValueError("At least one of fovx or fovy must be provided.")
        if fovx is None:
            fovx = safe_fov(fovy.clone())
        if fovy is None:
            fovy = safe_fov(fovx.clone())
        fovx = safe_fov(fovx)
        fovy = safe_fov(fovy)

        # ---------------- sanitize c2w ---------------- #
        c2w = torch.nan_to_num(c2w, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------------- sanitize bg_color ---------------- #
        if bg_color is not None:
            bg_color = torch.nan_to_num(bg_color, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            bg_color = self.bg_color  # use default (usually [0, 0, 0])

        # ---------------- sanitize rays ---------------- #
        if rays_o is not None:
            rays_o = torch.nan_to_num(rays_o, nan=0.0, posinf=0.0, neginf=0.0)
        if rays_d is not None:
            rays_d = torch.nan_to_num(rays_d, nan=0.0, posinf=0.0, neginf=0.0)


        # loop of loop...
        # 为了最后收集每个 batch、每个相机的渲染结果
        images = []
        alphas = []
        depths = []
        
        # batch size
        for b in range(B):
            # 3D 坐标（means）
            means3D = gaussians[b, :, 0:3].contiguous().float()
            rgbs = gaussians[b, :, 3:6].contiguous().float() # [N, 3]
            opacity = gaussians[b, :, 6:7].contiguous().float() #
            rotations = gaussians[b, :, 7:11].contiguous().float() #旋转四元数
            scales = gaussians[b, :, 11:].contiguous().float() # 尺度

            means3D = torch.nan_to_num(means3D, nan=0.0, posinf=1.0, neginf=0.0)
            rgbs = torch.nan_to_num(rgbs, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            opacity = torch.nan_to_num(opacity, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            rotations = torch.nan_to_num(rotations, nan=0.0, posinf=0.0, neginf=0.0)
            norm = torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-6)
            rotations = rotations / norm
            scales = torch.nan_to_num(scales, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-4, max=10.0)
                

            
            
            # means2D 预留一个空的屏幕2D位置（后面rasterizer会用到
            means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device=device) #(N,3)

            # 再进入循环：对每个相机处理
            for v in range(V):
                # 当前batch的第v个相机的FOV、位姿。
                fovx_ = fovx[b, v].clone() 
                fovy_ = fovy[b, v].clone()
                c2w_ = c2w[b, v].clone()
                
                # get the w2c and projection matrix: from gaussains to images
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w_, fovx=fovx_, fovy=fovy_, znear=self.znear, zfar=self.zfar
                )
                
                # 但我们在投影变换（比如 perspective projection）里，通常是以相机的朝向为中心，处理中心到边缘的半角度。
                # render novel views
                tan_half_fovx = torch.tan(fovx_ * 0.5)
                tan_half_fovy = torch.tan(fovy_ * 0.5)
                
                # 初始化一个高斯点 Rasterizer Settings
                if self.renderer_type == "vanilla":
                    raster_settings = GaussianRasterizationSettings(
                        image_height=new_resolution[0],
                        image_width=new_resolution[1],
                        tanfovx=tan_half_fovx, # set the half the FOV
                        tanfovy=tan_half_fovy,
                        bg=self.bg_color if bg_color is None else bg_color,
                        scale_modifier=scale_modifier,
                        viewmatrix=w2c, # world to cam matrix
                        projmatrix=proj,
                        sh_degree=0,
                        campos=cam_p, # center cam
                        prefiltered=False,
                        debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                else:
                    raise NotImplementedError

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                if self.renderer_type == "vanilla":
                    # 高斯点半径 (radii)（用不上）
                    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                    )
                    rendered_normal = None
                else:
                    raise NotImplementedError

                # assert not has_nan_or_inf(rendered_image),"Rendered Images Contains NAN or INF.............."
                rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)
                images.append(rendered_image)
                alphas.append(rendered_alpha)
                depths.append(rendered_depth)

        images = torch.stack(images, dim=0).view(B, V, 3, new_resolution[0], new_resolution[1])
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, new_resolution[0], new_resolution[1])
        depths = torch.stack(depths, dim=0).view(B, V, 1, new_resolution[0], new_resolution[1])

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
            "depth": depths
        }