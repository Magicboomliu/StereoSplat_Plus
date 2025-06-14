
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from dataclasses import dataclass


from einops import einsum, rearrange
from jaxtyping import Float
from ..encoder.common.gaussian_adapter import GaussianAdapter
from ..encoder.common.gaussian_adapter import GaussianAdapterCfg


def sigmoid_scaling(scaling:torch.Tensor, lower_bound=0.005, upper_bound=0.02):
    sig = torch.sigmoid(scaling)
    return lower_bound * (1 - sig) + upper_bound * sig



@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    


# @MODELS.register_module()
class TriPlaneVolumeGaussianDecoder(BaseModule):
    # gpv
    # 每个体素（Voxel）内放置的高斯（Gaussian）数量。
    def __init__(
        self, tpv_h, tpv_w, tpv_z, pc_range, gs_dim=14,
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, gpv=4, offset_max=None, scale_max=None,
        use_checkpoint=False,
        gaussian_head_settings_dict=None
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.gpv = gpv

        out_dims = in_dims if out_dims is None else out_dims

        # encoder
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        # Finally Decoder to get the Gaussains

        self.use_checkpoint = use_checkpoint

        # set activations
        # TODO: check if optimal
        self.pos_act = lambda x: torch.tanh(x)
        if offset_max is None:
            self.offset_max = [1.0] * 3 # meters
        else:
            self.offset_max = offset_max
        #self.scale_act = lambda x: sigmoid_scaling(x, lower_bound=0.005, upper_bound=0.02)
        if scale_max is None:
            self.scale_max = [1.0] * 3 # meters
        else:
            self.scale_max = scale_max
        
        self.opacity_act = lambda x: torch.sigmoid(x)
        
        # self.rot_act = lambda x: F.normalize(x, dim=-1)
        # self.rgb_act = lambda x: torch.sigmoid(x)
        # self.scale_act = lambda x: torch.sigmoid(x)

        # gaussians adapter
        self.gaussian_adapter_vs = GaussianAdapter(**gaussian_head_settings_dict)
        num_gaussian_parameters = self.gaussian_adapter_vs.d_in_for_volume
        
        gs_dim = num_gaussian_parameters
        self.gs_decoder = nn.Linear(out_dims, gs_dim*gpv)

        # obtain anchor points for gaussians
        gs_anchors = self.get_reference_points(tpv_h * scale_h, tpv_w * scale_w, tpv_z * scale_z, pc_range) # 1, w, h, z, 3
        self.register_buffer('gs_anchors', gs_anchors)
        
    
    @staticmethod
    def get_reference_points(H, W, Z, pc_range, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            H, W: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, -1).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, -1, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(2, 1, 0, 3) # w, h, z, 3
        ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1, 1) # b, w, h, z, 3
        return ref_3d
    
    def forward(self, tpv_list,img_meta=None ,debug=False):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        
        # get the tri-plane features 
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]   
        
        
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)
        
        
        

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw, 
                size=(self.tpv_h*self.scale_h, self.tpv_w*self.scale_w),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh, 
                size=(self.tpv_z*self.scale_z, self.tpv_h*self.scale_h),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz, 
                size=(self.tpv_w*self.scale_w, self.tpv_z*self.scale_z),
                mode='bilinear'
            )

        #print("before voxelize:{}".format(torch.cuda.memory_allocated(0)))
        tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
        tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
        tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)

        gaussians = tpv_hw + tpv_zh + tpv_wz  #(B,128,192,192,16)
        
        #print("after voxelize:{}".format(torch.cuda.memory_allocated(0)))
        gaussians = gaussians.permute(0, 2, 3, 4, 1) # bs, w, h, z, c
        bs, w, h, z, _ = gaussians.shape
        
        
        if self.use_checkpoint:
            gaussians = torch.utils.checkpoint.checkpoint(self.decoder, gaussians, use_reentrant=False)
            
            
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, gaussians, use_reentrant=False)
            gaussians = gaussians.view(bs, w, h, z, self.gpv, -1)
        else:
            gaussians = self.decoder(gaussians)
            gaussians = self.gs_decoder(gaussians)
            gaussians = gaussians.view(bs, w, h, z, self.gpv, -1)
        
        

        opacity = self.opacity_act(gaussians[..., :1])
        gs_offsets_x = self.pos_act(gaussians[..., 1:2]) * self.offset_max[0] # bs, w, h, z, 3
        gs_offsets_y = self.pos_act(gaussians[..., 2:3]) * self.offset_max[1] # bs, w, h, z, 3
        gs_offsets_z = self.pos_act(gaussians[..., 3:4]) * self.offset_max[2] # bs, w, h, z, 3
        #gs_offsets = gaussians[..., :3]
        gs_positions = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1) + self.gs_anchors[:, :, :, :, None, :]
        

        gaussians = self.gaussian_adapter_vs.forward_for_world(
                          opacities=opacity,
                          mean3D=gs_positions,
                          raw_gaussians=gaussians[...,4:],
                          scale_max=self.scale_max,
        )

        gaussians_output = Gaussians(
            rearrange(
                gaussians.means,
                "b tpv_x tpv_y tpv_z gpv xyz -> b (tpv_x tpv_y tpv_z gpv) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b tpv_x tpv_y tpv_z gpv i j -> b (tpv_x tpv_y tpv_z gpv) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b tpv_x tpv_y tpv_z gpv c d_sh -> b (tpv_x tpv_y tpv_z gpv) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b tpv_x tpv_y tpv_z gpv -> b (tpv_x tpv_y tpv_z gpv)",
            ),
        )

        return gaussians_output    

        #print("after decode:{}".format(torch.cuda.memory_allocated(0)))

        
        # x = torch.cat([gs_positions, gaussians[..., 3:]], dim=-1)
        # rgbs = self.rgb_act(x[..., 3:6])
        
        # rotation = self.rot_act(x[..., 7:11])
        # scale_x = self.scale_act(x[..., 11:12]) * self.scale_max[0]
        # scale_y = self.scale_act(x[..., 12:13]) * self.scale_max[1]
        # scale_z = self.scale_act(x[..., 13:14]) * self.scale_max[2]

        # if debug:
        #     opacity[:] = 1.0
        #     scale_x[:] = 0.5
        #     scale_y[:] = 0.5
        #     scale_z[:] = 0.5
        #     rgbs[..., 0] = 1.0
        #     rgbs[..., 1] = 0.0
        #     rgbs[..., 2] = 0.0

        # gaussians = torch.cat([gs_positions, rgbs, opacity, rotation, scale_x, scale_y, scale_z], dim=-1) # bs, w, h, z, gpv, 14
    
        # return gaussians
