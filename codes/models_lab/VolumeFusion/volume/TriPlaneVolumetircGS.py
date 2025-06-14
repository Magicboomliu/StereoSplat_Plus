import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import torch
import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange
from .tpvformer_encoder import TPVFormerEncoder
from .TriPlaneGSDecoder import TriPlaneVolumeGaussianDecoder



class TriPlaneVolumetircGS(nn.Module):
    def __init__(self,
                 encoder=None,
                 gs_decoder=None,
                 use_checkpoint=False,
                 **kwargs,
                 ):
        
        super().__init__()

        self.use_checkpoint = use_checkpoint
        
        self.encoder = TPVFormerEncoder(**encoder)
        
        self.tpv_h = self.encoder.tpv_h
        self.tpv_w = self.encoder.tpv_w
        self.tpv_z = self.encoder.tpv_z
        self.pc_range = self.encoder.pc_range  #[-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]
        
        
        self.pc_xrange = self.pc_range[3] - self.pc_range[0] #100
        self.pc_yrange = self.pc_range[4] - self.pc_range[1] #100
        self.pc_zrange = self.pc_range[5] - self.pc_range[2] #15
        
        # # FIXME Here
        self.feature_project_in = nn.Linear(164,128,bias=False)
        
        self.gs_decoder = TriPlaneVolumeGaussianDecoder(**gs_decoder)
        
        

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img_feats, 
                        candidate_gaussians, 
                        candidate_feats_in, 
                        img_metas=None, 
                        status="train"):
        """Forward training function.
        # candaites gaussains: the shared location masked gaussians between the pixel gaussains and the volume gaussains.
        # candadiates feature mask : the shared locations masked gaussains features.
        """
        
        candidate_feats = []
        
        for cf in candidate_feats_in:
            candidate_feats.append(self.feature_project_in(cf))
        
        
        if candidate_gaussians is not None and candidate_feats is not None:
            bs = len(candidate_feats) # batch size    
            _, c = candidate_feats[0].shape    # just get the dimension:14
            
    
            # project the feature into triplane
            # print(self.tpv_h)  # 192
            # print(self.tpv_w)  # 192
            # print(self.tpv_z)  # 16
    
            # here is the lidar coordinate
            # https://www.nuscenes.org/nuscenes#data-collection  
            project_feats_hw = candidate_feats[0].new_zeros((bs, self.tpv_h, self.tpv_w, c)) # 4x192x192x128: H x W
            project_feats_zh = candidate_feats[0].new_zeros((bs, self.tpv_z, self.tpv_h, c)) # 4x16x192x128:  Z x H
            project_feats_wz = candidate_feats[0].new_zeros((bs, self.tpv_w, self.tpv_z, c)) # 4x192x16x128:  W x Z
            
            # print(project_feats_hw.shape)
            # print(project_feats_hw.min())
            # print(project_feats_hw.max())
            # print(project_feats_hw.mean())
            # quit()
            
 
            

            # traversal across the batch size
            for i in range(bs):
                candidate_xyzs_i = candidate_gaussians[i].means # get the pixel-wise 3DGS
       
                # get H-plane index of the pixel gaussains: torch.Size([N1])
                candidate_hs_i = (self.tpv_h * (candidate_xyzs_i[..., 1] - self.pc_range[1]) / self.pc_yrange - 0.5).int()
                # get W-plane index:   torch.Size([N1])
                candidate_ws_i = (self.tpv_w * (candidate_xyzs_i[..., 0] - self.pc_range[0]) / self.pc_xrange - 0.5).int()
                # get Z-plane index:    torch.Size([N1])
                candidate_zs_i = (self.tpv_z * (candidate_xyzs_i[..., 2] - self.pc_range[2]) / self.pc_zrange - 0.5).int()
                # n, c
                #candidate_feats_i = candidate_feats[[i, valid_mask]]
                candidate_feats_i = candidate_feats[i] #-----> [N1,C]
                
                # hw: n, 2
                candidate_coords_hw_i = torch.stack([candidate_hs_i, candidate_ws_i], dim=-1) # (H x W) index from triplane
                # linder index, assume the hw plane as a linear for quick search.
                # [N1,]
                linear_inds_hw_i = (candidate_coords_hw_i[..., 0] * self.tpv_w + candidate_coords_hw_i[..., 1]).to(dtype=torch.int64)                
                project_feats_hw_i = project_feats_hw[i].view(-1, c) # [self.tpv_h * self.tpv_w,C]--> also as linear

                # add the pixel gs feature into the projected hw-plane: 如果多个点投影到同一位置就自动累加
                project_feats_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                
                # 创建一个与特征一样大小的零张量，用来统计每个像素被累加了多少次
                count_hw_i = project_feats_hw_i.new_zeros((self.tpv_h * self.tpv_w, c), dtype=torch.float32)
                ones_hw_i = torch.ones_like(candidate_feats_i)
                count_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), ones_hw_i)
                count_hw_i = torch.where(count_hw_i == 0, torch.ones_like(count_hw_i), count_hw_i)
                #每个位置除以累加次数，得到平均值
                project_feats_hw_i = (project_feats_hw_i / count_hw_i).view(self.tpv_h, self.tpv_w, c)
                project_feats_hw[i] = project_feats_hw_i  #(H,W,C)
                

                
                

                # zh: n, 2
                candidate_coords_zh_i = torch.stack([candidate_zs_i, candidate_hs_i], dim=-1)
                linear_inds_zh_i = (candidate_coords_zh_i[..., 0] * self.tpv_h + candidate_coords_zh_i[..., 1]).to(dtype=torch.int64)
                project_feats_zh_i = project_feats_zh[i].view(-1, c)
                project_feats_zh_i.scatter_add_(0, linear_inds_zh_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_zh_i = project_feats_zh_i.new_zeros((self.tpv_z * self.tpv_h, c), dtype=torch.float32)
                ones_zh_i = torch.ones_like(candidate_feats_i)
                count_zh_i.scatter_add_(0, linear_inds_zh_i.unsqueeze(-1).expand(-1, c), ones_zh_i)
                count_zh_i = torch.where(count_zh_i == 0, torch.ones_like(count_zh_i), count_zh_i)
                project_feats_zh_i = (project_feats_zh_i / count_zh_i).view(self.tpv_z, self.tpv_h, c)
                project_feats_zh[i] = project_feats_zh_i

                # wz: n, 2
                candidate_coords_wz_i = torch.stack([candidate_ws_i, candidate_zs_i], dim=-1)
                linear_inds_wz_i = (candidate_coords_wz_i[..., 0] * self.tpv_z + candidate_coords_wz_i[..., 1]).to(dtype=torch.int64)
                project_feats_wz_i = project_feats_wz[i].view(-1, c)
                project_feats_wz_i.scatter_add_(0, linear_inds_wz_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_wz_i = project_feats_wz_i.new_zeros((self.tpv_w * self.tpv_z, c), dtype=torch.float32)
                ones_wz_i = torch.ones_like(candidate_feats_i)
                count_wz_i.scatter_add_(0, linear_inds_wz_i.unsqueeze(-1).expand(-1, c), ones_wz_i)
                count_wz_i = torch.where(count_wz_i == 0, torch.ones_like(count_wz_i), count_wz_i)
                project_feats_wz_i = (project_feats_wz_i / count_wz_i).view(self.tpv_w, self.tpv_z, c)
                project_feats_wz[i] = project_feats_wz_i
            
            project_feats_hw = rearrange(project_feats_hw, "b h w c -> b c h w")
            project_feats_zh = rearrange(project_feats_zh, "b h w c -> b c h w")
            project_feats_wz = rearrange(project_feats_wz, "b h w c -> b c h w")
            project_feats = [project_feats_hw, project_feats_zh, project_feats_wz]
        else:
            project_feats = [None, None, None]

        
        
        # get shared features
        
        if self.use_checkpoint and status != "test":
            # img metas: including lidar2img, image shape.
            input_vars_enc = (img_feats, project_feats, img_metas)
            outs = torch.utils.checkpoint.checkpoint(
                self.encoder, *input_vars_enc, use_reentrant=False
            )
            # outs = list(outs)
            # outs = outs.append(img_metas)
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, outs,img_metas,use_reentrant=False)
            
        else:
            
            # gaussain encoding.
            outs = self.encoder(img_feats, project_feats, img_metas) #
            # outs = list(outs)
            # outs = outs.append(img_metas)
            # gaussain decoding.
            gaussians = self.gs_decoder(outs,img_metas) #(B,tpv_h, tpv_w, tpv_z,3,14), here 3 is the gpv

        
        return gaussians
