import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .vit_fpn import ViTFeaturePyramid
from .mv_transformer import (
    MultiViewFeatureTransformer,
    batch_features_camera_parameters,
)
from .matching import warp_with_pose_depth_candidates
from .utils import mv_feature_add_position
from .dpt_head import DPTHead
from .ldm_unet.unet import UNetModel, AttentionBlock
from einops import rearrange


class MultiViewUniMatch(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        grid_sample_disable_cudnn=False,
        **kwargs,
    ):
        super(MultiViewUniMatch, self).__init__()


        # CNN
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        # monocular backbones: final
        self.vit_type = vit_type

        # cost volume
        self.num_depth_candidates = num_depth_candidates

        # upsampler
        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        # CNN
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        if self.num_scales > 1:
            # generate multi-scale features
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128, scale_factors=[2**i for i in range(self.num_scales)]
            )

        # monodepth
        encoder = vit_type  # can also be 'vitb' or 'vitl'
        self.pretrained = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{:}14".format(encoder)
        )

        del self.pretrained.mask_token  # unused

        if self.num_scales > 1:
            # generate multi-scale features
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=vit_feature_channel,
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        # UNet regressor
        self.regressor = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head = nn.ModuleList()

        for i in range(self.num_scales):
            curr_depth_candidates = num_depth_candidates // (4**i)
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2**i)

            mono_feature_channels = vit_feature_channel // (2**i)

            # concat(cost volume, cnn feature, mv feature, mono feature)
            in_channels = (
                curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

            # unet channels
            channels = unet_channels // (2**i)

            # unet channel mult & unet_attn_resolutions
            if i > 0:
                unet_channel_mult = unet_channel_mult + [1]
                unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

            # unet
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=unet_num_res_blocks,
                    attention_resolutions=unet_attn_resolutions,
                    channel_mult=unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                )
            )

            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

            self.regressor.append(nn.Sequential(*modules))

            # regressor residual
            self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))

            # depth head
            self.depth_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels * 2, 3, 1, 1, padding_mode="replicate"
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        channels * 2,
                        curr_depth_candidates,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                )
            )

        # upsampler
        # concat(lowres_depth, cnn feature, mv feature, mono feature)
        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.upsampler = DPTHead(
            **model_configs[vit_type],
            downsample_factor=upsample_factor,
            num_scales=num_scales,
        )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        # images: [B, V, C, H, W]
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        # list of [BV, C, H, W], resolution from high to low
        features = self.backbone(concat)
        # reverse: resolution from low to high: select the higest resolution
        features = features[::-1]

        return features

    def forward(
        self,
        images,
        images_feat=None,
        attn_splits_list=None,
        intrinsics=None,
        min_depth=1.0 / 0.5,  # inverse depth range
        max_depth=1.0 / 100,
        num_depth_candidates=128,
        extrinsics=None,
        nn_matrix=None,
        **kwargs,
    ):
        '''
        Inputs: 
            images: torch.Tensor  # [B, V, 3, H, W]，多视角彩色图像输入（已归一化）
            attn_splits_list: List[int]  # e.g., [2] 表示 transformer 中 attention 区域划分
            intrinsics: torch.Tensor  # [B, V, 3, 3]，每个视角的相机内参矩阵（已归一化）
            extrinsics: torch.Tensor  # [B, V, 4, 4]，每个视角的相机外参矩阵（c2w）
            min_depth: torch.Tensor  # [B, V]，每个视角最小 inverse depth（1 / far）
            max_depth: torch.Tensor  # [B, V]，每个视角最大 inverse depth（1 / near）
            nn_matrix: torch.LongTensor  # [B, V, K]，近邻图像索引矩阵（可选）
        
        Outputs(dict): scale default is 4
            {"depth_preds": List[Tensor],  # 每层 scale 的最终深度图，List 中每项为 [B, V, H, W]
            "match_probs": List[Tensor],  # 每层 scale 的匹配概率 volume，List 中每项为 [B*V, D, H/scale, W/scale]
            "features_cnn_all_scales": List[Tensor],  # CNN 特征（多尺度）: List of [B*V, C, H/scale, W/scale]
            "features_cnn": List[Tensor],  # CNN 特征（用于最后预测的尺度）: 同上
            "features_mv": List[Tensor],  # Transformer 融合后的多视角特征: List of [B*V, C, H/scale, W/scale]
            "features_mono_intermediate": List[Tensor],  # ViT-DINO 中间层特征：List of [B*V, C, H/scale, W/scale]
            "features_mono": List[Tensor],  # 最终用来预测深度的单目特征: List of [B*V, C, H/scale, W/scale]
            }
            
        '''

        results_dict = {}
        depth_preds = []
        match_probs = []

        # first normalize images using imagenet 
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.regressor, num_views=v)

        # NOTE: in this codebase, intrinsics are normalized by image width and height
        # in unimatch's codebase: https://github.com/autonomousvision/unimatch, no normalization
        
        # recover the true instrinsics
        intrinsics = intrinsics.clone()
    
        # here is the inverse.
        # max_depth, min_depth: [B, V] -> [BV]
        max_depth = max_depth.view(-1)
        min_depth = min_depth.view(-1)

        # list of features, resolution low to high
        # list of [BV, C, H, W]
        features_list_cnn = self.extract_feature(images) # 3 levels
        
        # from 1/4, 1/2, 1/2 Resolution
        bs,views,cha,height,width = images_feat.shape
        images_feat = images_feat.reshape(-1,cha,height,width)
        
        features_list_cnn[0] = features_list_cnn[0] + images_feat
        
        features_list_cnn_all_scales = features_list_cnn        
        features_list_cnn = features_list_cnn[: self.num_scales] # get the 1/4 feature,. the lowest scale
        
        # recorde all the features and the lowest scale features        
        results_dict.update({"features_cnn_all_scales": features_list_cnn_all_scales})
        results_dict.update({"features_cnn": features_list_cnn})

        # mv transformer features
        # add position to features
        # attn_splits = 2  # 表示图像被划分成 2×2 个 patch 区域,similar to SwinTransformer
        attn_splits = attn_splits_list[0] # at
        
        features_cnn_pos = mv_feature_add_position(
            features_list_cnn[0], attn_splits, self.feature_channels
        ) # [BV, C, H, W]
        

        # list of [B, C, H, W]: features_list[i]: shape = [B, C, H, W]  # 表示第 i 个视角的 CNN 特征
        features_list = list(
            torch.unbind(
                rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
            )
        ) # The length is the V, each is [B,C,H,W], from [BV,C,H,W] ---> [[B,C,H,W]...[B,C,H,W]] (length is V)
        
        # GET THE CROSS VIEW ATTENTION
        features_list_mv = self.transformer(
            features_list,
            attn_num_splits=attn_splits,
            nn_matrix=nn_matrix,
        ) # The length is the V, each is [B,C,H,W], from [BV,C,H,W] ---> [[B,C,H,W]...[B,C,H,W]] (length is V)
        

        features_mv = rearrange(
            torch.stack(features_list_mv, dim=1), "b v c h w -> (b v) c h w"
        )  # [BV, C, H, W]
        

        if self.num_scales > 1:
            # multi-scale mv features: resolution from low to high
            # list of [BV, C, H, W]
            features_list_mv = self.mv_pyramid(features_mv)
        else:
            features_list_mv = [features_mv]

        # feature mv is the feature aggregated by the transformer
        results_dict.update({"features_mv": features_list_mv})

        # mono feature
        #  目的：配合 ViT（DINOv2）Backbone 的 patch embedding
        ori_h, ori_w = images.shape[-2:]
        # make sure the input image can be divided by 14? 
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        concat = F.interpolate(
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        )

        # get intermediate features
        intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }
        
        '''
        input image shape = [2, 3, 224, 840]
                    ↑  ↑    ↑    ↑
                    B  C    H    W
                    
        Patch 数量（H方向） = 224 / 14 = 16  
        Patch 数量（W方向） = 840 / 14 = 60  
        → 总共 16 × 60 = 960 个 patch

        对于每张图，ViT 将它切成了 960 个 patch；

        每个 patch 会输出一个 384 维的 token 向量；

        每层中间输出（如 get_intermediate_layers()）都是这种结构。
        '''

        # default is the 4 channels
        mono_intermediate_features = list(
            self.pretrained.get_intermediate_layers(
                concat, intermediate_layer_idx[self.vit_type], return_class_token=False
            )
        ) # torch.Size([2, 960, 384])

        for i in range(len(mono_intermediate_features)):
            curr_features = (
                mono_intermediate_features[i]
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            ) # [B, C, H', W'], here the H' is resize_h // 14, resize_w // 14
            
            # resize to 1/8 resolution of the images
            curr_features = F.interpolate(
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            # update the featuires
            mono_intermediate_features[i] = curr_features

        # this is the monocular feature list from the Dino-Vit 
        results_dict.update({"features_mono_intermediate": mono_intermediate_features})

        # last mono feature: most aggregated
        mono_features = mono_intermediate_features[-1]

        # to 1/4 resolustion
        if self.lowest_feature_resolution == 4:
            mono_features = F.interpolate(
                mono_features, scale_factor=2, mode="bilinear", align_corners=True
            )


        
        if self.num_scales > 1:
            # multi-scale mono features, resolution from low to high
            # list of [BV, C, H, W]
            features_list_mono = self.mono_pyramid(mono_features)
        else:
            features_list_mono = [mono_features]

        # this is the last scale monocular feature list from the Dino-Vit to 1/4
        results_dict.update({"features_mono": features_list_mono})

        depth = None

        # 遍历每一层尺度（粗→细），做多尺度的深度估计，帮助提升精度和鲁棒性。
        for scale_idx in range(self.num_scales):
            
            downsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )
            
 

            # scale intrinsics: 将相机内参按当前特征图的下采样因子进行缩放（焦距、主点都缩小）。
            intrinsics_curr = intrinsics.clone()  # [B, V, 3, 3]
            intrinsics_curr[:, :, :2] = intrinsics_curr[:, :, :2] / downsample_factor
            
    

            # build cost volume
            features_mv = features_list_mv[scale_idx]  # [BV, C, H, W]

            # list of [B, C, H, W]
            features_mv_curr = list(
                torch.unbind(
                    rearrange(features_mv, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
                )
            ) # [B,C,H,W], length is nums_cams

            # default the extrinsic shape is [B,V,4,4]

            intrinsics_curr = list(
                torch.unbind(intrinsics_curr, dim=1)
            )  # list of [B, 3, 3]
            extrinsics_curr = list(torch.unbind(extrinsics, dim=1))  # list of [B, 4, 4]

            '''
            ref_features: 当前视角的特征（作为 query）
            tgt_features: 与其匹配的其他视角特征（作为 key/value）
            
            '''
            # ref: [BV, C, H, W], [BV, 3, 3], [BV, 4, 4]
            # tgt: [BV, V-1, C, H, W], [BV, V-1, 3, 3], [BV, V-1, 4, 4], here the V-1 K-1（邻接矩阵中给定的邻居数 - 自己）
            (
                ref_features,
                ref_intrinsics,
                ref_extrinsics,
                tgt_features,
                tgt_intrinsics,
                tgt_extrinsics,
            ) = batch_features_camera_parameters(
                features_mv_curr,
                intrinsics_curr,
                extrinsics_curr,
                nn_matrix=nn_matrix,
            )

            b_new, _, c, h, w = tgt_features.size()

            # relative pose
            # extrinsics: c2w
            # 将 ref_extrinsics 转换到 tgt_extrinsics 的相对位姿（Pose = T_tgt^{-1} * T_ref）
            pose_curr = torch.matmul(
                tgt_extrinsics.inverse(), ref_extrinsics.unsqueeze(1)
            )  # [BV, V-1, 4, 4]

            if scale_idx > 0:
                # 2x upsample depth
                assert depth is not None
                depth = F.interpolate(
                    depth, scale_factor=2, mode="bilinear", align_corners=True
                ).detach()

            # 128
            num_depth_candidates = self.num_depth_candidates // (4**scale_idx)
            

            # generate depth candidates
            if scale_idx == 0:
                # min_depth, max_depth: [BV]
                depth_interval = (max_depth - min_depth) / (
                    self.num_depth_candidates - 1
                )  # [BV]
                
                # # 说明：生成 [BV, D, 1, 1] 的均匀 inverse depth 候选平面
                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]

                depth_candidates = min_depth.view(-1, 1, 1, 1) + linear_space * (
                    max_depth - min_depth
                ).view(
                    -1, 1, 1, 1
                )  # [BV, D, 1, 1], from # 说明：生成 [BV, D, 1, 1] 的均匀 inverse depth 候选平面
            else:
                # half interval each scale
                depth_interval = (
                    (max_depth - min_depth)
                    / (self.num_depth_candidates - 1)
                    / (2**scale_idx)
                )  # [BV]
                # [BV, 1, 1, 1]
                depth_interval = depth_interval.view(-1, 1, 1, 1)

                # [BV, 1, H, W]
                depth_range_min = (
                    depth - depth_interval * (num_depth_candidates // 2)
                ).clamp(min=min_depth.view(-1, 1, 1, 1))
                depth_range_max = (
                    depth + depth_interval * (num_depth_candidates // 2 - 1)
                ).clamp(max=max_depth.view(-1, 1, 1, 1))

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]
                depth_candidates = depth_range_min + linear_space * (
                    depth_range_max - depth_range_min
                )  # [BV, D, H, W]

            '''
            depth_candidates 是 [BV, D, H, W]（或 [BV, D, 1, 1] 在 scale=0）
            你希望它与 tgt_features 匹配，即需要变成 [BV * (V-1), D, H, W]
            所以这里通过 repeat + view 将其扩展为多个 target 对应的 depth 候选（复制给每个匹配视角）
            scale_idx == 0 特殊处理因为初始 depth 是 [BV, D, 1, 1]，需要 broadcast 到 [h, w]
            '''
            if scale_idx == 0:
                # [BV*(V-1), D, H, W]
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, h, w)
                    .view(-1, num_depth_candidates, h, w)
                )
            else:
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, 1, 1)
                    .view(-1, num_depth_candidates, h, w)
                )
                
            intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(
                -1, 3, 3
            )  # [BV, 3, 3]
            intrinsics_input = intrinsics_input.unsqueeze(1).repeat(
                1, tgt_features.size(1), 1, 1
            )  # [BV, V-1, 3, 3]
            



            # 给定一个参考图像的特征图、内参、相对姿态和一组深度候选，生成被目标图像 warp 回参考图像坐标系下的特征图（用于 plane-sweep stereo）
            warped_tgt_features = warp_with_pose_depth_candidates(
                rearrange(tgt_features, "b v ... -> (b v) ..."),
                rearrange(intrinsics_input, "b v ... -> (b v) ..."),
                rearrange(pose_curr, "b v ... -> (b v) ..."),
                1.0 / depth_candidates_curr,  # convert inverse depth to depth
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            )  # [BV*(V-1), C, D, H, W]
            
        

            # ref: [BV, C, H, W]
            # warped: [BV*(V-1), C, D, H, W] -> [BV, V-1, C, D, H, W]
            warped_tgt_features = rearrange(
                warped_tgt_features,
                "(b v) ... -> b v ...",
                b=b_new,
                v=tgt_features.size(1),
            ) # 这行代码的作用是 将 warp 后的目标图像特征从扁平的 [BV*(V-1), ...] 还原成 [BV, V-1, ...] 的结构，方便后续计算。


            # [BV, V-1, D, H, W] -> [BV, D, H, W]
            # average cross other views
            cost_volume = (
                (ref_features.unsqueeze(-3).unsqueeze(1) * warped_tgt_features).sum(2)
                / (c**0.5)
            ).mean(1)

            # regressor
            features_cnn = features_list_cnn[scale_idx]  # [BV, C, H, W]

            features_mono = features_list_mono[scale_idx]  # [BV, C, H, W]

            concat = torch.cat(
                (cost_volume, features_cnn, features_mv, features_mono), dim=1
            )

            out = self.regressor[scale_idx](concat) + self.regressor_residual[
                scale_idx
            ](concat) # torch.Size([2, 128, 56, 208])


            # depth pred
            match_prob = F.softmax(
                self.depth_head[scale_idx](out), dim=1
            )  # [BV, D, H, W]
            match_probs.append(match_prob)

            
            if scale_idx == 0:
                # [BV, D, H, W]
                depth_candidates = depth_candidates.repeat(1, 1, h, w)
            depth = (match_prob * depth_candidates).sum(
                dim=1, keepdim=True
            )  # [BV, 1, H, W]  # inverse depth map

            # upsample to the original resolution for supervison at training time only
            if self.training and scale_idx < self.num_scales - 1:
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=downsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth_preds.append(depth_bilinear)

            # final output, learned upsampler
            if scale_idx == self.num_scales - 1:
                residual_depth = self.upsampler(
                    mono_intermediate_features,
                    # resolution high to low
                    cnn_features=features_list_cnn_all_scales[::-1],
                    mv_features=(
                        features_mv if self.num_scales == 1 else features_list_mv[::-1]
                    ),
                    depth=depth,
                )

                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=self.upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth = (depth_bilinear + residual_depth).clamp(
                    min=min_depth.view(-1, 1, 1, 1), max=max_depth.view(-1, 1, 1, 1)
                )

                depth_preds.append(depth)

        # convert inverse depth to depth
        for i in range(len(depth_preds)):
            depth_pred = 1.0 / depth_preds[i].squeeze(1)  # [BV, H, W]
            depth_preds[i] = rearrange(
                depth_pred, "(b v) ... -> b v ...", b=b, v=v
            )  # [B, V, H, W]

        results_dict.update({"depth_preds": depth_preds}) # scale-wise [B.V,H,W]
        results_dict.update({"match_probs": match_probs})  # scale-wise [B.V,H/4,W/4]

        return results_dict


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
        isinstance(module, nn.ModuleList)
        or isinstance(module, nn.Sequential)
        or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)




if __name__=="__main__":
    
    
    pass