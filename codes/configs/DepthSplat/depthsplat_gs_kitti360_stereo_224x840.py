_base_ = []

model = dict(
    type='DepthSplat',  # 假设你的顶层模型名叫这个

    encoder=dict(
        type='DepthSplatEncoder',  # 原来的主类名

        name='depthsplat_encoder',
        num_depth_candidates=128, # cost volume matching candiadtaes
        
        num_surfaces=1, # number of surface is what?
        
        gaussians_per_pixel=1,

        # gaussains adapter is what?
        gaussian_adapter=dict(
            gaussian_scale_min=1e-10,
            gaussian_scale_max=3.0,
            sh_degree=2,
        ),

        d_feature=128,

        visualizer=dict(
            num_samples=8,
            min_resolution=256,
            export_ply=False,
        ),

        # loaded the unimatch_weaight
        unimatch_weights_path="/data1/zliu/pretrained_foundataion_models/depth_estimation/Depthsplat/gmflow-scale1-things-e9887eda.pth",

        multiview_trans_attn_split=2,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=[1, 1, 1],
        costvolume_unet_attn_res=[],
        depth_unet_feat_dim=64,
        depth_unet_attn_res=[],
        depth_unet_channel_mult=[1, 1, 1],
        downscale_factor=4,
        shim_patch_size=4,

        local_mv_match=2,

        monodepth_vit_type='vits',

        supervise_intermediate_depth=True,
        return_depth=True,

        num_scales=1,
        upsample_factor=4,
        lowest_feature_resolution=4,
        depth_unet_channels=128,
        grid_sample_disable_cudnn=False,

        large_gaussian_head=False,
        color_large_unet=False,
        init_sh_input_img=True,
        feature_upsampler_channels=64,
        gaussian_regressor_channels=64,

        train_depth_only=False,
    )
)