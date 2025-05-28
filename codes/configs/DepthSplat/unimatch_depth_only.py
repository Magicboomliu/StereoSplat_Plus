_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py'
    ]

# exp name
# output directionary
exp_name = "depthsplat_depth_estimation_only_only"
output_dir = "outputs/depthsplat_depth_estimation_only_224x840"

# learning rate setiing
lr = 2e-4
grad_max_norm = 1.0
print_freq = 1
save_freq = 3000
val_freq = 2000
max_epochs = 50
save_epoch_freq = -1

lr_scheduler_type = "constant_with_warmup"
max_train_steps = 50000
warmup_steps = 1000
mixed_precision = "no"
gradient_accumulation_steps = 1
resume_from = "latest"
report_to = "tensorboard"
seed=42

# only using the center for training
resolution = [224, 832]
datapath = "/data1/StereoDatasets/KITTI/KITTI360/"
train_filelist="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/depth_estimation_trainval/train.txt"
val_filelist="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/depth_estimation_trainval/val.txt"
test_filelist="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/depth_estimation_trainval/val.txt"
use_projected_lidar=True
use_pseudo_depth=True

# Dataset Configuration
dataset_params = dict(
    dataset_name="KITTI360Dataset",
    seed=seed,
    datapath=datapath,
    train_filelist=train_filelist,
    val_filelist=val_filelist,
    test_filelist=test_filelist,
    resolution=resolution,
    batch_size_train=1,
    batch_size_val=1,
    batch_size_test=4,
    num_workers=8,
    num_workers_val=8,
    num_workers_test=4,
    use_projected_lidar =use_projected_lidar,
    use_pseudo_depth=use_pseudo_depth
)

min_depth=0.3
max_depth=100

# Model Part: Definition
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

        monodepth_vit_type='vitb',

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