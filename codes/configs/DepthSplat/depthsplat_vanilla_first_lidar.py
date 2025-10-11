_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py'
    ]

# exp name
# output directionary
exp_name = "depthsplat_vanilla"
output_dir ="/data1/zliu/feedforward_outputs_revision/DepthSplat2Views/First_LiDAR_As_Ref/visualization"
validation_vis_progress=True


# learning rate setiing
lr = 1e-4
grad_max_norm = 1.0
print_freq = 1
save_freq = 3000
val_freq = 3000
max_epochs = 150
save_epoch_freq = -1

lr_scheduler_type = "constant_with_warmup"
max_train_steps = 100000
warmup_steps = 1000
mixed_precision = "no"
gradient_accumulation_steps = 1
resume_from = "latest"
report_to = "tensorboard"

seed=42

# only using the center for training
use_center, use_first, use_last = False, True, False
resolution = [112, 544]

# LiDAR Range id different
point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]

background_color=[0.0, 0.0, 0.0]
datapath = "/data1/StereoDatasets/KITTI/KITTI360"
train_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/train_complete/all.txt"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/train_complete/demo.txt"
test_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/train_complete/demo.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstLIDAR"
supp_view_nums=6
#unimatch_weights_path="/data1/zliu/feedforward_outputs/DepthSplat/Depth_Estimation_Only/depth_estimation_224x840/checkpoint-90000/model.safetensors"
unimatch_weights_path=None
camera_model='OpenCV' # select from openCV and openGL
world_center="First_LiDAR" # Select from "Center_LiDAR" or "First_Cam0" or "First_LiDAR"

depth_info_params = dict(
    use_pseudo_depth=True,
    pseudo_depth_type='NMRFStereo', # select from "MonocularDepthV2", "Metric3DV2","NMRFStereo"
    use_sparse_lidar=True
    )


dataset_params = dict(
    dataset_name="KITTI360Dataset",
    seed=seed,
    datapath=datapath,
    train_filelist=train_filelist,
    val_filelist=val_filelist,
    test_filelist=test_filelist,
    sequence=sequence,
    data_version=data_version,
    resolution=resolution,
    pc_range=point_cloud_range,
    use_center=use_center,
    use_first=use_first,
    use_last=use_last,
    batch_size_train=1,
    batch_size_val=1,
    batch_size_test=4,
    num_workers=8,
    num_workers_val=8,
    num_workers_test=4,
    supp_view_nums=supp_view_nums,
    depth_info_params = depth_info_params,
    camera_model=camera_model
)

num_cams = 2
near = 0.1
far = 1000.0

# define the 3D Space including the 
# image resolution
# Z-Near Planar
# Z-Far Planar
camera_args = dict(
    resolution=resolution,
    znear=near,
    zfar=far
)
train_depth_only=False
return_depth=True
max_depth=100
min_depth=0.3

# Define the Models
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
            gaussian_scale_max=1.5,
            sh_degree=2,
        ),

        d_feature=128,

        visualizer=dict(
            num_samples=8,
            min_resolution=256,
            export_ply=False,
        ),

        # loaded the unimatch_weaight
        unimatch_weights_path=unimatch_weights_path,
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

        
    )
)



loss_settings_dict = dict(
    
    depth_estimator_supervision=True,
    depth_estimator_suppervision_type='sparse_gt_pseudo', # 'sparse_gt', 'pseudo', 'sparse_gt_pseudo'
    rendered_depth_supervision=True,
    rendered_depth_supervision_type='sparse_gt_pseudo', # 'sparse_gt', 'pseudo', 'sparse_gt_pseudo'
    
    rendered_rgb_supervision=True,
    rendered_rgb_supervison_type="MSE_LPIPS",
    lpips_alpha=0.05,
    rendered_depth_weight=0.05,
    depth_estimation_weight=0.1,
)