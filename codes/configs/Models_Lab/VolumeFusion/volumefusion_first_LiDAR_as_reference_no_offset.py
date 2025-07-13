_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py'
    ]

# exp name
# output directionary
exp_name = "volumefusion_kitti360_stereo_224x1088"
output_dir = "/data1/zliu/feedforward_outputs_new/VolumeFusion/First_LiDAR_As_Ref_No_Offset/visualization"
validation_vis_progress=True

# learning rate setiing
lr = 8e-5
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
# resolution = [224, 832]
resolution = [224, 1088]
# resolution = [224, 544] #FIXME Here


# LiDAR Range id different
point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]
background_color=[0.0, 0.0, 0.0]
datapath = "/data1/StereoDatasets/KITTI/KITTI360"
train_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/all_2013_05_28_drive_0000_sync.txt"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
test_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstCAM"
supp_view_nums=3
world_center="First_LiDAR" # Select from "Center_LiDAR" or "First_Cam0" or "First_LiDAR"
# if neccssary
unimatch_weights_path="/data1/zliu/feedforward_outputs/DepthSplat/Depth_Estimation_Only/depth_estimation_224x840/checkpoint-90000/model.safetensors"
#unimatch_weights_path=None
camera_model='OpenCV' # select from openCV and openGL
used_3D_offset=False



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


# Volume Branch Parameterization

pc_range = point_cloud_range # [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]
# x range, y_range and z_range in the LiDAR coordiante
pc_xrange, pc_yrange, pc_zrange = pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]

_dim_ = 128
num_heads = 8
num_layers = 1
patch_sizes=[8, 8, 4, 2]
_ffn_dim_ = _dim_ * 2

# unit is a little bigger than the x_range,y_range and the z_range
tpv_h_ = 192
tpv_w_ = 192
tpv_z_ = 16
scale_h = 1
scale_w = 1
scale_z = 1
gpv = 3

# for filering
num_points_in_pillar = [8, 16, 16]
num_points = [16, 32, 32]

hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0


# cross layer
self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
            dropout=0.1),
        dict(
            type='TPVImageCrossAttention',
            pc_range=point_cloud_range,
            num_cams=2,
            dropout=0.1,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=1,
                floor_sampling_offset=False,
                tpv_h=tpv_h_,
                tpv_w=tpv_w_,
                tpv_z=tpv_z_),
            embed_dims=_dim_,
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
            dropout=0.1)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm'))



return_types = ["gs",'depth','feature']
# define the model 
 
use_checkpoint = True

loss_args = dict(
    use_volume=True,
    depth_estimation=True,
    use_fusion=True,
    use_cv=False,
    perceptual_resolution=[resolution[0], resolution[1]], # using the current resolustion
    
    gt_depth_type = 'sparse_pseudo', # select from 'sparse', 'pseudo','sparse_pseudo'
    
    depth_est_sup_dict= dict(
        branch_weight = 0.05,
        loss_type ='L2' # select from L2, L1 and DPM Loss
    ),
    
    
    volume_sup_dict = dict(
        recon_loss_vol_type="l2_mask", # masked reconstruction loss for volume gaussains,
        perceptual_loss_vol_type="mask", # prcepstion loss? SSIM Loss using masked
        depth_abs_loss_vol_type="mask", # depth abstract loss
        weight_recon_vol=1.0,
        weight_perceptual_vol=0.05,
        weight_depth_abs_vol=0.01,
        branch_weight = 1.0
        
    ),
    
    fusion_sup_dict = dict(
        recon_loss_type="l2", # reconstrunction loss
        weight_recon=1.0,
        weight_perceptual=0.05,
        weight_depth_abs=0.01,
        branch_weight =1.0,
    ),
    
    cv_sup_dict = dict(
        recon_loss_cv_type="l2", # reconstrunction loss
        weight_recon_cv=1.0,
        weight_perceptual_cv=0.05,
        weight_depth_abs_cv=0.01,
        branch_weight =1.0,
        
    ),
)


model = dict(
    type='VolumeFusion',
    use_checkpoint=use_checkpoint,
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/dino_resnet50_pretrain.pth',
            prefix=None)),
    
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    
    # depthsplat-like: cost-volume-based
    costvolume_gs = dict(
        depth_estimator_kwargs = dict(
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
                
        ),

        gaussain_head_kwargs = dict(
            monodepth_vit_type='vitb',
            upsample_factor=4,
            num_scales=1,
            gaussian_regressor_channels=64
        ),
    ),
    
    volume_gs=dict(
        use_checkpoint=use_checkpoint,
        
        encoder=dict(
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_feature_levels=1,
            num_layers=3,
            pc_range=point_cloud_range,
            num_cams=num_cams,
            num_points_in_pillar=num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                self_cross_layer, self_cross_layer, self_layer
            ],
            embed_dims=_dim_,
            positional_encoding=dict(
                type='TPVFormerPositionalEncoding',
                num_feats=[48, 48, 32],
                h=tpv_h_,
                w=tpv_w_,
                z=tpv_z_)),
        
        gs_decoder = dict(
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            pc_range=point_cloud_range,
            gs_dim=14,
            in_dims=_dim_,
            hidden_dims=2*_dim_,
            out_dims=_dim_,
            scale_h=scale_h,
            scale_w=scale_w,
            scale_z=scale_z,
            gpv=gpv,
            offset_max=[2 * pc_xrange / (tpv_h_*scale_h), 2 * pc_yrange / (tpv_w_*scale_w), 2 * pc_zrange / (tpv_z_*scale_z)],
            scale_max=[2 * pc_xrange / (tpv_h_*scale_h), 2 * pc_yrange / (tpv_w_*scale_w), 2 * pc_zrange / (tpv_z_*scale_z)]
        )
    ),
    
    
    losses_params=loss_args,
    camera_args=camera_args,
    dataset_params=dataset_params
    
)