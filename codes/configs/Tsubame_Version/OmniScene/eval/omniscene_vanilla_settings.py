_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

# exp name
# output directionary
exp_name = "omni_gs_kitti360_stereo_r50_224x804"
output_dir = "outputs/omni_gs_kitti360_novelview_r50_224x840"


# learning rate setiing
lr = 8e-5
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

# using the volume 
volume_only = False
use_checkpoint = True
seed = 0

# only using the center for training
use_center, use_first, use_last = False, True, False
# resolution = [224, 840]
# resolution = [168, 632]
resolution = [224, 1088]

# LiDAR Range id different
point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]
datapath = "/gs/KITTI360_For_Upload"
train_filelist="/home/2/ux04482/FeedStereoGS/filenames/kitti360/more_sup_trainval/train_2013_05_28_drive_0000_sync.txt"
val_filelist="/home/2/ux04482/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
test_filelist="/home/2/ux04482/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0"
supp_view_nums=3
camera_model='OpenGL' # select from OpenCV and OpenGL



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

eval_args = dict(
    save_vis=False,
    save_ply=False
)

# loss functions
loss_args = dict(
    recon_loss_type="l2", # reconstrunction loss
    recon_loss_vol_type="l2_mask", # masked reconstruction loss for volume gaussains
    perceptual_loss_vol_type="mask", # prcepstion loss? SSIM Loss using masked
    depth_abs_loss_vol_type="mask", # depth abstract loss
    mask_dptm=True,                 # whether using the mask dptm(metric3d-v2 as examples)
    perceptual_resolution=[resolution[0], resolution[1]], # using the current resolustion
    weight_recon=1.0,
    weight_perceptual=0.05,
    weight_depth_abs=0.01,
    weight_recon_vol=1.0,
    weight_perceptual_vol=0.05,
    weight_depth_abs_vol=0.01,
)

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


# model definition
model = dict(
    type='OmniGaussian',
    use_checkpoint=use_checkpoint,
    with_pixel=True,
    volume_only=volume_only,
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
            checkpoint='/home/Desktop/Project2025/FeedStereoGS/codes/pretrained/dino_resnet50_pretrain.pth',
            prefix=None)),
    
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    
    pixel_gs=dict(
        type="PixelGaussian",
        use_checkpoint=use_checkpoint,
        down_block=dict(
            type='MVDownsample2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        
        up_block=dict(
            type='MVUpsample2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        mid_block=dict(
            type='MVMiddle2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        patch_sizes=patch_sizes,
        in_embed_dim=_dim_,
        out_embed_dims=[_dim_, _dim_*2, _dim_*4, _dim_*4],
        num_cams=num_cams,
        near=near,
        far=far),
    
    
    volume_gs=dict(
        type="VolumeGaussian",
        use_checkpoint=use_checkpoint,
        encoder=dict(
            type='TPVFormerEncoder',
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
            type='VolumeGaussianDecoder',
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
    camera_args=camera_args,
    loss_args=loss_args,
    dataset_params=dataset_params
)