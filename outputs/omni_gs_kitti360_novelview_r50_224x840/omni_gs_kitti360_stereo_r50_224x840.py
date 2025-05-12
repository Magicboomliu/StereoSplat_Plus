_dim_ = 128
_ffn_dim_ = 256
camera_args = dict(
    resolution=[
        224,
        840,
    ], zfar=1000.0, znear=0.1)
datapath = '/data1/StereoDatasets/KITTI/KITTI360/'
dataset_params = dict(
    batch_size_test=4,
    batch_size_train=1,
    batch_size_val=1,
    datapath='/data1/StereoDatasets/KITTI/KITTI360/',
    dataset_name='KITTI360Dataset',
    num_workers=8,
    num_workers_test=4,
    num_workers_val=8,
    pc_range=[
        -80.0,
        -80.0,
        -3.0,
        80.0,
        80.0,
        12.0,
    ],
    resolution=[
        224,
        840,
    ],
    seed=0,
    sequence='2013_05_28_drive_0000_sync',
    test_filelist=
    '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt',
    train_filelist=
    '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/train_2013_05_28_drive_0000_sync.txt',
    use_center=True,
    use_first=False,
    use_last=False,
    val_filelist=
    '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt'
)
eval_args = dict(save_ply=False, save_vis=False)
exp_name = 'omni_gs_kitti360_stereo_r50_224x804'
far = 1000.0
gpv = 3
grad_max_norm = 1.0
gradient_accumulation_steps = 1
hybrid_attn_anchors = 16
hybrid_attn_init = 0
hybrid_attn_points = 32
loss_args = dict(
    depth_abs_loss_vol_type='mask',
    mask_dptm=True,
    perceptual_loss_vol_type='mask',
    perceptual_resolution=[
        224,
        840,
    ],
    recon_loss_type='l2',
    recon_loss_vol_type='l2_mask',
    weight_depth_abs=0.01,
    weight_depth_abs_vol=0.01,
    weight_perceptual=0.05,
    weight_perceptual_vol=0.05,
    weight_recon=1.0,
    weight_recon_vol=1.0)
lr = 0.0001
lr_scheduler_type = 'constant_with_warmup'
max_epochs = 300
max_train_steps = 5000
mixed_precision = 'no'
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        in_channels=3,
        init_cfg=dict(
            checkpoint='pretrained/dino_resnet50_pretrain.pth',
            prefix=None,
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    camera_args=dict(resolution=[
        224,
        840,
    ], zfar=1000.0, znear=0.1),
    dataset_params=dict(
        batch_size_test=4,
        batch_size_train=1,
        batch_size_val=1,
        datapath='/data1/StereoDatasets/KITTI/KITTI360/',
        dataset_name='KITTI360Dataset',
        num_workers=8,
        num_workers_test=4,
        num_workers_val=8,
        pc_range=[
            -80.0,
            -80.0,
            -3.0,
            80.0,
            80.0,
            12.0,
        ],
        resolution=[
            224,
            840,
        ],
        seed=0,
        sequence='2013_05_28_drive_0000_sync',
        test_filelist=
        '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt',
        train_filelist=
        '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/train_2013_05_28_drive_0000_sync.txt',
        use_center=True,
        use_first=False,
        use_last=False,
        val_filelist=
        '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt'
    ),
    loss_args=dict(
        depth_abs_loss_vol_type='mask',
        mask_dptm=True,
        perceptual_loss_vol_type='mask',
        perceptual_resolution=[
            224,
            840,
        ],
        recon_loss_type='l2',
        recon_loss_vol_type='l2_mask',
        weight_depth_abs=0.01,
        weight_depth_abs_vol=0.01,
        weight_perceptual=0.05,
        weight_perceptual_vol=0.05,
        weight_recon=1.0,
        weight_recon_vol=1.0),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=128,
        start_level=0,
        type='mmdet.FPN'),
    pixel_gs=dict(
        down_block=dict(
            num_attention_heads=8,
            num_layers=1,
            num_views=2,
            resnet_act_fn='silu',
            resnet_groups=32,
            type='MVDownsample2D'),
        far=1000.0,
        in_embed_dim=128,
        mid_block=dict(
            num_attention_heads=8,
            num_layers=1,
            num_views=2,
            resnet_act_fn='silu',
            resnet_groups=32,
            type='MVMiddle2D'),
        near=0.1,
        num_cams=2,
        out_embed_dims=[
            128,
            256,
            512,
            512,
        ],
        patch_sizes=[
            8,
            8,
            4,
            2,
        ],
        type='PixelGaussian',
        up_block=dict(
            num_attention_heads=8,
            num_layers=1,
            num_views=2,
            resnet_act_fn='silu',
            resnet_groups=32,
            type='MVUpsample2D'),
        use_checkpoint=True),
    type='OmniGaussian',
    use_checkpoint=True,
    volume_gs=dict(
        encoder=dict(
            embed_dims=128,
            num_feature_levels=1,
            num_layers=3,
            num_points_in_pillar=[
                8,
                16,
                16,
            ],
            num_points_in_pillar_cross_view=[
                16,
                16,
                16,
            ],
            pc_range=[
                -80.0,
                -80.0,
                -3.0,
                80.0,
                80.0,
                12.0,
            ],
            positional_encoding=dict(
                h=192,
                num_feats=[
                    48,
                    48,
                    32,
                ],
                type='TPVFormerPositionalEncoding',
                w=192,
                z=16),
            return_intermediate=False,
            tpv_h=192,
            tpv_w=192,
            tpv_z=16,
            transformerlayers=[
                dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.1,
                            embed_dims=128,
                            init_mode=0,
                            num_anchors=16,
                            num_heads=8,
                            num_points=32,
                            tpv_h=192,
                            tpv_w=192,
                            tpv_z=16,
                            type='TPVCrossViewHybridAttention'),
                        dict(
                            deformable_attention=dict(
                                embed_dims=128,
                                floor_sampling_offset=False,
                                num_heads=8,
                                num_levels=1,
                                num_points=[
                                    16,
                                    32,
                                    32,
                                ],
                                num_z_anchors=[
                                    8,
                                    16,
                                    16,
                                ],
                                tpv_h=192,
                                tpv_w=192,
                                tpv_z=16,
                                type='TPVMSDeformableAttention3D'),
                            dropout=0.1,
                            embed_dims=128,
                            num_cams=6,
                            pc_range=[
                                -80.0,
                                -80.0,
                                -3.0,
                                80.0,
                                80.0,
                                12.0,
                            ],
                            tpv_h=192,
                            tpv_w=192,
                            tpv_z=16,
                            type='TPVImageCrossAttention'),
                    ],
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='TPVFormerLayer'),
                dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.1,
                            embed_dims=128,
                            init_mode=0,
                            num_anchors=16,
                            num_heads=8,
                            num_points=32,
                            tpv_h=192,
                            tpv_w=192,
                            tpv_z=16,
                            type='TPVCrossViewHybridAttention'),
                        dict(
                            deformable_attention=dict(
                                embed_dims=128,
                                floor_sampling_offset=False,
                                num_heads=8,
                                num_levels=1,
                                num_points=[
                                    16,
                                    32,
                                    32,
                                ],
                                num_z_anchors=[
                                    8,
                                    16,
                                    16,
                                ],
                                tpv_h=192,
                                tpv_w=192,
                                tpv_z=16,
                                type='TPVMSDeformableAttention3D'),
                            dropout=0.1,
                            embed_dims=128,
                            num_cams=6,
                            pc_range=[
                                -80.0,
                                -80.0,
                                -3.0,
                                80.0,
                                80.0,
                                12.0,
                            ],
                            tpv_h=192,
                            tpv_w=192,
                            tpv_z=16,
                            type='TPVImageCrossAttention'),
                    ],
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='TPVFormerLayer'),
                dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.1,
                            embed_dims=128,
                            init_mode=0,
                            num_anchors=16,
                            num_heads=8,
                            num_points=32,
                            tpv_h=192,
                            tpv_w=192,
                            tpv_z=16,
                            type='TPVCrossViewHybridAttention'),
                    ],
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='TPVFormerLayer'),
            ],
            type='TPVFormerEncoder'),
        gs_decoder=dict(
            gpv=3,
            gs_dim=14,
            hidden_dims=256,
            in_dims=128,
            offset_max=[
                1.6666666666666667,
                1.6666666666666667,
                1.875,
            ],
            out_dims=128,
            pc_range=[
                -80.0,
                -80.0,
                -3.0,
                80.0,
                80.0,
                12.0,
            ],
            scale_h=1,
            scale_max=[
                1.6666666666666667,
                1.6666666666666667,
                1.875,
            ],
            scale_w=1,
            scale_z=1,
            tpv_h=192,
            tpv_w=192,
            tpv_z=16,
            type='VolumeGaussianDecoder'),
        type='VolumeGaussian',
        use_checkpoint=True),
    volume_only=False,
    with_pixel=True)
near = 0.1
num_cams = 2
num_heads = 8
num_layers = 1
num_points = [
    16,
    32,
    32,
]
num_points_in_pillar = [
    8,
    16,
    16,
]
optimizer = dict(
    lr=5e-05,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    type='AdamW',
    weight_decay=0.01)
output_dir = 'outputs/omni_gs_kitti360_novelview_r50_224x840'
patch_sizes = [
    8,
    8,
    4,
    2,
]
pc_range = [
    -80.0,
    -80.0,
    -3.0,
    80.0,
    80.0,
    12.0,
]
pc_xrange = 160.0
pc_yrange = 160.0
pc_zrange = 15.0
point_cloud_range = [
    -80.0,
    -80.0,
    -3.0,
    80.0,
    80.0,
    12.0,
]
print_freq = 5
report_to = 'tensorboard'
resolution = [
    224,
    840,
]
resume_from = 'latest'
save_epoch_freq = -1
save_freq = 3000
scale_h = 1
scale_w = 1
scale_z = 1
seed = 0
self_cross_layer = dict(
    attn_cfgs=[
        dict(
            dropout=0.1,
            embed_dims=128,
            init_mode=0,
            num_anchors=16,
            num_heads=8,
            num_points=32,
            tpv_h=192,
            tpv_w=192,
            tpv_z=16,
            type='TPVCrossViewHybridAttention'),
        dict(
            deformable_attention=dict(
                embed_dims=128,
                floor_sampling_offset=False,
                num_heads=8,
                num_levels=1,
                num_points=[
                    16,
                    32,
                    32,
                ],
                num_z_anchors=[
                    8,
                    16,
                    16,
                ],
                tpv_h=192,
                tpv_w=192,
                tpv_z=16,
                type='TPVMSDeformableAttention3D'),
            dropout=0.1,
            embed_dims=128,
            num_cams=6,
            pc_range=[
                -80.0,
                -80.0,
                -3.0,
                80.0,
                80.0,
                12.0,
            ],
            tpv_h=192,
            tpv_w=192,
            tpv_z=16,
            type='TPVImageCrossAttention'),
    ],
    feedforward_channels=256,
    ffn_dropout=0.1,
    operation_order=(
        'self_attn',
        'norm',
        'cross_attn',
        'norm',
        'ffn',
        'norm',
    ),
    type='TPVFormerLayer')
self_layer = dict(
    attn_cfgs=[
        dict(
            dropout=0.1,
            embed_dims=128,
            init_mode=0,
            num_anchors=16,
            num_heads=8,
            num_points=32,
            tpv_h=192,
            tpv_w=192,
            tpv_z=16,
            type='TPVCrossViewHybridAttention'),
    ],
    feedforward_channels=256,
    ffn_dropout=0.1,
    operation_order=(
        'self_attn',
        'norm',
        'ffn',
        'norm',
    ),
    type='TPVFormerLayer')
sequence = '2013_05_28_drive_0000_sync'
test_filelist = '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt'
tpv_h_ = 192
tpv_w_ = 192
tpv_z_ = 16
train_filelist = '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/train_2013_05_28_drive_0000_sync.txt'
use_center = True
use_checkpoint = True
use_first = False
use_last = False
val_filelist = '/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt'
val_freq = 1000
volume_only = False
warmup_steps = 1000
work_dir = '/home/zliu/Project2025/FeedStereoGS/outputs/omni_gs_kitti360_novelview_r50_224x840'
