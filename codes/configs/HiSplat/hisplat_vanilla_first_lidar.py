_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py'
    ]


exp_name = "hisplat_vanilla"
output_dir ="/data/zliu/feedforward_outputs_revision/HiSplat_2Views/First_LiDAR_As_Ref/visualization"
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
seed=1024

# only using the center for training
use_center, use_first, use_last = False, True, False
resolution = [112, 544]

# LiDAR Range id different
point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]

background_color=[0.0, 0.0, 0.0]
datapath = "/data/StereoDatasets/KITTI/KITTI360"
train_filelist="/home/Project2025/FeedStereoGS/filenames/kitti360/train_complete/all.txt"
val_filelist="/home/Project2025/FeedStereoGS/filenames/kitti360/train_complete/demo.txt"
test_filelist="/home/Project2025/FeedStereoGS/filenames/kitti360/train_complete/demo.txt"
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
max_depth=1000
min_depth=0.01


unimatch_weights_path = "/home/Project2025/pretrained_tools_models/checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth"
vit_path ="/home/Project2025/pretrained_tools_models/checkpoints/dinov2_vitb14_pretrain.pth"



loss_settings_dict = dict(
    depth_estimator_supervision=True,
    depth_estimator_suppervision_type='sparse_gt_pseudo', # 'sparse_gt', 'pseudo', 'sparse_gt_pseudo'
    rendered_depth_supervision=True,
    rendered_depth_supervision_type='sparse_gt_pseudo', # 'sparse_gt', 'pseudo', 'sparse_gt_pseudo'
    
    rendered_rgb_supervision=True,
    rendered_rgb_supervison_type="MSE_LPIPS",
    lpips_alpha=0.05,
    rendered_depth_weight=0.15,
)

