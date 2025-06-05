_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py'
    ]

# exp name
# output directionary
exp_name = "depthsplat_kitti360_stereo_224x840"
output_dir = "outputs/depthsplat_kitti360_stereo_224x840"

# learning rate setiing
lr = 2e-4
grad_max_norm = 1.0
print_freq = 1
save_freq = 3000
val_freq = 2000
max_epochs = 150
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
use_center, use_first, use_last = False, True, False
resolution = [224, 840]

# LiDAR Range id different
point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]

datapath = "/data/KITTI/KITTI360_For_docker"
train_filelist="/home/Desktop/Project2025/FeedStereoGS/filenames/kitti360/more_sup_trainval/train_2013_05_28_drive_0000_sync.txt"
val_filelist="/home/Desktop/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
test_filelist="/home/Desktop/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0"
supp_view_nums=3


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
    depth_info_params = depth_info_params
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
