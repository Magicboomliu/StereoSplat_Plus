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
use_center, use_first, use_last = True, False, False
resolution = [224, 840]

# LiDAR Range id different
point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]

datapath = "/data1/StereoDatasets/KITTI/KITTI360/"
train_filelist="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/trainval/train_2013_05_28_drive_0000_sync.txt"
#train_filelist="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/trainval/small_train.txt"
val_filelist="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
test_filelist="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0"
supp_view_nums=3
use_stereo=False

# Dataset Configuration
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
    use_stereo=use_stereo
)
