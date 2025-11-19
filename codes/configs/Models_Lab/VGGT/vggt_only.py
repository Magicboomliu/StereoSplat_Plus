_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py'
    ]

# exp name
# output directionary
exp_name = "vggt_only_kitti360_stereo_224x1088"
output_dir = "/data1/zliu/feedforward_outputs_new/VGGT_Only/visualization"
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

resolution = [112, 518]
point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]

background_color=[0.0, 0.0, 0.0]
datapath = "/data1/StereoDatasets/KITTI/KITTI360"
train_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/more_sup_trainval/train_2013_05_28_drive_0000_sync.txt"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
test_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0"
camera_model='OpenCV' # select from openCV and openGL



input_type="all" # select from all, or "stereo" or "max"
max_input_views=10
pair_images=2
names_of_frames=2


depth_info_params = dict(
    use_pseudo_depth=True,
    pseudo_depth_type='NMRFStereo', # select from "MonocularDepthV2", "Metric3DV2","NMRFStereo"
    use_sparse_lidar=True
    )

dataset_params = dict(
    dataset_name="KITTI360DatasetVGGT",
    seed=seed,
    datapath=datapath,
    train_filelist=train_filelist,
    val_filelist=val_filelist,
    sequence=sequence,
    data_version=data_version,
    resolution=resolution,
    pc_range=point_cloud_range,
    batch_size_train=1,
    batch_size_val=1,
    batch_size_test=4,
    num_workers=8,
    num_workers_val=8,
    num_workers_test=4,
    depth_info_params = depth_info_params,
    camera_model=camera_model,
    input_type=input_type,
    max_input_views=max_input_views,
    pair_images=pair_images,
    names_of_frames=names_of_frames
    
)

num_cams = 2
near = 0.1
far = 1000.0


camera_args = dict(
    resolution=resolution,
    znear=near,
    zfar=far
)
train_depth_only=False
return_depth=True
max_depth=100
min_depth=0.3


vggt_pretrained_weight="/data1/zliu/foundation_model/model.pt"




loss_args = dict(
    use_depth_loss=True,
    use_pcd_loss=True,
    use_pos_enc_loss=True,
    depth_conf_loss=True,
    pcd_conf_loss=True,
    
    alpha_weight=0.5,
    
    depth_loss_weight=1.0,
    pcd_loss_weight=1.0,
    pos_enc_loss_weight=1.0,
    
    
)
