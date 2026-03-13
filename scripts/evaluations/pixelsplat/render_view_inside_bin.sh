Get_Rendered_RGBS_Depths_Metrics(){
cd ../../..
cd codes/Validation

configs_path="/home/IROS2026/Diff-StereoSplat/codes/configs/PixelSplat/pixelsplat_vanilla_first_lidar.py"
output_folder="/data/zliu/IROS26/Compared_With_Others/PixelSplat_2Views/Feedforward_Views"
val_filelist="/home/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/all_sequential.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR"
pretrained_model_path="/data/zliu/feedforward_outputs_revision/PixelSplat_2Views/First_LiDAR_As_Ref/saved_models/checkpoints/"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml pixelsplat/rendered_views_inside_bin.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --output_vis

}

Get_Rendered_RGBS_Depths_Metrics