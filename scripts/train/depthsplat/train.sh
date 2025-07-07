TRAIN_KITTI360_DepthSplat_Vanilla_Center_LiDAR_As_Ref(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_origin_center_lidar_as_ref.py"
work_dir="/data1/zliu/feedforward_outputs_new/depthsplat_vanilla/saved_models/"
resume_from="/data1/zliu/feedforward_outputs/DepthSplat/depthsplat_fully_supervised/checkpoint-147000/"
#resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch --config-file accelerate_config.yaml train_kitti360_depthsplat_vanilla.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}


TRAIN_KITTI360_DepthSplat_Revised_Center_LiDAR_As_Ref_No_3D_Offset(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_revised_center_lidar_as_ref_no_offset.py"
work_dir="/data1/zliu/feedforward_outputs_new/depthsplat_revised_center_lidar_as_ref_no_offset/saved_models"
resume_from="/data1/zliu/temp_for_0617/DepthSplat/SH0_Version/checkpoint-99000/"
# resume_from="None"
#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 accelerate launch --config-file accelerate_config.yaml train_kitti360_depthsplat_revised.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}

TRAIN_KITTI360_DepthSplat_Revised_Center_LiDAR_As_Ref_With_3D_Offset(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_revised_center_lidar_as_ref_with_offset.py"
work_dir="/data1/zliu/feedforward_outputs_new/depthsplat_revised_center_lidar_as_ref_with_offset/saved_models"
resume_from="/data1/zliu/temp_for_0617/DepthSplat/SH0_Version/checkpoint-99000/"
# resume_from="None"
#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 accelerate launch --config-file accelerate_config.yaml train_kitti360_depthsplat_revised.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}



TRAIN_KITTI360_DepthSplat_Revised_Center_LiDAR_As_Ref_With_3D_Offset
# TRAIN_KITTI360_DepthSplat_Revised_Center_LiDAR_As_Ref_No_3D_Offset
# TRAIN_KITTI360_DepthSplat_Vanilla_Center_LiDAR_As_Ref