TRAIN_KITTI360_VolumeFusion_Center_LiDAR_As_Ref_No_Offset(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/Models_Lab/VolumeFusion/VolumeFusion/volumefusion_center_lidar_as_ref_no_offset.py"
work_dir="/gs/FeedForwardGS_New/VolumeFusion/CCenter_LiDAR_As_Ref_No_Offset/saved_models"
#resume_from="/data1/zliu/temp_for_0617/VolumeFusion/Current_Version/checkpoint-90000/"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_volumefusion.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}



TRAIN_KITTI360_VolumeFusion_First_Cam_As_Ref_No_Offset(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/Models_Lab/VolumeFusion/VolumeFusion/volumefusion_first_cam_as_ref_no_offset.py"
work_dir="/gs/FeedForwardGS_New/VolumeFusion/First_Cams_As_Ref_No_Offset/saved_models"
# resume_from="/data1/zliu/temp_for_0617/VolumeFusion/Current_Version/checkpoint-90000/"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_volumefusion.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}


TRAIN_KITTI360_VolumeFusion_First_LiDAR_As_Ref_No_Offset(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/Models_Lab/VolumeFusion/VolumeFusion/volumefusion_first_lidar_as_ref_no_offset.py"
work_dir= "/gs/FeedForwardGS_New/VolumeFusion/First_LiDAR_As_Ref_No_Offset/saved_models"
# resume_from="/data1/zliu/temp_for_0617/VolumeFusion/Current_Version/checkpoint-90000/"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_volumefusion.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}





TRAIN_KITTI360_VolumeFusion_First_LiDAR_As_Ref_No_Offset
#TRAIN_KITTI360_VolumeFusion_First_Cam_As_Ref_No_Offset
#TRAIN_KITTI360_VolumeFusion_Center_LiDAR_As_Ref_No_Offset
