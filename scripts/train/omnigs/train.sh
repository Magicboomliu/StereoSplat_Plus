TRAIN_KITTI360_OmniScene_Center_LiDAR_As_Ref_Metric3DV2(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/OmniScene/omnigs_vanilla_center_LiDAR_as_reference.py"
work_dir="/data1/zliu/feedforward_outputs_new/OmniScene/MetricV2/Center_LiDAR_As_Ref/saved_models"
resume_from="None"
# resume_from="/data1/zliu/temp_for_0617/OmniScene/Metric3Dv2/checkpoint-45000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

TRAIN_KITTI360_OmniScene_First_Cam_As_Ref_Metric3DV2(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/OmniScene/omnigs_vanilla_first_cam_as_reference.py"
work_dir="/data1/zliu/feedforward_outputs_new/OmniScene/MetricV2/First_Cam_As_Ref/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}


TRAIN_KITTI360_OmniScene_Center_LiDAR_As_Ref_NMRFStereo(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/OmniScene/omnigs_NMRF_center_LiDAR_as_reference.py"
work_dir="/data1/zliu/feedforward_outputs_new/OmniScene/Stereo/Center_LiDAR_As_Ref/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

TRAIN_KITTI360_OmniScene_First_Cam_As_Ref_NMRFStereo(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/OmniScene/omnigs_NMRF_first_cam_as_reference.py"
work_dir="/data1/zliu/feedforward_outputs_new/OmniScene/Stereo/First_Cam_As_Ref/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}




#TRAIN_KITTI360_OmniScene_Center_LiDAR_As_Ref_Metric3DV2
# TRAIN_KITTI360_OmniScene_First_Cam_As_Ref_Metric3DV2

# TRAIN_KITTI360_OmniScene_Center_LiDAR_As_Ref_NMRFStereo
TRAIN_KITTI360_OmniScene_First_Cam_As_Ref_NMRFStereo