TRAIN_KITTI360_OmniScene_Center_LiDAR_As_Ref_Metric3DV2(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/OmniScene/omni_gs_center_lidar_as_ref_MetircV2.py"
work_dir="/gs/FeedForwardGS_New/OmniScene/Center_LiDAR_As_Ref/Metric3DV2/saved_models"
resume_from="None"
# resume_from="/data1/zliu/temp_for_0617/OmniScene/Metric3Dv2/checkpoint-45000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

TRAIN_KITTI360_OmniScene_First_Cam_As_Ref_Metric3DV2(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/OmniScene/omni_gs_first_cam_as_ref_MetricV2.py"
work_dir="/gs/FeedForwardGS_New/OmniScene/First_Cam_As_Ref/Metric3DV2/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}


TRAIN_KITTI360_OmniScene_First_LiDAR_As_Ref_Metric3DV2(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/OmniScene/omni_gs_first_lidar_as_ref_MetricV2.py"
work_dir="/gs/FeedForwardGS_New/OmniScene/First_LiDAR_As_Ref/Metric3DV2/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}



TRAIN_KITTI360_OmniScene_Center_LiDAR_As_Ref_NMRFStereo(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/OmniScene/omni_gs_center_lidar_as_ref_NMRFStereo.py"
work_dir="/gs/FeedForwardGS_New/OmniScene/Center_LiDAR_As_Ref/Stereo/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

TRAIN_KITTI360_OmniScene_First_Cam_As_Ref_NMRFStereo(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/OmniScene/omni_gs_first_cam_as_ref_NMRFStereo.py"
work_dir="/gs/FeedForwardGS_New/OmniScene/First_Cam_As_Ref/Stereo/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}


TRAIN_KITTI360_OmniScene_First_LiDAR_As_Ref_NMRFStereo(){
cd ../../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/OmniScene/omni_gs_first_lidar_as_ref_NMRFStereo.py"
work_dir="/gs/FeedForwardGS_New/OmniScene/First_LiDAR_As_Ref/Stereo/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

TRAIN_KITTI360_OmniScene_First_LiDAR_As_Ref_NMRFStereo
TRAIN_KITTI360_OmniScene_First_LiDAR_As_Ref_Metric3DV2

#TRAIN_KITTI360_OmniScene_Center_LiDAR_As_Ref_Metric3DV2
#TRAIN_KITTI360_OmniScene_First_Cam_As_Ref_Metric3DV2
#TRAIN_KITTI360_OmniScene_Center_LiDAR_As_Ref_NMRFStereo
# TRAIN_KITTI360_OmniScene_First_Cam_As_Ref_NMRFStereo