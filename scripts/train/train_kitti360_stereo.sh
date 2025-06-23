TRAIN_KITTI360_OmniScene(){
cd ../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/OmniScene/omniscene_vanilla_settings.py"
work_dir="/data1/zliu/feedforward_outputs/Debug/OmniScene/First_As_Input/Baseline_Supp3_Metric3D_V2/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

# Depth Estimation Only
TRAIN_KITTI360_Unimatch_Only(){
cd ../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/unimatch_depth_only.py"
work_dir="/data/feedforward_outputs/output_models/Unimatch/depth_estimation_224x840"
resume_from="/data/feedforward_outputs/output_models/Unimatch/checkpoint-90000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_unimatch_only.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}


TRAIN_KITTI360_DepthSplat(){
cd ../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_kitti360_stereo_224x840.py"
work_dir="/data1/zliu/feedforward_outputs//depthsplatAllSupervisedFromScatch"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config-file accelerate_config.yaml train_kitti360_stereo_depthsplat.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}

TRAIN_KITTI360_VolumeFusion(){
cd ../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_configs.py"
work_dir="/data1/zliu/feedforward_outputs/VolumeFusion/Debugs"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch --config-file accelerate_config.yaml train_kitti360_stereo_volumefusion.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}

TRAIN_KITTI360_DepthSplat_Revised(){
cd ../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_revised_config.py"
work_dir="/data1/zliu/feedforward_outputs/depthsplat_revised"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 accelerate launch --config-file accelerate_config.yaml train_kitti360_stereo_depthsplat_revised.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}

TRAIN_KITTI360_VolumeFusion

# TRAIN_KITTI360_DepthSplat
# TRAIN_KITTI360_OmniScene
# TRAIN_KITTI360_Unimatch_Only
#TRAIN_KITTI360_VolumeFusion