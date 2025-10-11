TRAIN_KITTI360_DepthSplat_Vanilla(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_vanilla_first_lidar.py"
work_dir="/data1/zliu/feedforward_outputs_revision/DepthSplat2Views/First_LiDAR_As_Ref/saved_models"
resume_from="None"
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

TRAIN_KITTI360_DepthSplat_Vanilla