Train_KITTI360_PixelSplat_Vanilla(){
cd ../../..
cd codes
configs_path="/home/Project2025/FeedStereoGS/codes/configs/PixelSplat/pixelsplat_vanilla_first_lidar.py"
work_dir="/data/zliu/feedforward_outputs_revision/PixelSplat_2Views/First_LiDAR_As_Ref/saved_models"
resume_from="/data/zliu/feedforward_outputs_revision/PixelSplat_2Views/First_LiDAR_As_Ref/saved_models/checkpoint-60000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch --config-file accelerate_config.yaml train_kitti360_pixelsplat_vanilla.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}

Train_KITTI360_PixelSplat_Vanilla