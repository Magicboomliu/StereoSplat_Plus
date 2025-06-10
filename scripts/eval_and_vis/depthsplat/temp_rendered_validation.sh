Rendered_FLC_Views_And_Depths(){
cd ../../..
cd codes/validation
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_kitti360_stereo_224x840.py"
work_dir="/data1/zliu/feedforward_outputs/DepthSplat/Temp_Visualization/"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml depthsplat/temp_baseline_validation.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}

Rendered_FLC_Views_And_Depths