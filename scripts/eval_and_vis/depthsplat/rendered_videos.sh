Rendered_FLC_Views_And_Depths(){
cd ../../..
cd codes/validation
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/eval/depthsplat_gs_kitti360_stereo_224x840.py"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
#val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
work_dir="/data1/zliu/feedforward_outputs/DepthSplat/Temp_Visualization/After_State/"
resume_from="/data1/zliu/feedforward_outputs/DepthSplat/depthsplat_fully_supervised/checkpoint-36000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml depthsplat/render_video.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from \
    --val_filelist $val_filelist \
    # --output_vis
}

Rendered_FLC_Views_And_Depths