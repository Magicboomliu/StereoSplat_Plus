Render_KITTI360_OmniScene_Render_Videos(){
cd ../../..
cd codes/validation
configs_path="/home/Desktop/Project2025/FeedStereoGS/codes/configs/OmniScene/eval/omniscene_vanilla_settings.py"
output_dir="/data/feedforward_outputs/visualization_outputs/omniscene_vanilla_settings/rendered_videos"
load_from="/data/feedforward_outputs/checkpoint-48000/"
validation_list="/home/Desktop/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml rendered_short_videos.py \
    --py-config $configs_path \
    --output_dir  $output_dir \
    --load_from $load_from \
    --validation_list $validation_list
    # --output_vis

}

Render_KITTI360_OmniScene_Render_Videos