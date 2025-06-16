Render_KITTI360_OmniScene_Render_Videos(){
cd ../../..
cd codes/validation
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/OmniScene/eval/omniscene_vanilla_settings.py"
output_dir="/data1/zliu/temp_for_0617/Vis_And_Evals/Omniscene/Metric3DV2/rendered_videos"
load_from="/data1/zliu/temp_for_0617/OmniScene/Metric3Dv2/checkpoint-45000/"
validation_list="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
# validation_list="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"

# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 accelerate launch --config-file accelerate_config_singleGPU.yaml omniscene/rendered_short_videos.py \
    --py-config $configs_path \
    --output_dir  $output_dir \
    --load_from $load_from \
    --validation_list $validation_list
    # --output_vis

}

Render_KITTI360_OmniScene_Render_Videos