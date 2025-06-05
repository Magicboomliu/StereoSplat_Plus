Render_KITTI360_OmniScene_LFC_Views(){
cd ../../..
cd codes/validation
configs_path="/home/Desktop/Project2025/FeedStereoGS/codes/configs/OmniScene/eval/omniscene_vanilla_settings.py"
output_dir="/data/feedforward_outputs/visualization_outputs/omniscene_vanilla_settings"
load_from="/data/feedforward_outputs/output_models/OmniScene/"
validation_list="/home/Desktop/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"

# demo.txt
# val_2013_05_28_drive_0000_sync_complete.txt

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml omniscene/rendered_FLC_view_and_depth.py \
    --py-config $configs_path \
    --output_dir  $output_dir \
    --load_from $load_from \
    --validation_list $validation_list \
    --output_vis

}

Render_KITTI360_OmniScene_LFC_Views