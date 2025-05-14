Render_KITTI360_OmniScene_LFC_Views(){
cd ../../..
cd codes/validation
configs_path="/home/zliu/Desktop/Project2025/FeedStereoGS/codes/configs/OmniScene/omni_gs_kitti360_stereo_r50_224x840.py"
output_dir="/home/zliu/Desktop/Project2025/FeedStereoGS/temp/feedstereo_outputs/omni_gs_kitti360_novelview_r50_224x840"
load_from="/media/zliu/data12/outputs/omni_gs_kitti360_novelview_r50_224x840/omni_gs_kitti360_stereo_r50_224x804/pretrain/checkpoint-12000/"
validation_list="/home/zliu/Desktop/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch rendered_FLC_view_and_depth.py \
    --py-config $configs_path \
    --output_dir  $output_dir \
    --load_from $load_from \
    --validation_list $validation_list
    # --output_vis

}

Render_KITTI360_OmniScene_LFC_Views