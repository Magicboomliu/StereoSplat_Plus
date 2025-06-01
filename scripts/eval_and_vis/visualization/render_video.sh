Render_KITTI360_OmniScene_Render_Videos(){
cd ../../..
cd codes/validation
configs_path="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/codes/configs/OmniScene/eval/vanilla_settings_nmrfstereo_depth_supp6.py"
output_dir="/data1/zliu/feedforward_outputs/Vanilla_Omni_Scene/Visualizations_And_Evaluations/20250531/Vanilla_Settings_NMRFStereo_Depth_Supp6/rendered_videos"
load_from="/data1/zliu/feedforward_outputs/Vanilla_Omni_Scene/NMRFStereo_Based/6_View_Supp/checkpoint-45000/"
validation_list="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
#configs 
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