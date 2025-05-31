Render_KITTI360_OmniScene_LFC_Views(){
cd ../../..
cd codes/validation
configs_path="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/codes/configs/OmniScene/eval/vanilla_settings_nmrfstereo_depth_dynamic_input_train_supp6.py"
output_dir="/data1/zliu/feedforward_outputs/Vanilla_Omni_Scene/Visualizations_And_Evaluations/20250531/Vanilla_Settings_NMRFStereo_Depth_Dynamic_Input_Train_Supp6"
load_from="/data1/zliu/feedforward_outputs/Vanilla_Omni_Scene/NMRFStereo_Based/adaptive_input_training/omni_gs_kitti360_novelview_r50_224x840/checkpoint-48000/"
validation_list="/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"

# demo.txt
# val_2013_05_28_drive_0000_sync_complete.txt

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 accelerate launch --config-file accelerate_config_singleGPU.yaml rendered_FLC_view_and_depth.py \
    --py-config $configs_path \
    --output_dir  $output_dir \
    --load_from $load_from \
    --validation_list $validation_list \
    # --output_vis

}

Render_KITTI360_OmniScene_LFC_Views