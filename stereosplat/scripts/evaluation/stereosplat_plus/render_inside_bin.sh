StereoSplat_Plus_Without_Difix3D() {
cd ../../..
accelerate_config_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/accelerate_configs/inference/gpu_0.yaml"
validator_script="validator/stereosplat/rendered_view_inside_bin_plus_diffix.py"
# mmengine 配置（仓库 codes 下；与修改后的 Python 中 Difix 的 codes 路径一致）
config_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus_two_stage/no_difix3d/"
val_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
# pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/2_4_6_View_Variances/checkpoint-51000/"
pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/stage_2_psuedo_gt_mix_training/checkpoint-134000/"
# diffix3d
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" "$validator_script" \
--config_path "$config_path" \
--output_folder "$output_folder" \
--val_filelist "$val_filelist" \
--demo_filelist "$demo_filelist" \
--ablation_type "$ablation_type" \
--dataset_type "$dataset_type" \
--pretrained_model_path "$pretrained_model_path" \
--pretrained_diffix_model_path "$pretrained_diffix_model_path" \
--timestep "$timestep" \
--prompt "$prompt" 
# --output_vis 
}


StereoSplat_Plus_Difix3D_Old() {
cd ../../..
accelerate_config_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/accelerate_configs/inference/gpu_0.yaml"
validator_script="validator/stereosplat/rendered_view_inside_bin_plus_diffix.py"
# mmengine 配置（仓库 codes 下；与修改后的 Python 中 Difix 的 codes 路径一致）
config_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus/default_manner_old_finetuned/"
val_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

# pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/2_4_6_View_Variances/checkpoint-51000/"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"

# diffix3d
pretrained_diffix_model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"

prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" "$validator_script" \
--config_path "$config_path" \
--output_folder "$output_folder" \
--val_filelist "$val_filelist" \
--demo_filelist "$demo_filelist" \
--ablation_type "$ablation_type" \
--dataset_type "$dataset_type" \
--pretrained_model_path "$pretrained_model_path" \
--pretrained_diffix_model_path "$pretrained_diffix_model_path" \
--timestep "$timestep" \
--prompt "$prompt" \
--use_diffix3d \
--use_ref \
--deterministic_vae_encode \
--deterministic_scheduler_step \
# --output_vis 

}




StereoSplat_Plus_Difix3D_New(){

cd ../../..

accelerate_config_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/accelerate_configs/inference/gpu_1.yaml"
validator_script="validator/stereosplat/rendered_view_inside_bin_plus_diffix.py"

# mmengine 配置（仓库 codes 下；与修改后的 Python 中 Difix 的 codes 路径一致）
config_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/Mix_Training_With_Difix3D/stereosplat_plus_two_stage/with_lastest_difix3d/"
val_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/stage_2_psuedo_gt_mix_training_with_difix3d/lastest_checkpoints/checkpoint-250000/"
#pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"

# diffix3d
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" "$validator_script" \
--config_path "$config_path" \
--output_folder "$output_folder" \
--val_filelist "$val_filelist" \
--demo_filelist "$demo_filelist" \
--ablation_type "$ablation_type" \
--dataset_type "$dataset_type" \
--pretrained_model_path "$pretrained_model_path" \
--pretrained_diffix_model_path "$pretrained_diffix_model_path" \
--timestep "$timestep" \
--prompt "$prompt" \
--use_diffix3d \
--use_ref 
# --output_vis 

}

# StereoSplat_Plus_Without_Difix3D
StereoSplat_Plus_Difix3D_New
