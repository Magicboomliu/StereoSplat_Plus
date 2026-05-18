# Using the seperated model for the StereoSplat Plus 
# First using the Model 1 to render the psuedo views.
# Then together with the first frame stereo views with psuedo views and feed it into the Model 2 to render the final stereo views.

stereosplat_plus_round_2_with_seperated_model_no_difix3d(){

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_2.yaml"
validator_script="validator/stereosplat_plus/posed_input_view_injected_selected_stage2.py"

pseudo_ratio="0.50 1.0"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus_two_stage_seperated_model/no_difix3d/0.5_1.0"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"


pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/stage_2_psuedo_gt_mix_training/checkpoint-134000/"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
stage_1_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/use_gt_views/checkpoint-159000/model.safetensors"


prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" "$validator_script" \
--config_path "$configs_path" \
--output_folder "$output_folder" \
--stage_1_model_path "$stage_1_model_path" \
--val_filelist "$val_filelist" \
--demo_filelist "$demo_filelist" \
--ablation_type "$ablation_type" \
--dataset_type "$dataset_type" \
--pretrained_model_path "$pretrained_model_path" \
--pseudo_ratio $pseudo_ratio \
--timestep "$timestep" \
--pretrained_diffix_model_path "$pretrained_diffix_model_path" \
--prompt "$prompt" \
# --use_diffix3d \
# --use_ref \
# --use_gt_view \
# Optional: --output_vis

}



stereosplat_plus_round_2_with_seperated_model_with_difix3d(){

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_3.yaml"
validator_script="validator/stereosplat_plus/posed_input_view_injected_selected_stage2.py"

pseudo_ratio="0.50 1.0"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus_two_stage_seperated_model/with_difix3d/0.5_1.0"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"


pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/stage_2_psuedo_gt_mix_training/checkpoint-134000/"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
stage_1_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/use_gt_views/checkpoint-159000/model.safetensors"


prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" "$validator_script" \
--config_path "$configs_path" \
--output_folder "$output_folder" \
--stage_1_model_path "$stage_1_model_path" \
--val_filelist "$val_filelist" \
--demo_filelist "$demo_filelist" \
--ablation_type "$ablation_type" \
--dataset_type "$dataset_type" \
--pretrained_model_path "$pretrained_model_path" \
--pseudo_ratio $pseudo_ratio \
--timestep "$timestep" \
--pretrained_diffix_model_path "$pretrained_diffix_model_path" \
--prompt "$prompt" \
--use_diffix3d \
--use_ref \
# --use_gt_view \
# Optional: --output_vis

}


# stereosplat_plus_round_2_with_seperated_model_no_difix3d
stereosplat_plus_round_2_with_seperated_model_with_difix3d






