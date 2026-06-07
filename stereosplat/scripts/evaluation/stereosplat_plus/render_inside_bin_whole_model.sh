# StereoSplat+ unified-model progressive inference (single checkpoint, two forward passes in-code)
# 15D conf: same stage2 mix-training weights as separated-model eval

StereoSplat_Plus_Without_Difix3D() {
cd ../../..

REPO="/home/zliu/IROS2026/Conf/StereoSplat_Plus/stereosplat"

accelerate_config_path="${REPO}/accelerate_configs/inference/gpu_0.yaml"
validator_script="validator/stereosplat/rendered_view_inside_bin_plus_diffix.py"
config_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus_two_stage/with_conf/stereosplat_plus_unified_model/no_difix3d/"
val_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${REPO}/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

STAGE2_MODEL_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_resume"
pretrained_model_path="${STAGE2_MODEL_DIR}/latest"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH}"

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
# Optional: --output_vis  (saves rendered_conf/)

}



StereoSplat_Plus_With_Difix3D(){

cd ../../..

REPO="/home/zliu/IROS2026/Conf/StereoSplat_Plus/stereosplat"

accelerate_config_path="${REPO}/accelerate_configs/inference/gpu_1.yaml"
validator_script="validator/stereosplat/rendered_view_inside_bin_plus_diffix.py"
config_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus_two_stage/with_conf/stereosplat_plus_unified_model/with_difix3d/"
val_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${REPO}/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

STAGE2_MODEL_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_resume"
pretrained_model_path="${STAGE2_MODEL_DIR}/latest"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH}"

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
# Optional: --output_vis

}

StereoSplat_Plus_Without_Difix3D
#StereoSplat_Plus_With_Difix3D
