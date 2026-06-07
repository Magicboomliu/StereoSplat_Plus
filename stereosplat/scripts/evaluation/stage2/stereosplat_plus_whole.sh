#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# Stage2 | stereosplat_plus | whole | progressive unified model

eval_stage2_stereosplat_plus_whole_no_difix() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_0.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage2/stereosplat_plus/whole/no_difix3d"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

STAGE2_MODEL_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_resume"
pretrained_model_path="${STAGE2_MODEL_DIR}/latest"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" eval/run.py \
  --training_stage stage2 \
  --eval_mode stereosplat_plus \
  --architecture whole \
  --config_path "$configs_path" \
  --output_folder "$output_folder" \
  --val_filelist "$val_filelist" \
  --demo_filelist "$demo_filelist" \
  --ablation_type "$ablation_type" \
  --dataset_type "$dataset_type" \
  --pretrained_model_path "$pretrained_model_path"
  # --output_vis

}



eval_stage2_stereosplat_plus_whole_with_difix() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_1.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage2/stereosplat_plus/whole/with_difix3d"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

STAGE2_MODEL_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_resume"
pretrained_model_path="${STAGE2_MODEL_DIR}/latest"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" eval/run.py \
  --training_stage stage2 \
  --eval_mode stereosplat_plus \
  --architecture whole \
  --config_path "$configs_path" \
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

eval_stage2_stereosplat_plus_whole_no_difix
#eval_stage2_stereosplat_plus_whole_with_difix
