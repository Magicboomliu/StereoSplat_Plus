#!/usr/bin/env bash
# Deprecated: prefer scripts/evaluation/stage{1,2}/*.sh
# Kept so old paths keep working; launches eval/run.py directly.

if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: run with bash, not sh." >&2
  exit 1
fi

_conf_fusion_resolve_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  STEREOSPLAT_ROOT="$(cd "${script_dir}/../../../../../" && pwd)"
  cd "$STEREOSPLAT_ROOT" || exit 1
}

_conf_fusion_export_env() {
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"
}

_conf_fusion_default_paths() {
  configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
  val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
  demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo.txt"
  ablation_type="NMRFStereo"
  dataset_type="First_LiDAR_3_Uniform"
  pseudo_ratio="0.50 1.0"

  STAGE2_MODEL_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_resume"
  STAGE1_MODEL_PATH="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/latest/checkpoint-145000"
  pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"

  prompt="remove degradation"
  timestep=199
}

# Usage: _conf_fusion_launch <gpu_yaml> <training_stage> <eval_mode> <architecture> <output_folder> <extra args...>
_conf_fusion_launch() {
  local gpu_yaml="$1"
  local training_stage="$2"
  local eval_mode="$3"
  local architecture="$4"
  local output_folder="$5"
  shift 5

  local accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/${gpu_yaml}"

  TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
  pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" eval/run.py \
    --training_stage "$training_stage" \
    --eval_mode "$eval_mode" \
    --architecture "$architecture" \
    --config_path "$configs_path" \
    --output_folder "$output_folder" \
    --val_filelist "$val_filelist" \
    --demo_filelist "$demo_filelist" \
    --ablation_type "$ablation_type" \
    --dataset_type "$dataset_type" \
    --pseudo_ratio $pseudo_ratio \
    --timestep "$timestep" \
    --pretrained_diffix_model_path "$pretrained_diffix_model_path" \
    --prompt "$prompt" \
    "$@"
}

_conf_fusion_launch_vis() {
  _conf_fusion_launch "$@" --output_vis
}
