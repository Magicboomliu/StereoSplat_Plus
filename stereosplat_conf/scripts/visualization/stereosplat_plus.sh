#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

stereosplat_plus_vis() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_0.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/trainval/demo.txt"

pretrained_model_path="${STEREOSPLAT_CHECKPOINT:-/path/to/stereosplat_plus_conf_checkpoint}"
pretrained_diffix_model_path="${DIFIX3D_WEIGHTS:-/path/to/model_130001.pkl}"
prompt="remove degradation"
timestep=199
output_folder="${STEREOSPLAT_ROOT}/outputs/visualization/stereosplat_plus"

if [ ! -e "$pretrained_model_path" ]; then
  echo "[ERROR] checkpoint not found: $pretrained_model_path"
  exit 1
fi
if [ ! -e "$pretrained_diffix_model_path" ]; then
  echo "[ERROR] Difix3D checkpoint not found: $pretrained_diffix_model_path"
  exit 1
fi
if [ ! -f "$demo_filelist" ]; then
  echo "[ERROR] demo filelist not found: $demo_filelist"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[Launch] visualization (pixel_fusion + Difix3D + conf fusion), single GPU"
echo "[Launch] demo_filelist=${demo_filelist}"
echo "[Launch] weights=${pretrained_model_path}"
echo "[Launch] difix3d=${pretrained_diffix_model_path}"
echo "[Launch] output=${output_folder}"

pixi run -e cu118 accelerate launch --config_file "$accelerate_config_path" eval/run.py \
  --training_stage stage2 \
  --eval_mode pixel_fusion \
  --architecture whole \
  --config_path "$configs_path" \
  --output_folder "$output_folder" \
  --demo_filelist "$demo_filelist" \
  --pretrained_model_path "$pretrained_model_path" \
  --pretrained_diffix_model_path "$pretrained_diffix_model_path" \
  --timestep "$timestep" \
  --prompt "$prompt" \
  --use_diffix3d \
  --use_ref \
  --conf_pixel_level_fusion \
  --fusion_mode soft \
  --output_vis

}

stereosplat_plus_vis
