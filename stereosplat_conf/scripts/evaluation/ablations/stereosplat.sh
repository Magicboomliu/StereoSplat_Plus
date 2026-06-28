#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# stereosplat_conf.sh — with-conf weights, 2-view eval (--eval_mode stereosplat), multi-GPU

StereoSplat_2Views_Eval_MultiGPU() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/multi_gpu.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"

pretrained_model_path="/data1/zliu/IROS26/camera_ready_models/Ablations/withconf/stereosplat"
output_folder="${STEREOSPLAT_ROOT}/outputs/eval/ablations/withconf/stereosplat_conf"

if [ ! -e "$pretrained_model_path" ]; then
  echo "[ERROR] checkpoint not found: $pretrained_model_path"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[Launch] eval_mode=stereosplat (2-view), multi_gpu"
echo "[Launch] weights=${pretrained_model_path}"
echo "[Launch] output=${output_folder}"

pixi run -e cu118 accelerate launch --config_file "$accelerate_config_path" eval/run_multi_gpu.py \
  --eval_mode stereosplat \
  --architecture whole \
  --config_path "$configs_path" \
  --output_folder "$output_folder" \
  --val_filelist "$val_filelist" \
  --pretrained_model_path "$pretrained_model_path"
  # --output_vis  # not supported with multi-GPU in run_multi_gpu.py

}

StereoSplat_2Views_Eval_MultiGPU
