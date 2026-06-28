#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# render_inside_bin.sh
# 【Stage2 权重】stereosplat 模式 — 纯 2-view forward，bin 内 first/center/last 渲染
# 多 GPU 并行 eval（val 分片 → gather → metric.json）
# 对应: --eval_mode stereosplat --architecture whole
# 入口: eval/run_multi_gpu.py（NOT validator wrapper / 单卡 eval/run.py）

StereoSplat_2Views_Eval_MultiGPU() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/multi_gpu.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
# val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"

val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
# ---- Model checkpoint (dir or model.safetensors) ----
# complete training
# STAGE2_MODEL_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_resume"
# pretrained_model_path="${STAGE2_MODEL_DIR}/latest"

# ablations training
checkpoint_name="checkpoint-80000"
STAGE2_MODEL_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/ablations"
pretrained_model_path="${STAGE2_MODEL_DIR}/${checkpoint_name}"

output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat/withconf/stage2_mix_training/ablations/2_views/val-multi-gpu/${checkpoint_name}"

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

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
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
