#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# pixel_fusion_pose_injection_single_model_multi_gpu.sh
# 【Stage2 权重】4-GPU 并行 validation — 逻辑与训练时 fusion_metric.json 完全一致
# 入口: eval/run_multi_gpu.py（NOT eval/run.py）
# 输出: fusion_metric.json（2view / pseudo_multiview / pseudo_fused）

run_fusion_validation_multi_gpu() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/multi_gpu.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
# ---- Val filelist (must match training val for comparable numbers) ----
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
# val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"

# ---- Model checkpoint (best1: checkpoint-5500) ----
pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_debug/checkpoint-4000/model.safetensors"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199
# Self-pseudo training → progressive pass uses current model (no frozen Stage1)
self_pseudo_flag="--self_pseudo"

# output_folder="/data1/zliu/IROS26/stereosplat_ablations/withconf/stage2/stereosplat_plus/whole-model-self-pseudo/fusion/val-multi-gpu/checkpoint-5500"


output_folder="/data1/zliu/IROS26/stereosplat_ablations/withconf/stage2/stereosplat_plus/whole-model-self-pseudo/fusion/val-multi-gpu/checkpoint-4000"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config_file "$accelerate_config_path" eval/run_multi_gpu.py \
  --config_path "$configs_path" \
  --output_folder "$output_folder" \
  --val_filelist "$val_filelist" \
  --pretrained_model_path "$pretrained_model_path" \
  --pretrained_diffix_model_path "$pretrained_diffix_model_path" \
  --timestep "$timestep" \
  --prompt "$prompt" \
  --use_ref \
  $self_pseudo_flag
  # Two-model (non-self-pseudo): remove --self_pseudo and add:
  # --stage_1_model_path "/path/to/stage1/checkpoint-145000/model.safetensors"

}

run_fusion_validation_multi_gpu
