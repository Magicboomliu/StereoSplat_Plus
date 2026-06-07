#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# oracle_gt_upper_bound_pose_injection.sh
# 【Stage1 权重】Oracle 上界：
#   2-view→G_base；渲染 pseudo→reinject→G_plus；双路渲染后用 GT 逐像素选误差更小者融合
# 使用: --use_gt_view（不走 conf 融合）

run_oracle_eval() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_3.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
output_folder="/data1/zliu/IROS26/stereosplat_ablations/withconf/stage1/stereosplat_plus/upper_bound/0.50_1.0"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pseudo_ratio="0.50 1.0"

STAGE1_MODEL_PATH="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/latest/checkpoint-145000"
pretrained_model_path="${STAGE1_MODEL_PATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" eval/run.py \
  --training_stage stage1 \
  --eval_mode stereosplat_plus \
  --architecture whole \
  --config_path "$configs_path" \
  --output_folder "$output_folder" \
  --val_filelist "$val_filelist" \
  --demo_filelist "$demo_filelist" \
  --ablation_type "$ablation_type" \
  --dataset_type "$dataset_type" \
  --pseudo_ratio $pseudo_ratio \
  --pretrained_model_path "$pretrained_model_path" \
  --use_gt_view
  # --output_vis

}

run_oracle_eval
