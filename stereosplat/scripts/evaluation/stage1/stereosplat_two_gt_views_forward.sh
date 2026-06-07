#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# stereosplat_two_gt_views_forward.sh
# 【Stage1 权重】stereosplat 模式：只用 2 张 GT 前向视角直接渲染，无 pseudo view / 无 Difix3D / 无 conf 融合
# 对应: --training_stage stage1 --eval_mode stereosplat --architecture whole

run_metric_eval() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_0.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage1/stereosplat"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

STAGE1_MODEL_PATH="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/latest/checkpoint-145000"
pretrained_model_path="${STAGE1_MODEL_PATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" eval/run.py \
  --training_stage stage1 \
  --eval_mode stereosplat \
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

run_metric_eval
