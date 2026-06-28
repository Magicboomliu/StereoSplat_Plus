#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# pixel_fusion_pose_injection_single_model_multi_gpu_no_difix3d.sh
# 【Stage2 权重】多 GPU 并行 inference — 关闭 Difix3D pseudo 增强
# 与 pixel_fusion_pose_injection_single_model_multi_gpu.sh 相同，但传 --no_difix3d
# （progressive pass 只用 raw pseudo，不走 DifixRef）
# 入口: eval/run_multi_gpu.py
# 输出: metric.json（2v / mv / fuse）

run_fusion_validation_multi_gpu_no_difix3d() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/multi_gpu.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"

CKPT_STEP=20000
pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_debug/checkpoint-${CKPT_STEP}/model.safetensors"
self_pseudo_flag="--self_pseudo"

output_folder="/data1/zliu/IROS26/stereosplat_ablations/withconf/stage2/stereosplat_plus/whole-model-self-pseudo/fusion/val-multi-gpu/${CKPT_STEP}_no_difix3d"

if [ ! -f "$pretrained_model_path" ]; then
  echo "[ERROR] checkpoint not found: $pretrained_model_path"
  echo "        candidates/: $(ls /data1/zliu/IROS26/camera_ready_models/candidates/ 2>/dev/null | tr '\n' ' ')"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[Launch no_difix3d] ckpt=${CKPT_STEP}, output=${output_folder}"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config_file "$accelerate_config_path" eval/run_multi_gpu.py \
  --config_path "$configs_path" \
  --output_folder "$output_folder" \
  --val_filelist "$val_filelist" \
  --pretrained_model_path "$pretrained_model_path" \
  --no_difix3d \
  --conf_pixel_level_fusion \
  --fusion_mode soft \
  $self_pseudo_flag

}

run_fusion_validation_multi_gpu_no_difix3d
