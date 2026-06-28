#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# pixel_fusion_pose_injection_single_model_multi_gpu_tuned.sh
# 【Stage2 conf_tune 权重】多 GPU 并行 inference（split conf_head 架构）
# 必须使用 input_invariant_stereosplat_stage2_conf_tune.py，否则 unified-head
# inference 会 silent mismatch（missing gaussian_head / gs_decoder keys）。
# 入口: eval/run_multi_gpu_tune.py（split conf_head + load 校验）
# 原 unified-head Stage2 仍用 eval/run_multi_gpu.py
# 输出: metric.json（2v / mv / fuse，每路 16 项；mean_* = all_view）

run_fusion_validation_multi_gpu_conf_tune() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_3.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2_conf_tune.py"

# ---- Val filelist (align with conf_tune training val) ----
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val_tiny.txt"
# val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"

# ---- conf_tune checkpoint (split rgb_geom_head + conf_head) ----
# Option A: explicit step
pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_conf_tune/checkpoint-5000/model.safetensors"
# Option B: latest under work_dir (uncomment and comment Option A)
# work_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_conf_tune"
# latest_ckpt="$(ls -d "${work_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
# pretrained_model_path="${latest_ckpt}/model.safetensors"

pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199
self_pseudo_flag="--self_pseudo"

output_folder="/data1/zliu/IROS26/stereosplat_ablations/withconf/stage2/stereosplat_plus/conf-tune/fusion/val-multi-gpu/checkpoint-5500"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[Infer conf_tune] config=${configs_path}"
echo "[Infer conf_tune] weights=${pretrained_model_path}"
echo "[Infer conf_tune] output=${output_folder}"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config_file "$accelerate_config_path" eval/run_multi_gpu_tune.py \
  --config_path "$configs_path" \
  --output_folder "$output_folder" \
  --val_filelist "$val_filelist" \
  --pretrained_model_path "$pretrained_model_path" \
  --pretrained_diffix_model_path "$pretrained_diffix_model_path" \
  --timestep "$timestep" \
  --prompt "$prompt" \
  --use_ref \
  --conf_pixel_level_fusion \
  --fusion_mode soft \
  $self_pseudo_flag

}

run_fusion_validation_multi_gpu_conf_tune
