#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# pixel_fusion_pose_injection_single_model_multi_gpu.sh
# 【Stage2 权重 · 4 GPU】pixel_fusion 单模型 pose injection + soft conf 融合
# 对应: eval/run_multi_gpu.py + accelerate_configs/inference/multi_gpu.yaml
# 与 pixel_fusion_pose_injection_single_model.sh 参数一致，但 4 卡并行且 metric.json 全局聚合正确

run_with_conf_pixel_level_fusion_multi_gpu() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/multi_gpu.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val_tiny.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pseudo_ratio="0.50 1.0"

# ---- Model path: point to self-pseudo checkpoint ----
# pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo/checkpoint-18000/"
pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/latest/checkpoint-145000/"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199

# ---- Pixel-level conf fusion (train/val-aligned soft fusion) ----
fusion_mode="soft"

output_folder="/data1/zliu/IROS26/stereosplat_ablations/withconf/stage2/stereosplat_plus/whole-model-self-pseudo/fusion/pixel-level-fusion/${fusion_mode}_multi_gpu_reference"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Do NOT set CUDA_VISIBLE_DEVICES; multi_gpu.yaml uses GPUs 0,1,2,3
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config_file "$accelerate_config_path" eval/run_multi_gpu.py \
  --training_stage stage2 \
  --eval_mode pixel_fusion \
  --architecture whole \
  --config_path "$configs_path" \
  --output_folder "$output_folder" \
  --val_filelist "$val_filelist" \
  --demo_filelist "$demo_filelist" \
  --ablation_type "$ablation_type" \
  --dataset_type "$dataset_type" \
  --pseudo_ratio $pseudo_ratio \
  --pretrained_model_path "$pretrained_model_path" \
  --pretrained_diffix_model_path "$pretrained_diffix_model_path" \
  --timestep "$timestep" \
  --prompt "$prompt" \
  --use_diffix3d \
  --use_ref \
  --conf_pixel_level_fusion \
  --fusion_mode "$fusion_mode"
  # legacy hard fusion: add --fusion_mode legacy --conf_fusion_margin 0.05
  # --output_vis  # NOT supported on multi-GPU (use single-GPU run.py instead)

}

run_with_conf_pixel_level_fusion_multi_gpu
