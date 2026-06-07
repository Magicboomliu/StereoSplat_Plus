#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# gs_conf_voxel_fusion_pose_injection_single_model.sh
# 【Stage1 权重】3D GS 体素 conf mean + margin + base 优先门控融合，融合后单次渲染
# 对应: --training_stage stage1 --eval_mode pixel_fusion --architecture whole --gs_conf_fusion
# 函数: run_gs_conf_voxel_fusion（底部默认启用）
# base 优先: 仅当体素内 mean(conf_base) < gs_fusion_base_conf_thresh 时 plus 才可赢

run_gs_conf_voxel_fusion() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1

accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_0.yaml"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
gs_fusion_voxel_size="0.1"
gs_fusion_margin="0.05"
gs_fusion_conf_agg="mean"
gs_fusion_base_conf_thresh="0.60"
output_folder="/data1/zliu/IROS26/stereosplat_ablations/withconf/stage1/stereosplat_plus/fusion/gs-voxel-conf/${gs_fusion_conf_agg}/base_thresh_${gs_fusion_base_conf_thresh}/margin_${gs_fusion_margin}/voxel_${gs_fusion_voxel_size}/0.5_1.0"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pseudo_ratio="0.50 1.0"

STAGE1_MODEL_PATH="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/latest/checkpoint-145000"
pretrained_model_path="${STAGE1_MODEL_PATH}"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STEREOSPLAT_ROOT}:${PYTHONPATH}"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" eval/run.py \
  --training_stage stage1 \
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
  --gs_conf_fusion \
  --gs_fusion_voxel_size "$gs_fusion_voxel_size" \
  --gs_fusion_margin "$gs_fusion_margin" \
  --gs_fusion_conf_agg "$gs_fusion_conf_agg" \
  --gs_fusion_base_conf_thresh "$gs_fusion_base_conf_thresh"
  # --output_vis

}

run_gs_conf_voxel_fusion
