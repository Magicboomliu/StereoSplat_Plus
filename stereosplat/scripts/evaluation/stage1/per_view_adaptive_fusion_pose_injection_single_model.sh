#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# per_view_adaptive_fusion_pose_injection_single_model.sh
# 【Stage1 权重】Per-view adaptive conf fusion（L1/L2/L3，见 docs/plan_2weeks.md）
# 对应: --fusion_mode per_view_adaptive --conf_pixel_level_fusion
# 函数: run_s1_first_protect / run_s1_first_protect_zscore / run_s1_zscore_margin /
#       run_s1_zscore_soft / run_s1_first_small_margin（底部注释切换）

_run_per_view_adaptive() {
  fusion_first_margin="$1"
  fusion_center_margin="$2"
  fusion_last_margin="$3"
  fusion_calibration="$4"
  fusion_temperature="$5"
  exp_tag="$6"

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
  cd "$STEREOSPLAT_ROOT" || exit 1

  accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_1.yaml"
  configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
  output_folder="/data1/zliu/IROS26/stereosplat_ablations/withconf/stage1/stereosplat_plus/fusion/per-view-adaptive/${exp_tag}/0.5_1.0"
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

  launch_args=(
    --training_stage stage1
    --eval_mode pixel_fusion
    --architecture whole
    --config_path "$configs_path"
    --output_folder "$output_folder"
    --val_filelist "$val_filelist"
    --demo_filelist "$demo_filelist"
    --ablation_type "$ablation_type"
    --dataset_type "$dataset_type"
    --pseudo_ratio $pseudo_ratio
    --pretrained_model_path "$pretrained_model_path"
    --pretrained_diffix_model_path "$pretrained_diffix_model_path"
    --timestep "$timestep"
    --prompt "$prompt"
    --use_diffix3d
    --use_ref
    --conf_pixel_level_fusion
    --fusion_mode per_view_adaptive
    --fusion_first_margin "$fusion_first_margin"
    --fusion_center_margin "$fusion_center_margin"
    --fusion_last_margin "$fusion_last_margin"
    --fusion_calibration "$fusion_calibration"
  )

  if [ -n "$fusion_temperature" ]; then
    launch_args+=(--fusion_temperature "$fusion_temperature")
  fi

  TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
  pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" eval/run.py \
    "${launch_args[@]}"
    # --output_vis
}

# S1-1: first 强制 base，center/last legacy margin=0
run_s1_first_protect() {
  _run_per_view_adaptive 999.0 0.0 0.0 none "" "first999_none"
}

# S1-2: first 强制 base + per-image zscore（推荐首选）
run_s1_first_protect_zscore() {
  _run_per_view_adaptive 999.0 0.0 0.0 zscore "" "first999_zscore"
}

# S1-4: zscore + center margin 0.05
run_s1_zscore_margin() {
  _run_per_view_adaptive 999.0 0.05 0.0 zscore "" "first999_zscore_cmargin005"
}

# S1-5: zscore + center margin 0.05 + last margin -0.05（略偏 plus）
run_s1_zscore_margin_last_bias() {
  _run_per_view_adaptive 999.0 0.05 -0.05 zscore "" "first999_zscore_c005_l-005"
}

# S1-6: zscore + soft blending T=10
run_s1_zscore_soft() {
  _run_per_view_adaptive 999.0 0.0 0.0 zscore 10.0 "first999_zscore_soft10"
}

# S1-8: first 小 margin（允许少量 first fusion）+ zscore
run_s1_first_small_margin() {
  _run_per_view_adaptive 0.1 0.0 0.0 zscore "" "first01_zscore"
}

# run_s1_first_protect
run_s1_first_protect_zscore
# run_s1_zscore_margin
# run_s1_zscore_margin_last_bias
# run_s1_zscore_soft
# run_s1_first_small_margin
