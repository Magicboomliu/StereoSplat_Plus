#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Stage2 | stereosplat_plus | separated | frozen Stage1 + Stage2

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

RESULTS_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage2/stereosplat_plus/separated"

eval_stage2_stereosplat_plus_separated_no_difix() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch gpu_2.yaml stage2 stereosplat_plus separated \
    "${RESULTS_BASE}/no_difix3d/0.5_1.0" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest" \
    --stage_1_model_path "$STAGE1_MODEL_PATH"
}

eval_stage2_stereosplat_plus_separated_with_difix() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch gpu_3.yaml stage2 stereosplat_plus separated \
    "${RESULTS_BASE}/with_difix3d/0.5_1.0" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest" \
    --stage_1_model_path "$STAGE1_MODEL_PATH" \
    --use_diffix3d --use_ref
}

eval_stage2_stereosplat_plus_separated_no_difix
#eval_stage2_stereosplat_plus_separated_with_difix
