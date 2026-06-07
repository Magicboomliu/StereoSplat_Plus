#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Stage1 | stereosplat_plus | whole | progressive unified

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

RESULTS_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage1/stereosplat_plus/whole"

eval_stage1_stereosplat_plus_no_difix() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage1
  _eval_export_env
  _eval_launch gpu_0.yaml stage1 stereosplat_plus whole \
    "${RESULTS_BASE}/no_difix3d" \
    --pretrained_model_path "$STAGE1_MODEL_PATH"
}

eval_stage1_stereosplat_plus_with_difix() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage1
  _eval_export_env
  _eval_launch gpu_1.yaml stage1 stereosplat_plus whole \
    "${RESULTS_BASE}/with_difix3d" \
    --pretrained_model_path "$STAGE1_MODEL_PATH" \
    --use_diffix3d --use_ref
}

eval_stage1_stereosplat_plus_vis() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage1
  _eval_export_env
  _eval_launch_vis gpu_0.yaml stage1 stereosplat_plus whole \
    "${RESULTS_BASE}/vis" \
    --pretrained_model_path "$STAGE1_MODEL_PATH" \
    --use_diffix3d --use_ref
}

eval_stage1_stereosplat_plus_no_difix
#eval_stage1_stereosplat_plus_with_difix
#eval_stage1_stereosplat_plus_vis
