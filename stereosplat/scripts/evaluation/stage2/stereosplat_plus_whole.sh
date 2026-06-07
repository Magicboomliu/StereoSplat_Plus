#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Stage2 | stereosplat_plus | whole | progressive unified model

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

RESULTS_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage2/stereosplat_plus/whole"

eval_stage2_stereosplat_plus_whole_no_difix() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch gpu_0.yaml stage2 stereosplat_plus whole \
    "${RESULTS_BASE}/no_difix3d" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest"
}

eval_stage2_stereosplat_plus_whole_with_difix() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch gpu_1.yaml stage2 stereosplat_plus whole \
    "${RESULTS_BASE}/with_difix3d" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest" \
    --use_diffix3d --use_ref
}

eval_stage2_stereosplat_plus_whole_vis() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch_vis gpu_0.yaml stage2 stereosplat_plus whole \
    "${RESULTS_BASE}/vis" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest" \
    --use_diffix3d --use_ref
}

eval_stage2_stereosplat_plus_whole_no_difix
#eval_stage2_stereosplat_plus_whole_with_difix
#eval_stage2_stereosplat_plus_whole_vis
