#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Stage2 | pixel_fusion | whole | toggle fusion on/off

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

RESULTS_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage2/pixel_fusion/whole"

eval_stage2_pixel_fusion_whole_deactivate() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch gpu_0.yaml stage2 pixel_fusion whole \
    "${RESULTS_BASE}/fusion_deactivate/with_difix3d/0.5_1.0" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest" \
    --use_diffix3d --use_ref
}

eval_stage2_pixel_fusion_whole_activate() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch gpu_1.yaml stage2 pixel_fusion whole \
    "${RESULTS_BASE}/fusion_activate/with_difix3d/0.5_1.0" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest" \
    --use_diffix3d --use_ref \
    --conf_pixel_level_fusion
}

#eval_stage2_pixel_fusion_whole_deactivate
eval_stage2_pixel_fusion_whole_activate
