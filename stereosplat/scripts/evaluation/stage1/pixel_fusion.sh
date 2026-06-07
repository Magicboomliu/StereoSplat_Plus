#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Stage1 | pixel_fusion | whole | Stage1 checkpoint ablation

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

RESULTS_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage1/pixel_fusion/whole"

eval_stage1_pixel_fusion_deactivate() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage1
  _eval_export_env
  _eval_launch gpu_0.yaml stage1 pixel_fusion whole \
    "${RESULTS_BASE}/fusion_deactivate/with_difix3d/0.5_1.0" \
    --pretrained_model_path "$STAGE1_MODEL_PATH" \
    --use_diffix3d --use_ref
}

eval_stage1_pixel_fusion_activate() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage1
  _eval_export_env
  _eval_launch gpu_1.yaml stage1 pixel_fusion whole \
    "${RESULTS_BASE}/fusion_activate/with_difix3d/0.5_1.0" \
    --pretrained_model_path "$STAGE1_MODEL_PATH" \
    --use_diffix3d --use_ref \
    --conf_pixel_level_fusion
}

eval_stage1_pixel_fusion_deactivate
#eval_stage1_pixel_fusion_activate
