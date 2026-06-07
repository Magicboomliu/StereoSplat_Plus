#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# Separated dual-model eval. Toggle fusion: fusion_deactivate vs fusion_activate.
# Validator: posed_input_view_injected_selected_sep_model_pixel_level.py
# Flag: --conf_pixel_level_fusion (activate only)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

VALIDATOR="validator/stereosplat_plus_conf_fusion/two-stage/posed_input_view_injected_selected_sep_model_pixel_level.py"
OUTPUT_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus_two_stage/with_conf/pixel_level_fusion/seperated_model"

stereosplat_plus_sep_model_pixel_level_fusion_deactivate() {
  _conf_fusion_resolve_root
  _conf_fusion_default_paths
  _conf_fusion_export_env
  pretrained_model_path="${STAGE2_MODEL_DIR}/latest"
  _conf_fusion_launch gpu_0.yaml "$VALIDATOR" \
    "${OUTPUT_BASE}/fusion_deactivate/with_difix3d/0.5_1.0" \
    --stage_1_model_path "$STAGE1_MODEL_PATH" \
    --pretrained_model_path "$pretrained_model_path" \
    --use_diffix3d --use_ref
}

stereosplat_plus_sep_model_pixel_level_fusion_activate() {
  _conf_fusion_resolve_root
  _conf_fusion_default_paths
  _conf_fusion_export_env
  pretrained_model_path="${STAGE2_MODEL_DIR}/latest"
  _conf_fusion_launch gpu_1.yaml "$VALIDATOR" \
    "${OUTPUT_BASE}/fusion_activate/with_difix3d/0.5_1.0" \
    --stage_1_model_path "$STAGE1_MODEL_PATH" \
    --pretrained_model_path "$pretrained_model_path" \
    --use_diffix3d --use_ref \
    --conf_pixel_level_fusion
}

#stereosplat_plus_sep_model_pixel_level_fusion_deactivate
stereosplat_plus_sep_model_pixel_level_fusion_activate
