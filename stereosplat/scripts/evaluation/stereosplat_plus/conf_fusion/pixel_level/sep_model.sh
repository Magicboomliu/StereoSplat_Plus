#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Deprecated → scripts/evaluation/stage2/pixel_fusion_separated.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

OUTPUT_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus_two_stage/with_conf/pixel_level_fusion/seperated_model"

stereosplat_plus_sep_model_pixel_level_fusion_deactivate() {
  _conf_fusion_resolve_root
  _conf_fusion_default_paths
  _conf_fusion_export_env
  _conf_fusion_launch gpu_0.yaml stage2 pixel_fusion separated \
    "${OUTPUT_BASE}/fusion_deactivate/with_difix3d/0.5_1.0" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest" \
    --stage_1_model_path "$STAGE1_MODEL_PATH" \
    --use_diffix3d --use_ref
}

stereosplat_plus_sep_model_pixel_level_fusion_activate() {
  _conf_fusion_resolve_root
  _conf_fusion_default_paths
  _conf_fusion_export_env
  _conf_fusion_launch gpu_1.yaml stage2 pixel_fusion separated \
    "${OUTPUT_BASE}/fusion_activate/with_difix3d/0.5_1.0" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest" \
    --stage_1_model_path "$STAGE1_MODEL_PATH" \
    --use_diffix3d --use_ref \
    --conf_pixel_level_fusion
}

#stereosplat_plus_sep_model_pixel_level_fusion_deactivate
stereosplat_plus_sep_model_pixel_level_fusion_activate
