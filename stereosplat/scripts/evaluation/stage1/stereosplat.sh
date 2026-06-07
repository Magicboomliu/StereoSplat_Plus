#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Stage1 | stereosplat | whole

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

RESULTS_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage1/stereosplat"

eval_stage1_stereosplat() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage1
  _eval_export_env
  _eval_launch gpu_0.yaml stage1 stereosplat whole \
    "${RESULTS_BASE}" \
    --pretrained_model_path "$STAGE1_MODEL_PATH"
}

eval_stage1_stereosplat
