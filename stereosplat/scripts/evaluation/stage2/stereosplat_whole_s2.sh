#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Stage2 | stereosplat | whole | Stage2 checkpoint

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

RESULTS_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage2/stereosplat/whole_s2_ckpt"

eval_stage2_stereosplat_whole_s2() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch gpu_0.yaml stage2 stereosplat whole \
    "${RESULTS_BASE}" \
    --pretrained_model_path "${STAGE2_MODEL_DIR}/latest"
}

eval_stage2_stereosplat_whole_s2
