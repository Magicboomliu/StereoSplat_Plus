#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Stage2 dataloader + Stage1 checkpoint (plain 2-view forward ablation)

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

RESULTS_BASE="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/with_conf/stage2/stereosplat/whole_s1_ckpt"

eval_stage2_stereosplat_whole_s1() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch gpu_0.yaml stage2 stereosplat whole \
    "${RESULTS_BASE}" \
    --pretrained_model_path "$STAGE1_MODEL_PATH"
}

eval_stage2_stereosplat_whole_s1_vis() {
  _eval_resolve_root
  _eval_default_paths
  _eval_config_stage2
  _eval_export_env
  _eval_launch_vis gpu_0.yaml stage2 stereosplat whole \
    "${RESULTS_BASE}/vis" \
    --pretrained_model_path "$STAGE1_MODEL_PATH"
}

eval_stage2_stereosplat_whole_s1
#eval_stage2_stereosplat_whole_s1_vis
