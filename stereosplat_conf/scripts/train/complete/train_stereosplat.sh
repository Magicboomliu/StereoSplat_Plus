#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

Train_Stereosplat_With_Conf_On_KITTI360() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO" || exit 1

accelerate_config_path="${REPO}/accelerate_configs/accelerate_config.yaml"
configs_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"

work_dir="${REPO}/outputs/models/all/withconf/stereosplat_conf"
output_dir="${REPO}/outputs/logs/all/withconf/stereosplat_conf"

resume_from=""
exp_name="input_invariant_stereosplat_with_conf_kitti360_112x544"

# Edit these paths, or export env vars before running (see README).
datapath="${KITTI360_DATAPATH:-/path/to/KITTI360}"
unimatch_weights_path="${UNIMATCH_WEIGHTS:-/path/to/checkpoint-90000/model.safetensors}"

train_filelist="${REPO}/filenames/kitti360/train_complete/train.txt"
val_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
test_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstLIDAR"
supp_view_nums=6
world_center="First_LiDAR_3_Uniform"

use_wandb=false
wandb_api_key="${WANDB_API_KEY:-}"
wandb_entity="${WANDB_ENTITY:-your_entity}"
wandb_project="${WANDB_PROJECT:-StereoSplat_Plus_Conf}"
wandb_mode="${WANDB_MODE:-online}"
wandb_run_name="${WANDB_RUN_NAME:-input_invariant_stereosplat_with_conf_kitti360}"

if [ ! -d "$datapath" ]; then
  echo "[ERROR] KITTI-360 datapath not found: $datapath"
  echo "        Set KITTI360_DATAPATH or edit datapath in this script."
  exit 1
fi
if [ ! -f "$unimatch_weights_path" ]; then
  echo "[ERROR] UniMatch weights not found: $unimatch_weights_path"
  echo "        Set UNIMATCH_WEIGHTS or edit unimatch_weights_path in this script."
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH}"

pixi run -e cu118 python -m accelerate.commands.launch --config_file "$accelerate_config_path" \
  trainer/train_kitti360_stereosplat_with_conf.py \
  --py-config "$configs_path" \
  --work-dir "$work_dir" \
  --resume-from "${resume_from:-None}" \
  ${exp_name:+--exp-name "$exp_name"} \
  ${output_dir:+--output-dir "$output_dir"} \
  ${datapath:+--datapath "$datapath"} \
  ${train_filelist:+--train-filelist "$train_filelist"} \
  ${val_filelist:+--val-filelist "$val_filelist"} \
  ${test_filelist:+--test-filelist "$test_filelist"} \
  ${sequence:+--sequence "$sequence"} \
  ${data_version:+--data-version "$data_version"} \
  ${supp_view_nums:+--supp-view-nums $supp_view_nums} \
  ${world_center:+--world-center "$world_center"} \
  ${unimatch_weights_path:+--unimatch-weights-path "$unimatch_weights_path"} \
  $([ "$use_wandb" = true ] && echo --use-wandb) \
  $([ "$use_wandb" = true ] && [ -n "$wandb_entity" ] && echo --wandb-entity "$wandb_entity") \
  $([ "$use_wandb" = true ] && [ -n "$wandb_project" ] && echo --wandb-project "$wandb_project") \
  $([ "$use_wandb" = true ] && [ -n "$wandb_mode" ] && echo --wandb-mode "$wandb_mode") \
  $([ "$use_wandb" = true ] && [ -n "$wandb_run_name" ] && echo --wandb-run-name "$wandb_run_name") \
  $([ "$use_wandb" = true ] && [ -n "$wandb_api_key" ] && echo --wandb-api-key "$wandb_api_key")

}

Train_Stereosplat_With_Conf_On_KITTI360
