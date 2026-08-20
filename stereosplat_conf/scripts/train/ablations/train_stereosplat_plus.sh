#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

Train_StereoSplat_Stage2_Self_Pseudo() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO" || exit 1

accelerate_config_path="${REPO}/accelerate_configs/accelerate_config.yaml"
configs_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"

work_dir="${REPO}/outputs/models/ablations/withconf/stereosplat_plus_conf"
output_dir="${REPO}/outputs/logs/ablations/withconf/stereosplat_plus_conf"

resume_from=""
exp_name="stereosplat_kitti360_stage2_self_pseudo_with_conf_and_difix3d"

datapath="${KITTI360_DATAPATH:-/path/to/KITTI360}"
stage_1_model_path="${STAGE1_CHECKPOINT:-/path/to/stage1_stereosplat_conf_ablation}"
unimatch_weights_path="${UNIMATCH_WEIGHTS:-/path/to/checkpoint-90000/model.safetensors}"
pretrained_difix3d="${DIFIX3D_WEIGHTS:-/path/to/model_130001.pkl}"

train_filelist="${REPO}/filenames/kitti360/trainval/train_2013_05_28_drive_0000_sync.txt"
val_filelist="${REPO}/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
test_filelist="${REPO}/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"

sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstLIDAR"
supp_view_nums=6
world_center="First_Stage2"

mix_psuedo_views_ratio=0.9
mix_difix3d_ratio=0.9

use_wandb=false
wandb_api_key="${WANDB_API_KEY:-}"
wandb_entity="${WANDB_ENTITY:-your_entity}"
wandb_project="${WANDB_PROJECT:-StereoSplat_Plus_Conf_Ablations}"
wandb_mode="${WANDB_MODE:-online}"
wandb_run_name="${WANDB_RUN_NAME:-stereosplat_plus_ablation}"

if [ ! -d "$datapath" ]; then
  echo "[ERROR] KITTI-360 datapath not found: $datapath"
  exit 1
fi
if [ ! -e "$stage_1_model_path" ]; then
  echo "[ERROR] Stage 1 checkpoint not found: $stage_1_model_path"
  exit 1
fi
if [ ! -f "$unimatch_weights_path" ]; then
  echo "[ERROR] UniMatch weights not found: $unimatch_weights_path"
  exit 1
fi
if [ ! -f "$pretrained_difix3d" ]; then
  echo "[ERROR] Difix3D weights not found: $pretrained_difix3d"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[Launch] resume_from=${resume_from:-<none>}"

pixi run -e cu118 python -m accelerate.commands.launch --config_file "$accelerate_config_path" \
  trainer/train_kitti360_stereosplat_plus_with_difix3d.py \
  --py-config "$configs_path" \
  --work-dir "$work_dir" \
  $([ -n "$resume_from" ] && echo --resume-from "$resume_from") \
  --stage_1_model_path "$stage_1_model_path" \
  --mix_psuedo_views_ratio "$mix_psuedo_views_ratio" \
  --mix_difix3d_ratio "$mix_difix3d_ratio" \
  --pretrained_difix3d "$pretrained_difix3d" \
  --self_pseudo \
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

Train_StereoSplat_Stage2_Self_Pseudo
