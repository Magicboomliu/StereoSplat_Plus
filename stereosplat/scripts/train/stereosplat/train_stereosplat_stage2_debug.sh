#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# train_stereosplat_stage2_debug.sh
# Self-Pseudo debug — selective key-view joint margin vs 2v (KPI-F).
#
# Config: input_invariant_stereosplat_stage2_no_detach.py
#   margin_detach_ref=True  (mean + fusion_mv key margins: ref detached)
#   margin_detach_ref_key_2v_views=False  (#4-7: mv/fused vs 2v center/last joint grad)
#   weight_margin_key_views=0.005, weight_fusion_mv_margin=1.5, fusion_mv_psnr_margin=0.3
#   save_freq=10, val_freq=10, max_train_steps=10021 (resume@10000 → +21 iters)
#
# Data (debug subset, same as stage2_self_pseudo_debug):
#   train/val/test → val.txt / val_tiny.txt / val_tiny.txt
#
# Save: NEW work_dir / output_dir (below) — does NOT overwrite stage2_self_pseudo_debug
# Resume default: continue from checkpoint-5550 (detach control run in no_detach_debug work_dir)
# Continue this run later: resume_from="latest" (within no_detach_debug work_dir)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.." || exit 1

REPO="$(pwd)"

accelerate_config_path="${REPO}/accelerate_configs/accelerate_config.yaml"
configs_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2_no_detach.py"

# ---- NEW folders (checkpoints + val json saved here) ----
work_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_no_detach_debug/"
output_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/train_visualization/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_no_detach_debug/"

# ---- Resume: default best1 (detach run) checkpoint-5500 ----
PREV_DEBUG_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_debug"
resume_from="/data1/zliu/IROS26/camera_ready_models/candidates/checkpoint-10000"
# Continue no_detach run: resume_from="latest"
# Train from Stage1 only: resume_from=""

stage_1_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/latest/checkpoint-145000/"

exp_name="stereosplat_kitti360_stage2_self_pseudo_no_detach_debug"
datapath="/data1/StereoDatasets/KITTI/KITTI360"
train_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
val_filelist="${REPO}/filenames/kitti360/train_complete/val_tiny.txt"
test_filelist="${REPO}/filenames/kitti360/train_complete/val_tiny.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstLIDAR"
supp_view_nums=6
world_center="First_Stage2"
unimatch_weights_path="/data1/zliu/feedforward_outputs_new/depth_estimation_224x840/checkpoint-90000/model.safetensors"

pretrained_difix3d="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
mix_psuedo_views_ratio=0.9
mix_difix3d_ratio=0.9

use_wandb=true
wandb_api_key="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"
wandb_entity="liuzihua1004"
wandb_project="StereoSplat_Plus_Conf_Latest"
wandb_mode="online"
wandb_run_name="stereosplat_self_pseudo_no_detach_from5500"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[Launch] config=${configs_path}"
echo "[Launch] work_dir=${work_dir}"
echo "[Launch] resume_from=${resume_from:-<none>}"
echo "[Launch] save_freq=10, val_freq=10, max_train_steps=10021 (short debug)"
echo "[Launch] margin_detach_ref_key_2v_views=False, weight_margin_key_views=0.005, weight_fusion_mv_margin=1.5, fusion_mv_psnr_margin=0.3"

pixi run -e cu118 python -m accelerate.commands.launch --config_file "$accelerate_config_path" \
    trainer/train_kitti360_stereosplat_stage2_with_difix3d.py \
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
    ${supp_view_nums:+--supp-view-nums "$supp_view_nums"} \
    ${world_center:+--world-center "$world_center"} \
    ${unimatch_weights_path:+--unimatch-weights-path "$unimatch_weights_path"} \
    $([ "$use_wandb" = true ] && echo --use-wandb) \
    $([ "$use_wandb" = true ] && [ -n "$wandb_entity" ] && echo --wandb-entity "$wandb_entity") \
    $([ "$use_wandb" = true ] && [ -n "$wandb_project" ] && echo --wandb-project "$wandb_project") \
    $([ "$use_wandb" = true ] && [ -n "$wandb_mode" ] && echo --wandb-mode "$wandb_mode") \
    $([ "$use_wandb" = true ] && [ -n "$wandb_run_name" ] && echo --wandb-run-name "$wandb_run_name") \
    $([ "$use_wandb" = true ] && [ -n "$wandb_api_key" ] && echo --wandb-api-key "$wandb_api_key")
