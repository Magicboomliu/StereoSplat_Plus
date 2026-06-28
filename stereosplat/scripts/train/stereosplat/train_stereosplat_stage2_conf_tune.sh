#!/bin/bash
# Stage2 conf-only fine-tuning (route A)
# - Split conf_head; freeze RGB/geometry; train fuse pixel + LPIPS only
# - Does NOT touch the main stage2 work_dir / checkpoints

Train_StereoSplat_Stage2_Conf_Tune(){
cd ../../..

REPO="/home/zliu/IROS2026/Conf/StereoSplat_Plus/stereosplat"

accelerate_config_path="${REPO}/accelerate_configs/accelerate_config.yaml"
configs_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2_conf_tune.py"
work_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_conf_tune_clean/"
output_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/train_visualization/stereosplat/Input_View_Invariant/withconf/stage2_conf_tune_clean/"

# FIRST LAUNCH: resume_from=""  (loads init_ckpt via weight surgery)
# RESUME:       resume_from="latest" or checkpoint dir under work_dir
resume_from=""
# resume_from="latest"

# Stage2 checkpoint for weight surgery (must exist; unified-head Stage2 ckpt)
init_ckpt="/data1/zliu/IROS26/camera_ready_models/candidates/checkpoint-5500"

exp_name="stereosplat_kitti360_stage2_conf_tune_from5500_clean"
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

mix_psuedo_views_ratio=1.0
mix_difix3d_ratio=0.9

use_wandb=true
wandb_api_key="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"
wandb_entity="liuzihua1004"
wandb_project="StereoSplat_Plus_Conf_Latest"
wandb_mode="online"
wandb_run_name="stereosplat_stage2_conf_tune_from5500_clean"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -e "$init_ckpt" ]; then
  echo "[ERROR] init_ckpt not found: $init_ckpt"
  echo "        Available under candidates/: $(ls /data1/zliu/IROS26/camera_ready_models/candidates/ 2>/dev/null | tr '\n' ' ')"
  exit 1
fi

if [ -n "$resume_from" ]; then
  if [ "$resume_from" = "latest" ]; then
    _resume_target="${work_dir}latest"
  else
    _resume_target="$resume_from"
  fi
  if [ ! -d "$_resume_target" ] || [ ! -f "$_resume_target/model.safetensors" ]; then
    echo "[ERROR] resume checkpoint not found: $_resume_target"
    echo "        Set resume_from=\"\" for first launch from init_ckpt."
    ls -d "${work_dir}"checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -5
    exit 1
  fi
  echo "[Resume] target=${_resume_target} (init_ckpt load will be skipped)"
fi

echo "[Launch conf_tune] init_ckpt=${init_ckpt} resume_from=${resume_from:-<none>}"

pixi run -e cu118 python -m accelerate.commands.launch --config_file $accelerate_config_path trainer/train_kitti360_stereosplat_stage2_conf_tune.py \
    --py-config $configs_path \
    --work-dir $work_dir \
    --init_ckpt $init_ckpt \
    $([ -n "$resume_from" ] && echo --resume-from "$resume_from") \
    --mix_psuedo_views_ratio $mix_psuedo_views_ratio \
    --mix_difix3d_ratio $mix_difix3d_ratio \
    --pretrained_difix3d $pretrained_difix3d \
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

Train_StereoSplat_Stage2_Conf_Tune
