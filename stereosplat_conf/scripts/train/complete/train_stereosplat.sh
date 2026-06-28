#!/bin/bash
Train_Stereosplat_With_Conf_On_KITTI360(){
cd ../../..

REPO="/home/zliu/IROS2026/Conf/StereoSplat_Plus/stereosplat_conf"

accelerate_config_path="${REPO}/accelerate_configs/accelerate_config.yaml"
configs_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"

work_dir="../../../../outputs/models/all/withconf/stereosplat_conf"
output_dir="../../../../outputs/logs/all/withconf/stereosplat_conf"


resume_from=""
# Optional overrides (leave empty to use cfg defaults)
exp_name="input_invariant_stereosplat_with_conf_kitti360_112x544"
datapath="/data1/StereoDatasets/KITTI/KITTI360"

train_filelist="${REPO}/filenames/kitti360/train_complete/train.txt"
val_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
test_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstLIDAR"
supp_view_nums=6
world_center="First_LiDAR_3_Uniform"
unimatch_weights_path="/data1/zliu/feedforward_outputs_new/depth_estimation_224x840/checkpoint-90000/model.safetensors"

use_wandb=false


wandb_api_key="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"
wandb_entity="liuzihua1004"
wandb_project="StereoSplat_Plus_Conf_Ablations"
# online | offline | disabled
wandb_mode="online"
wandb_run_name="input_invariant_stereosplat_with_conf_kitti360"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd):${PYTHONPATH}"
# Force single GPU regardless of ~/.cache/huggingface/accelerate/default_config.yaml
# export CUDA_VISIBLE_DEVICES=0

pixi run -e cu118 python -m accelerate.commands.launch --config_file $accelerate_config_path trainer/train_kitti360_stereosplat_with_conf.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
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
