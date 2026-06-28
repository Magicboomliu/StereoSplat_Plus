#!/bin/bash
Train_Stereosplat_With_Conf_On_KITTI360(){
cd ../../..

REPO="/home/zliu/IROS2026/Conf/StereoSplat_Plus/stereosplat"

accelerate_config_path="${REPO}/accelerate_configs/accelerate_config.yaml"
configs_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
work_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/ablations/"
output_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/StereoSplat_With_Conf/basemodel/Input_View_Invariant/ablations/"
# Set to empty string "" to train from scratch; or provide an existing checkpoint path
resume_from=""
# Optional overrides (leave empty to use cfg defaults)
exp_name="input_invariant_stereosplat_with_conf_kitti360_112x544"
datapath="/data1/StereoDatasets/KITTI/KITTI360"

train_filelist="${REPO}/filenames/kitti360/trainval/all_2013_05_28_drive_0000_sync.txt"
val_filelist="${REPO}/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
test_filelist="${REPO}/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstLIDAR"
supp_view_nums=6
world_center="First_LiDAR_3_Uniform"
unimatch_weights_path="/data1/zliu/feedforward_outputs_new/depth_estimation_224x840/checkpoint-90000/model.safetensors"

use_wandb=true
# W&B settings (only used when use_wandb=true)
# 推荐：不要把 key 写进脚本；更安全的做法是先执行 `wandb login` 或在 shell 里 export WANDB_API_KEY
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
