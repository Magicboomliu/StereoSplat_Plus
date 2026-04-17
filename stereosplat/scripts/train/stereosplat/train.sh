Train_Stereosplat_On_KITTI360(){
cd ../../..

accelerate_config_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/accelerate_configs/accelerate_config_singleGPU.yaml"
configs_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
work_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/use_gt_views/"
output_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/train_visualization/stereosplat/Input_View_Invariant/use_gt_views/"
resume_from="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
# Optional overrides (leave empty to use cfg defaults)
exp_name="input_invariant_stereosplat_kitti360_stereo_114x544"
datapath="/data1/StereoDatasets/KITTI/KITTI360"
train_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/all.txt"
val_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/demo.txt"
test_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/demo.txt"
sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstLIDAR"
supp_view_nums=6
world_center="First_LiDAR_3_Uniform" 
unimatch_weights_path="/data1/zliu/feedforward_outputs_new/depth_estimation_224x840/checkpoint-90000/model.safetensors"


use_wandb=false
# W&B settings (only used when use_wandb=true)
# 推荐：不要把 key 写进脚本；更安全的做法是先执行 `wandb login` 或在 shell 里 export WANDB_API_KEY
wandb_api_key="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"
wandb_entity="liuzihua1004"
wandb_project="StereoSplat"
# online | offline | disabled
wandb_mode="online"
wandb_run_name="input_invariant_stereosplat_kitti360_default"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export PYTHONPATH="$(pwd):${PYTHONPATH}"

pixi run -e cu118 accelerate launch --config-file $accelerate_config_path trainer/train_kitti360_stereosplat.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from \
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


Train_Stereosplat_On_KITTI360