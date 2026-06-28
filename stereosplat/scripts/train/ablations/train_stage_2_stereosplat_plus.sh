# =============================================================================
# Variant 2: SELF-BOOTSTRAP (one unified model, --self_pseudo)
#   - ONE model; no separate frozen Stage1.
#   - Two separate parameters with distinct roles:
#       stage_1_model_path : FIRST LAUNCH ONLY — loads model weights only
#                            (no optimizer / scheduler / global_iter)
#       resume_from        : RESUME — restores everything (model + optimizer +
#                            scheduler + global_iter) from an accelerate checkpoint.
#                            Set to "latest" to find the newest checkpoint automatically.
#                            Set to "" or "None" to skip resume.
#   Typical workflow:
#     First launch  → stage_1_model_path=<Stage1 ckpt>, resume_from=""
#     Resume run    → stage_1_model_path=<same>, resume_from="latest"
#                     (accelerator.load_state overrides stage_1 init automatically)
# =============================================================================
Train_StereoSplat_Stage2_Self_Pseudo(){
cd ../../..

REPO="/home/zliu/IROS2026/Conf/StereoSplat_Plus/stereosplat"

accelerate_config_path="${REPO}/accelerate_configs/accelerate_config.yaml"
configs_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
# work_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_debug/"
# output_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/train_visualization/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_debug/"

work_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/ablations/stage2_self_pseudo/"
output_dir="/data1/zliu/IROS26/Compared_With_Others_Pixi/train_visualization/stereosplat/Input_View_Invariant/withconf/ablations/stage2_self_pseudo/"


# FIRST LAUNCH: set resume_from="" so no accelerate resume happens;
#               stage_1_model_path provides the initial model weights.
# RESUME RUN:   set resume_from to checkpoint dir; restores optimizer/scheduler/global_iter.
resume_from=""
# resume_from=""

# Weights-only init for first launch (ignored on resume since accelerator.load_state
# will override the model weights anyway, but harmless to keep set).
stage_1_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage1/ablations/checkpoint-85000"

# Optional overrides (leave empty to use cfg defaults)
exp_name="stereosplat_kitti360_stage2_self_pseudo_with_conf_and_difix3d"
datapath="/data1/StereoDatasets/KITTI/KITTI360"

# train_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
# val_filelist="${REPO}/filenames/kitti360/train_complete/val_tiny.txt"
# test_filelist="${REPO}/filenames/kitti360/train_complete/val_tiny.txt"

train_filelist="${REPO}/filenames/kitti360/trainval/all_2013_05_28_drive_0000_sync.txt"
val_filelist="${REPO}/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
test_filelist="${REPO}/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"


sequence='2013_05_28_drive_0000_sync'
data_version="bin_infos_8.0_FirstLIDAR"
supp_view_nums=6
world_center="First_Stage2"
unimatch_weights_path="/data1/zliu/feedforward_outputs_new/depth_estimation_224x840/checkpoint-90000/model.safetensors"

# Difix3D pretrained model (kept: pseudo views can still be enhanced)
pretrained_difix3d="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"

mix_psuedo_views_ratio=0.9   # 90% of view_num>2 iters use pseudo views
mix_difix3d_ratio=0.9        # 90% of pseudo-view iters apply Difix3D enhancement

use_wandb=true
wandb_api_key="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"
wandb_entity="liuzihua1004"
wandb_project="StereoSplat_Plus_Conf_Latest"
wandb_mode="online"
wandb_run_name="stereosplat_plus_ablations"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# Difix3D base weights are loaded from local HF cache; avoid network 504
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Multi-GPU: do NOT set CUDA_VISIBLE_DEVICES here; accelerate_config.yaml uses gpu 0,1,2,3

echo "[Launch] resume_from=${resume_from:-<none>}"

pixi run -e cu118 python -m accelerate.commands.launch --config_file $accelerate_config_path trainer/train_kitti360_stereosplat_stage2_with_difix3d.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    $([ -n "$resume_from" ] && echo --resume-from "$resume_from") \
    --stage_1_model_path $stage_1_model_path \
    --mix_psuedo_views_ratio $mix_psuedo_views_ratio \
    --mix_difix3d_ratio $mix_difix3d_ratio \
    --pretrained_difix3d $pretrained_difix3d \
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


# =============================================================================
# Pick ONE to run (comment / uncomment):
# =============================================================================
# Train_StereoSplat_Stage2_With_Conf_And_Difix3D       # two-model (default)
Train_StereoSplat_Stage2_Self_Pseudo               # self-bootstrap (--self_pseudo)
