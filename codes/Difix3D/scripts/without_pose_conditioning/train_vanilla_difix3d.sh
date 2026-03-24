Train_Difix_Revision_Ref_Single_GPU_Without_Relative_Pose() {
cd ../..
pretrained_model_name_or_path="nvidia/difix_ref"
pretrained_model_path="None"
output_dir="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20"
# JSON：顶层为 train/test，或 training/validation（PairedDataset 已兼容 make_near_view_finetuning 输出）
dataset_path="/home/zliu/IROS2026/StereoSplat_Plus/codes/Difix3D/filenames/Validation_Set/all_results_dict.json"

max_train_steps=120000
resolution_h=112
resolution_w=544

learning_rate=2e-5
train_batch_size=4
dataloader_num_workers=8
checkpointing_steps=10000
print_freq=10
eval_freq=5000
viz_freq=100
num_samples_eval=100

lambda_l2=1.0
lambda_ssim=0.5

report_to="wandb"
tracker_project_name="DIFIX3D_Finetuning"
tracker_run_name="vanilla_difix3d_psnr20"
timestep=199

accelerate launch --mixed_precision=bf16 --config-file gpu_config_2.yaml \
src/train_difix_revision_with_ref.py \
--pretrained_model_name_or_path "$pretrained_model_name_or_path" \
--pretrained_model_path "$pretrained_model_path" \
--output_dir "$output_dir" \
--dataset_path "$dataset_path" \
--max_train_steps "$max_train_steps" \
--resolution_h "$resolution_h" \
--resolution_w "$resolution_w" \
--learning_rate "$learning_rate" \
--print_freq "$print_freq" \
--train_batch_size "$train_batch_size" \
--dataloader_num_workers "$dataloader_num_workers" \
--checkpointing_steps "$checkpointing_steps" \
--eval_freq "$eval_freq" \
--viz_freq "$viz_freq" \
--num_samples_eval "$num_samples_eval" \
--lambda_l2 "$lambda_l2" \
--lambda_ssim "$lambda_ssim" \
--report_to "$report_to" \
--tracker_project_name "$tracker_project_name" \
--tracker_run_name "$tracker_run_name" \
--timestep "$timestep" \
--enable_xformers_memory_efficient_attention \
# --use_wandb


}


Train_Difix_Revision_Ref_Multi_GPU() {
  cd "$(dirname "$0")/.."

  pretrained_model_name_or_path="nvidia/difix_ref"
  pretrained_model_path="None"
  output_dir="/data4/zliu/Difix3D/Pretrained_Models/Finetune_Difix_Revision_Ref_multi"
  dataset_path="/home/zliu/IROS2026/StereoSplat_Plus/codes/Difix3D/filenames/all_results_dict.json"

  max_train_steps=200000
  resolution_h=112
  resolution_w=544

  learning_rate=2e-5
  train_batch_size=4
  dataloader_num_workers=8
  checkpointing_steps=10000
  print_freq=10
  eval_freq=5000
  viz_freq=100
  num_samples_eval=100

  lambda_l2=1.0
  lambda_ssim=0.5

  report_to="wandb"
  tracker_project_name="difix_revision_ref_multi"
  tracker_run_name="train"
  timestep=199
  main_process_port=29502

  accelerate launch --mixed_precision=bf16 --main_process_port "$main_process_port" \
    --config-file accelerate_config_multiGPU.yaml \
    src/train_difix_revision_with_ref.py \
    --pretrained_model_name_or_path "$pretrained_model_name_or_path" \
    --pretrained_model_path "$pretrained_model_path" \
    --output_dir "$output_dir" \
    --dataset_path "$dataset_path" \
    --max_train_steps "$max_train_steps" \
    --resolution_h "$resolution_h" \
    --resolution_w "$resolution_w" \
    --learning_rate "$learning_rate" \
    --print_freq "$print_freq" \
    --train_batch_size "$train_batch_size" \
    --dataloader_num_workers "$dataloader_num_workers" \
    --checkpointing_steps "$checkpointing_steps" \
    --eval_freq "$eval_freq" \
    --viz_freq "$viz_freq" \
    --num_samples_eval "$num_samples_eval" \
    --lambda_l2 "$lambda_l2" \
    --lambda_ssim "$lambda_ssim" \
    --report_to "$report_to" \
    --tracker_project_name "$tracker_project_name" \
    --tracker_run_name "$tracker_run_name" \
    --timestep "$timestep" \
    --enable_xformers_memory_efficient_attention
    # --use_wandb
}

Train_Difix_Revision_Ref_Single_GPU_Without_Relative_Pose
