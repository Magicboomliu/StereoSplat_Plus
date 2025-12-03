Train_Diffix_No_Ref_One_Single_GPU(){
cd ..
pretrained_model_name_or_path="nvidia/difix"
pretrained_model_path=None
output_dir="/data4/zliu/Difix3D/Pretrained_Models/Fintune_Difix_No_Ref"
dataset_path="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
max_train_steps=200000

# 分成两个变量，而不是数组
resolution_h=112
resolution_w=544
learning_rate=2e-5
train_batch_size=1
dataloader_num_workers=8
enable_xformers_memory_efficient_attention=True
checkpointing_steps=10000
print_freq=10
eval_freq=5000
viz_freq=100
lambda_lpips=1.0
lambda_l2=1.0
lambda_gram=0.001
gram_loss_warmup_steps=2000
report_to="wandb"
tracker_project_name="difix_no_ref"
tracker_run_name="train"
timestep=199

accelerate launch --mixed_precision=bf16 --config-file accelerate_config_singleGPU.yaml src/train_difix_no_ref.py \
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
    --lambda_lpips "$lambda_lpips" \
    --lambda_l2 "$lambda_l2" \
    --lambda_gram "$lambda_gram" \
    --gram_loss_warmup_steps "$gram_loss_warmup_steps" \
    --report_to "$report_to" \
    --tracker_project_name "$tracker_project_name" \
    --tracker_run_name "$tracker_run_name" \
    --timestep "$timestep" \
    --use_wandb
}



Train_Diffix_No_Ref_Multiple_GPUs(){
cd ..

pretrained_model_name_or_path="nvidia/difix"
pretrained_model_path=None
output_dir="/data4/zliu/Difix3D/Pretrained_Models/Fintune_Difix_No_Ref_All"
dataset_path="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
max_train_steps=200000

# 分成两个变量，而不是数组
resolution_h=112
resolution_w=544

learning_rate=2e-5
train_batch_size=4
dataloader_num_workers=8
enable_xformers_memory_efficient_attention=True
checkpointing_steps=10000
print_freq=10
eval_freq=5000
viz_freq=100
lambda_lpips=1.0
lambda_l2=1.0
lambda_gram=0.001
gram_loss_warmup_steps=2000
report_to="wandb"
tracker_project_name="difix_no_ref"
tracker_run_name="train"
timestep=199
main_process_port=28505

accelerate launch --mixed_precision=bf16 --main_process_port $main_process_port --config-file accelerate_config_multiGPU.yaml src/train_difix_no_ref.py \
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
    --lambda_lpips "$lambda_lpips" \
    --lambda_l2 "$lambda_l2" \
    --lambda_gram "$lambda_gram" \
    --gram_loss_warmup_steps "$gram_loss_warmup_steps" \
    --report_to "$report_to" \
    --tracker_project_name "$tracker_project_name" \
    --tracker_run_name "$tracker_run_name" \
    --timestep "$timestep" \
    --use_wandb
}

# Train_Diffix_No_Ref_One_Single_GPU

Train_Diffix_No_Ref_Multiple_GPUs