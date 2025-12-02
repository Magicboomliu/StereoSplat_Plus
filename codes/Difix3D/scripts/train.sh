Train_Diffix_Ref_One_Single_GPU(){
cd ..

output_dir="/data4/zliu/Difix3D/"



accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/difix/train \
    --dataset_path="data/data.json" \
    --max_train_steps 10000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 --eval_freq 1000 --viz_freq 100 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train" --timestep 199
}


