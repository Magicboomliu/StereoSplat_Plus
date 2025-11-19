evaluation_with_local_pretrained_model(){
cd ../../..
cd codes/models_lab/diffix3D/validations

ablation_type="local_finetuning_no_6_no_4_views"
model_path="/data3/zliu/CVPR25/OutputModels/KITTI360_NOGrammLoss_Filtered/checkpoints/model_60001.pkl"
input_image="/home/zliu/Project2025/Difix3D/FeedStereoGS/assets/center_stereo.png"
prompt="remove degradation"
output_dir="/data3/zliu/CVPR25/Evaluations/$ablation_type/"
timestep=199
root_dir="/data3/zliu/CVPR25/GSEnhanceDataset/KITTI360/"
validation_filename_path="/home/zliu/Project2025/Difix3D/FeedStereoGS/filenames/KITTI360/train_val_data_split.json"

CUDA_VISIBLE_DEVICES=1 python evaluation_with_local_train_checkpoints.py\
    --model_path "$model_path" \
    --prompt "$prompt" \
    --output_dir "$output_dir" \
    --timestep 199 \
    --root_dir "$root_dir" \
    --validation_filename_path "$validation_filename_path" \
    --ablation_type "$ablation_type" \
    --vis

}

evaluation_with_local_pretrained_model