Generated_New_validationSet(){
cd ../../..
cd /home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Validation

configs_path="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/configs/Ablations/volumefusion/configs.py"
output_folder="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/TEMP/badcases"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/feedforward_outputs_ablations/Training_2_Views/saved_models/checkpoint-57000/"
revision_validation_filelist="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete_version2.txt"
angle_threshold=15


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumeufison/validationset_filtering.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --revision_validation_filelist $revision_validation_filelist \
    --angle_threshold $angle_threshold \
    # --output_vis

}

Generated_New_validationSet