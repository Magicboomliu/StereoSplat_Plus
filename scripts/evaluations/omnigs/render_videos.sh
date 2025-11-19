Get_Rendered_Videos(){
cd ../../..
cd /home/zliu/Project2025/FeedStereoGS/codes/Validation

config_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/OmniScene/eval/omnigs_vanilla_first_LiDAR_as_reference.py"
output_folder="/home/zliu/Project2025/EvaluationResults/20250714/OmniGS/First_LiDAR_As_Ref/MetricV2/"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
ablation_type="MetricV2" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR"
pretrained_model_path="/data1/zliu/feedforward_outputs_new/OmniScene/MetricV2/First_LiDAR_As_Ref/"
#pretrained_model_path="/data1/zliu/temp_for_0617/OmniScene/NMRFStereo/checkpoint-45000/"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml omnigs/rendered_videos.py \
    --config_path  $config_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path 

}

Get_Rendered_Videos