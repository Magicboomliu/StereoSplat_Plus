Get_Rendered_RGBS_Depths_Metrics(){
cd ../../..
cd codes/Validation

configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/MVSplat/mvsplat_vanilla_first_lidar.py"
output_folder="/data1/zliu/forward_outputs_compared_with_others/mvsplat/bev_views"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR"
pretrained_model_path="/data1/zliu/feedforward_outputs_revision/MVSplat_2Views/First_LiDAR_As_Ref/saved_models/checkpoint-252000/"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml mvsplat/rendered_all_novel_bev_views.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --output_vis

}

Get_Rendered_RGBS_Depths_Metrics