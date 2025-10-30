Get_Rendered_RGBS_Depths_Metrics(){
cd ../../..
cd /home/Project2025/FeedStereoGS/codes/Validation

configs_path="/home/Project2025/FeedStereoGS/codes/configs/PixelSplat/pixelsplat_vanilla_first_lidar.py"
output_folder="/data/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/PixelSplat/2Views/First_LiDAR_As_Ref/NMRFStereo/"
val_filelist="/home/Project2025/FeedStereoGS/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/Project2025/FeedStereoGS/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR"
pretrained_model_path="/data/zliu/feedforward_outputs_revision/PixelSplat_2Views/First_LiDAR_As_Ref/saved_models/checkpoint-126000/"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml pixelspalt/rendered_all_forward_views_vanilla.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    # --output_vis

}

Get_Rendered_RGBS_Depths_Metrics