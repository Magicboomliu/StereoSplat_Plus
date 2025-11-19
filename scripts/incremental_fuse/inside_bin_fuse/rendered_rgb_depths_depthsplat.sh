Get_Rendered_RGBS_Depths_Metrics(){

cd ../../../..
cd /home/zliu/Project2025/FeedStereoGS/codes/Incremental3DGS
config_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/eval/depthsplat_gs_revised_first_lidar_as_ref_no_offset.py"
output_folder="/home/zliu/Project2025/EvaluationResults/20250714/Incremental_Fusion/Internal_Bins/Depthsplat/First_LiDAR_As_Ref/"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR"
pretrained_model_path="/data1/zliu/feedforward_outputs_new/Depthsplat/first_lidar_as_ref/checkpoint-99000/"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml Depthsplat/inside_bin_fusion.py \
    --config_path  $config_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --output_vis

}

Get_Rendered_RGBS_Depths_Metrics