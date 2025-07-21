IncrementalFusion(){
cd ../../..

cd /home/zliu/Project2025/FeedStereoGS/Step2FusionCodes/evaluations

config_path="/home/zliu/Project2025/FeedStereoGS/Step2FusionCodes/configs/Models_Lab/VolumeFusion/eval/no_op.py"
output_folder="/data1/zliu/feedforward_outputs_fusion/VolumeFusion/No_Operation/"
val_filelist="/home/zliu/Project2025/FeedStereoGS/Step2FusionCodes/filenames/raw_filenames/2013_05_28_drive_0000_sync_list.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/Step2FusionCodes/filenames/raw_filenames/2013_05_28_drive_0000_sync_list.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR"
pretrained_model_path="/data1/zliu/feedforward_outputs_new/Depthsplat/first_lidar_as_ref/checkpoint-150000/"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumefusion/volumefusion_incremental_fusion.py \
    --config_path  $config_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --output_vis

}





IncrementalFusion