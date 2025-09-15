Get_Rendered_RGBS_Depths_Metrics(){
cd ../../..
cd /home/zliu/Project2025/FeedStereoGS/codes/Validation

config_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision.py"
output_folder="/home/zliu/Project2025/EvaluationResults/20250915/VolumeFusion/NoOffset/2View_2Match"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"

demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR"
pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/2view_2matching_112x544/checkpoint-30000/"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumeufison/rendered_all_rgbs_and_depths_inside_bin.py \
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