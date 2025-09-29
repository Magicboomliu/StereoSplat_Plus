Generate_Diff_Datasets(){
cd ../../..
cd /home/zliu/Project2025/FeedStereoGS/codes/DiffusionDatasetConfiguration

config_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_v2.py"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/all_2013_05_28_drive_0000_sync.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

# pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/2view_2matching_112x544/checkpoint-30000/"
#pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/4view_3matching_112x544/checkpoint-30000/"
pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/6view_4matching_112x544_RandomSample/checkpoint-39000/"
view_nums=2
matching_nums=2
output_folder="/data3/zliu/CVPR25/GSEnhanceDataset/KITTI360/Subset$view_nums"_"$matching_nums"



export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumefusion/generate_low_quality_gt_pairs.py \
    --config_path  $config_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --view_nums $view_nums \
    --matching_nums $matching_nums \
    # --output_vis

}

Generate_Diff_Datasets