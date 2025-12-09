Progressive_Twice_Diffix3D_Batch(){

cd ../../..
cd /home/zliu/Project2025/FeedforwardGS_Ablations/FeedStereoGS/codes/Validation

configs_path="/home/zliu/Project2025/FeedforwardGS_Ablations/FeedStereoGS/codes/configs/Ablations/volumefusion_train_randomly/configs.py"
output_folder="/data1/zliu/feedforward_outputs_ablations/Progressive_Inference/Updated_1208/Progressive_Twice_Diffix3D_Once/Testing"
val_filelist="/home/zliu/Project2025/FeedforwardGS_Ablations/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete_version2.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/KITTI360_Complete/RandomView_RandomSample/saved_models/checkpoint-159000"
prompt="remove degradation"
timestep=199

difix_remote_path="nvidia/difix"
pretrained_diffix_model_path="/data1/zliu/pretrained_foundataion_models/difix3d_lastest/difix_no_ref/hard/model_70001.pkl"
use_difix_type="local"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml ablations/volumefusion_with_difix3d_batch.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --pretrained_diffix_model_path $pretrained_diffix_model_path \
    --timestep $timestep \
    --prompt "$prompt" \
    --difix_remote_path $difix_remote_path \
    --use_difix_type $use_difix_type \
    --output_vis \
    --use_diffix3d \
    # --output_vis
}

Progressive_Twice_Diffix3D_Batch