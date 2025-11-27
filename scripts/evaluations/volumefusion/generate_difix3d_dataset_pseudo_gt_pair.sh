Generated_Finetuning_Dataset(){
cd ../../..
cd /home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Validation

configs_path="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
output_folder="/data1/zliu/KITTI360_Completed/KITTI360_Degradtations_Pairs"
train_filelist="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/train_complete/train.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
iterations=2


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=0 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumeufison/rendered_views_for_diffix3d_training.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --train_filelist $train_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --iterations $iterations

}
Generated_Finetuning_Dataset