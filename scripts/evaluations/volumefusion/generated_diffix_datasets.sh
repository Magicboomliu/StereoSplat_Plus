Generated_MultiViews_For_TrainingDiffix(){

cd ../../..
cd /home/zliu/Project2025/FeedStereoGS/codes/Validation

view_nums=6
matching_nums=4
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
output_folder="/data1/zliu/feedforward_outputs_revision/Evaluations/KITTI_Diffix3D_Training/$view_nums"View_$matching_nums"Match"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/train_complete/all.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/train_complete/all.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/KITTI360_Complete/RandomView_RandomSample/saved_models/checkpoint-159000/"



export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumeufison/rendered_views_for_diffix3d_training.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --view_nums $view_nums \
    --matching_nums $matching_nums
    # --output_vis

}

Generated_MultiViews_For_TrainingDiffix