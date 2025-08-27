Baseline_IncrementalFusion(){
cd ../../..

cd /home/zliu/Project2025/FeedStereoGS/codes/evaluations

config_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/eval/no_op.py"
output_folder="/data1/zliu/feedforward_outputs_fusion/VolumeFusion/ForVis/"
pretrained_model_path="/data1/zliu/feedforward_outputs_new/VolumeFusion/First_LiDAR_As_Ref_No_Offset/saved_models/checkpoint-150000/"
semi_global_map="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/semi_global_maps/2013_05_28_drive_0000_sync"
ablation_type="no_fusion_as_one_version"
# ablation_type="GT"
#ablation_type="no_fusion"
# ablation_type='no_fusion_as_one_version'

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumefusion/baseline_fusion.py \
    --config_path  $config_path \
    --output_folder $output_folder \
    --pretrained_model_path $pretrained_model_path \
    --semi_global_map $semi_global_map \
    --ablation_type $ablation_type \
    --output_vis

}
IncrementalFusion(){

cd /home/zliu/Project2025/FeedStereoGS/codes/evaluations

config_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/eval/optimize_fusion.py"
output_folder="/data1/zliu/feedforward_outputs_fusion/VolumeFusion/OptimizeFusion/Windows_Optimization_Only/Plus_Alpha_Geometry_Gradient"
pretrained_model_path="/data1/zliu/feedforward_outputs_new/VolumeFusion/First_LiDAR_As_Ref_No_Offset/saved_models/checkpoint-150000/"
semi_global_map="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/semi_global_maps/2013_05_28_drive_0000_sync"
ablation_type="incremental_fusion"

geometry_based_pruning_depth_error_threshold=0.05
gradient_based_pruning_keep_ratio=0.10
opacity_based_pruning_threshold=0.10

window_optimization_iterations=50
global_optimization_iterations=20
lambda_depth=0.1

# ablation_type="GT"
#ablation_type="no_fusion"
# ablation_type='no_fusion_as_one_version'



export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumefusion/baseline_incremental_fusion.py \
    --config_path  $config_path \
    --output_folder $output_folder \
    --pretrained_model_path $pretrained_model_path \
    --semi_global_map $semi_global_map \
    --ablation_type $ablation_type \
    --output_vis \
    --use_window_loss_based_optimization \
    --window_optimization_iterations $window_optimization_iterations \
    --lambda_depth $lambda_depth \
    --use_pruning \
    --use_opacity_based_pruning \
    --opacity_based_pruning_threshold $opacity_based_pruning_threshold \
    --use_gradient_based_pruning \
    --gradient_based_pruning_keep_ratio $gradient_based_pruning_keep_ratio \
    --use_geometry_based_pruning \
    --geometry_based_pruning_depth_error_threshold $geometry_based_pruning_depth_error_threshold \





}



IncrementalFusion

# Baseline_IncrementalFusion