ProgressiveInference_Iteration_Twice(){
cd ../../..
cd /home/zliu/Project2025/FeedStereoGS/codes/Validation

config_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_v2.py"
output_folder="/home/zliu/Project2025/EvaluationResults/20250930/VolumeFusion/WithOffset/Progressive_With_Diffix3D/Iteration2Times_With_EnhanceTrick_Postprocessing"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/2_4_6_View_Variances/checkpoint-51000/"

# diffix3d part
pretrained_diffix_model_path="/data3/zliu/CVPR25/OutputModels/KITTI360_NOGrammLoss_Filtered/checkpoints/model_60001.pkl"
prompt="remove degradation"
timestep=199



export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumeufison/rendered_progressive_with_diffix.py\
    --config_path  $config_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --pretrained_diffix_model_path $pretrained_diffix_model_path \
    --timestep $timestep \
    --prompt "$prompt" \
    --use_ref \
    --use_diffix3d \
    --use_diffix3d_postprocessing \
    # --output_vis


}

ProgressiveInference_Iteration_Once(){
cd ../../..
cd /home/zliu/Project2025/FeedStereoGS/codes/Validation

config_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_v2.py"
output_folder="/home/zliu/Project2025/EvaluationResults/20250930/VolumeFusion/WithOffset/Progressive_With_Diffix3D_IterationOnce/IterationOnce_With_Enhancement_PostProcessing"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
demo_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/2_4_6_View_Variances/checkpoint-51000/"
# diffix3d part
pretrained_diffix_model_path="/data3/zliu/CVPR25/OutputModels/KITTI360_NOGrammLoss_Filtered/checkpoints/model_60001.pkl"
prompt="remove degradation"
timestep=199


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumeufison/rendered_progessive_with_diffix_iter_1.py\
    --config_path  $config_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --pretrained_diffix_model_path $pretrained_diffix_model_path \
    --timestep $timestep \
    --prompt "$prompt" \
    --use_ref \
    --use_diffix3d \
    --use_diffix3d_postprocessing \
    # --output_vis
    # --output_vis
    # --output_vis


}

#ProgressiveInference_Iteration_Twice

ProgressiveInference_Iteration_Once