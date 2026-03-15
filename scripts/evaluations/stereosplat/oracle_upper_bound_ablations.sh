Oracle_Upper_Bound_Ablations_SET01(){
cd ../../..
cd codes/Validation

pseudo_ratio="0.5 1.0"
output_folder="/data1/zliu/IROS26/Compared_With_Others/StereoSplat_2Views/Debug/Oracle_Upper_Bound_Ablations/0.5_1.0"
configs_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
# 输出目录用固定名，避免 ${pseudo_ratio} 展开成 "0.25 0.5" 导致多出一个裸参数
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
# diffix3d part
pretrained_diffix_model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
prompt="remove degradation"
timestep=199


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file gpu_config_0.yaml stereosplat/oracle_upper_bound_ablation.py \
    --config_path  "$configs_path" \
    --output_folder "$output_folder" \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --pseudo_ratio $pseudo_ratio \
    --timestep $timestep \
    --pretrained_diffix_model_path $pretrained_diffix_model_path \
    --prompt "$prompt" \
    --use_diffix3d \
    --use_ref \
    # --output_vis

}


Oracle_Upper_Bound_Ablations_SET02(){


cd ../../..
cd codes/Validation

pseudo_ratio="0.25 0.5"
output_folder="/data1/zliu/IROS26/Compared_With_Others/StereoSplat_2Views/Debug/Oracle_Upper_Bound_Ablations/0.25_0.5"
configs_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
# 输出目录用固定名，避免 ${pseudo_ratio} 展开成 "0.25 0.5" 导致多出一个裸参数
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
# diffix3d part
pretrained_diffix_model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
prompt="remove degradation"
timestep=199


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file gpu_config_1.yaml stereosplat/oracle_upper_bound_ablation.py \
    --config_path  "$configs_path" \
    --output_folder "$output_folder" \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --pseudo_ratio $pseudo_ratio \
    --timestep $timestep \
    --pretrained_diffix_model_path $pretrained_diffix_model_path \
    --prompt "$prompt" \
    --use_diffix3d \
    --use_ref \
    # --output_vis

}


Oracle_Upper_Bound_Ablations_SET03(){


cd ../../..
cd codes/Validation

pseudo_ratio="0.125 0.25"
output_folder="/data1/zliu/IROS26/Compared_With_Others/StereoSplat_2Views/Debug/Oracle_Upper_Bound_Ablations/0.125_0.25"
configs_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
# 输出目录用固定名，避免 ${pseudo_ratio} 展开成 "0.25 0.5" 导致多出一个裸参数
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
# diffix3d part
pretrained_diffix_model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
prompt="remove degradation"
timestep=199


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file gpu_config_2.yaml stereosplat/oracle_upper_bound_ablation.py \
    --config_path  "$configs_path" \
    --output_folder "$output_folder" \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --pseudo_ratio $pseudo_ratio \
    --timestep $timestep \
    --pretrained_diffix_model_path $pretrained_diffix_model_path \
    --prompt "$prompt" \
    --use_diffix3d \
    --use_ref \
    # --output_vis

}


Oracle_Upper_Bound_Ablations_SET04(){


cd ../../..
cd codes/Validation

pseudo_ratio="0.33 0.66"
output_folder="/data1/zliu/IROS26/Compared_With_Others/StereoSplat_2Views/Debug/Oracle_Upper_Bound_Ablations/0.33_0.66"
configs_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
# 输出目录用固定名，避免 ${pseudo_ratio} 展开成 "0.25 0.5" 导致多出一个裸参数
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
# diffix3d part
pretrained_diffix_model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
prompt="remove degradation"
timestep=199


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file gpu_config_3.yaml stereosplat/oracle_upper_bound_ablation.py \
    --config_path  "$configs_path" \
    --output_folder "$output_folder" \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --pseudo_ratio $pseudo_ratio \
    --timestep $timestep \
    --pretrained_diffix_model_path $pretrained_diffix_model_path \
    --prompt "$prompt" \
    --use_diffix3d \
    --use_ref \
    # --output_vis

}


# Oracle_Upper_Bound_Ablations_SET01
# Oracle_Upper_Bound_Ablations_SET02
# Oracle_Upper_Bound_Ablations_SET03
Oracle_Upper_Bound_Ablations_SET04