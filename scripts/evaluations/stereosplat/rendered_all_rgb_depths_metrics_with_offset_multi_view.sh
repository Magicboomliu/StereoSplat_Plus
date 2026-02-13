StereoSplat_Multi_View_Input_6_Views(){

cd ../../..
cd codes/Validation

configs_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others/StereoSplat_6Views/forward_views"
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml stereosplat/rendered_all_rgbs_and_depths_inside_bin_multiview6.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    # --output_vis
}

StereoSplat_Multi_View_Input_4_Views(){

cd ../../..
cd codes/Validation

configs_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others/StereoSplat_4Views/forward_views"
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml stereosplat/rendered_all_rgbs_and_depths_inside_bin_multiview4.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    # --output_vis
}


# StereoSplat_Multi_View_Input_4_Views
StereoSplat_Multi_View_Input_6_Views