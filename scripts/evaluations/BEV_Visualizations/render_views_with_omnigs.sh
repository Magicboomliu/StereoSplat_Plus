Get_Rendered_RGBS_Depths_Metrics(){
cd ../../..
cd codes/Validation

configs_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/OmniScene/eval/omnigs_vanilla_first_LiDAR_as_reference.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others/OmniScene_2Views/bev_views/"
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="MetricV2" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/OmniScene/checkpoints/"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml omnigs/rendered_novel_bev_views.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    --output_vis

}

Get_Rendered_RGBS_Depths_Metrics