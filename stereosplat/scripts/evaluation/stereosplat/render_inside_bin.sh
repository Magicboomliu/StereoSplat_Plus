StereoSplat_2Views_Eval(){
cd ../../..
# Run from stereosplat project root
accelerate_config_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/accelerate_configs/inference/gpu_0.yaml"
configs_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat/2views_training_stage2/"
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/stage_2_psuedo_gt_mix_training/checkpoint-102000/"
# pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
pixi run -e cu118 accelerate launch --config-file $accelerate_config_path validator/stereosplat/rendered_views_inside_bin.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    # --output_vis

}

StereoSplat_2Views_Eval