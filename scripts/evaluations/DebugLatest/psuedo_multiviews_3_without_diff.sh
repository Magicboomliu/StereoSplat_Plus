Ablation_Psuedo_MultiViews_3_Without_Difix3D_SETUP01(){
cd ../../..

cd codes/Validation

config_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others/StereoSplat_2Views/Debug/Ablation_Psuedo_MultiViews_3_Without_Difix3D/SETUP01"
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
#pretrained_model_path="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models/2_4_6_View_Variances/checkpoint-51000/"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
# diffix3d part
pretrained_diffix_model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
prompt="remove degradation"
timestep=199
pseudo_ratio="0.5 1.0"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file gpu_config_2.yaml Temp_Exp/stereosplat_plus_gt_pose_custom_view.py \
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
    --pseudo_ratio $pseudo_ratio \
    # --use_diffix3d \
    # --use_ref \
    # --output_vis
    # --use_diffix3d \
    # --output_vis
    # --output_vis

}


Ablation_Psuedo_MultiViews_3_Without_Difix3D_SETUP01