Evaluation_The_Difix3D_Model_On_Validation_Set(){

cd ../../..
cd codes/Validation



config_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others/Diff-StereoSplatV2/forward_views_alls_no_diffix"
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/val.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
finetuned_difix3d_dataset_path="/data1/zliu/IROS26/Compared_With_Others/Diff-StereoSplat/difix_finetuning_dataset/"
pretrained_diffix_model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
prompt="remove degradation"
timestep=199


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file gpu_config_0.yaml difix3d/validation_evaluation.py \
    --config_path  "$config_path" \
    --output_folder "$output_folder" \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --timestep $timestep \
    --pretrained_diffix_model_path $pretrained_diffix_model_path \
    --finetuned_difix3d_dataset_path $finetuned_difix3d_dataset_path \
    --prompt "$prompt" \
    --use_ref \


}


Evaluation_The_Difix3D_Model_On_Validation_Set