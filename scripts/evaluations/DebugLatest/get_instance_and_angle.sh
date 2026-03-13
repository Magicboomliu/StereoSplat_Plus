Get_Camera_And_Ground_Angle(){

cd ../../..
cd codes/Validation

config_path="/home/zliu/IROS2026/Diff-StereoSplat/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others/Diff-StereoSplatV2/Grounded_Angles"
val_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/all_sequential.txt"
demo_filelist="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml stereosplat/get_ground_and_camera_angle.py \
    --config_path  $config_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type 
    # --use_diffix3d \


}


Get_Camera_And_Ground_Angle