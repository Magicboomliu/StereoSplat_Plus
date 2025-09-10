TRAIN_VolumeFusion_Revision(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision.py"
work_dir="/data1/zliu/feedforward_outputs_revision/VolumeFusion/FirwstCAM_As_Ref/saved_models"
# resume_from="/data1/zliu/temp_for_0617/VolumeFusion/Current_Version/checkpoint-90000/"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_volumefusion_revision.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from


}


TRAIN_VolumeFusion_Revision