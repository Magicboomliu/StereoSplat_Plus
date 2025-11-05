Train_Cost_Volume_Branch_Only(){
cd ../../..
cd codes
configs_path="/home/zliu/Project2025/FeedforwardGS_Ablations/FeedStereoGS/codes/configs/Ablations/cost_volume_branch_only/configs.py"
work_dir="/data1/zliu/feedforward_outputs_ablations/Cost_Volume_Branch_Only/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch --config-file accelerate_config.yaml train_kitti360_volumefusion_cv_branch_only.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}


Train_Cost_Volume_Branch_Only