Evaluate_The_Depth_Of_Unimatch(){
cd ../../..
cd codes/validation
configs_path="/home/Desktop/Project2025/FeedStereoGS/codes/configs/DepthSplat/unimatch_depth_only.py"
output_dir="/data/feedforward_outputs/visualization_outputs/unimatch_depth/"
resume_from="/data/feedforward_outputs/output_models/Unimatch/checkpoint-90000/"

 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml unimatch/depth_quality_analysis.py \
    --py-config $configs_path \
    --output_dir  $output_dir \
    --resume-from $resume_from \
    # --output_vis

}



Evaluate_The_Depth_Of_Unimatch