TRAIN_KITTI360_OmniScene(){
cd ../..
cd codes
configs_path="/home/Desktop/Project2025/FeedStereoGS/codes/configs/OmniScene/omniscene_vanilla_settings.py"
work_dir="/data/feedforward_outputs/Vanilla_OmniScene"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_omnigs.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}


TRAIN_KITTI360_Unimatch_Only(){
cd ../..
cd codes
configs_path="/home/Desktop/Project2025/FeedStereoGS/codes/configs/DepthSplat/unimatch_depth_only.py"
work_dir="/data/feedforward_outputs/output_models/Unimatch/depth_estimation_224x840"
resume_from="/data/feedforward_outputs/output_models/Unimatch/checkpoint-90000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_unimatch_only.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}


TRAIN_KITTI360_DepthSplat(){
cd ../..
cd codes
configs_path="/home/Desktop/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_kitti360_stereo_224x840.py"
work_dir="/data/feedforward_outputs/output_models/DepthSplat/depth_splat_simple_224x840"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_depthsplat.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}


TRAIN_KITTI360_DepthSplat

# TRAIN_KITTI360_OmniScene
# TRAIN_KITTI360_Unimatch_Only