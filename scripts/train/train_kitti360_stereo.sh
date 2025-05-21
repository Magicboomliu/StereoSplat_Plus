TRAIN_KITTI360_OmniScene(){
cd ../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/OmniScene/omni_gs_kitti360_stereo_r50_224x840_large_tpv.py"
work_dir="/gs/output_models/saved_models/large_tpv"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml


nvidia-smi

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml  train_kitti360_stereo.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

TRAIN_KITTI360_OmniScene
