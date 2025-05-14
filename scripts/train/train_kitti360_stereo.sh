TRAIN_KITTI360_OmniScene(){
cd ../..
cd codes
configs_path="/home/zliu/Desktop/Project2025/FeedStereoGS/codes/configs/OmniScene/omni_gs_kitti360_stereo_r50_224x840.py"
work_dir="/media/zliu/data12/dataset/KITTI/VSRD_Format/feedstereo_outputs/omni_gs_kitti360_novelview_r50_224x840"
resume_from="/media/zliu/data12/outputs/omni_gs_kitti360_novelview_r50_224x840/omni_gs_kitti360_stereo_r50_224x804/pretrain/checkpoint-12000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch train_kitti360_stereo.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

TRAIN_KITTI360_OmniScene

