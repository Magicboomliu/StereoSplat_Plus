TRAIN_KITTI360_OmniScene(){
cd ../..
cd codes
configs_path="configs/OmniScene/omni_gs_kitti360_novelview_r50_224x400.py"
work_dir="outputs/omni_gs_kitti360_novelview_r50_224x840"
resume_from="None"


CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config-file accelerate_config.yaml train_kitti360_stereo.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from

}

TRAIN_KITTI360_OmniScene

