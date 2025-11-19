TRAIN_KITTI360_OmniScene(){
cd ../..
cd codes
#configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/OmniScene/omni_gs_first_as_input_supp3_metric3dv2.py"
#work_dir="/gs/FeedForwardGS/OmniScene/First_As_Input/Baseline_Supp3_Metric3Dv2/saved_models"
#resume_from="/gs/FeedForwardGS/OmniScene/First_As_Input/Baseline_Supp3_Metric3Dv2/saved_models/checkpoint-30000/"

configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/OmniScene/omni_gs_first_as_input_supp3_nmrfstereo.py"
work_dir="/gs/FeedForwardGS/OmniScene/First_As_Input/Baseline_Supp3_NMRFStereo/saved_models"
resume_from="/gs/FeedForwardGS/OmniScene/First_As_Input/Baseline_Supp3_NMRFStereo/saved_models/checkpoint-30000/"

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
configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/unimatch_depth_only.py"
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

configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/DepthSplat/depthsplat_First_As_Input_Supp3_DepthEst_RGB_loss.py"
work_dir="/gs/FeedForwardGS/DepthSplat/First_As_Input/Baseline_DepthEst_RGB_Loss/saved_models"
resume_from="/gs/FeedForwardGS/DepthSplat/First_As_Input/Baseline_DepthEst_RGB_Loss/saved_models/checkpoint-18000/"

#configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/DepthSplat/depthsplat_First_As_Input_Supp3_RGB_Loss_Only.py"
#work_dir="/gs/FeedForwardGS/DepthSplat/First_As_Input/Baseline_RGB_Loss_Only/saved_models"
#resume_from="/gs/FeedForwardGS/DepthSplat/First_As_Input/Baseline_DepthEst_RGB_Loss/saved_models/checkpoint-18000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_depthsplat.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}


TRAIN_KITTI360_DepthSplatRevised(){
cd ../..
cd codes

configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/DepthSplat/depthsplat_gs_revised.py"
work_dir="/gs/FeedForwardGS/DepthSplat/Revised/First_As_Input/Baseline_DepthEst_RGB_Loss/saved_models"
resume_from="none"

#configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/DepthSplat/depthsplat_First_As_Input_Supp3_RGB_Loss_Only.py"
#work_dir="/gs/FeedForwardGS/DepthSplat/First_As_Input/Baseline_RGB_Loss_Only/saved_models"
#resume_from="/gs/FeedForwardGS/DepthSplat/First_As_Input/Baseline_DepthEst_RGB_Loss/saved_models/checkpoint-18000/"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_depthsplat_revised.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}




TRAIN_KITTI360_VolumeFusion(){
cd ../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Tsubame_Version/Models_Lab/VolumeFusion/VolumeFusion/volumefusion_configs.py"
work_dir="/gs/FeedForwardGS/VolumeFusion/First_As_Input/Baseline_Volume_and_Final_Supp/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_stereo_volumefusion.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}





TRAIN_KITTI360_DepthSplat_Revised_FirstCAM_Ref(){
cd ../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_revised_firstcam_as_ref.py"
work_dir="/gs/FeedForwardGS/DepthSplat/First_As_Input_FirstCam_As_Ref/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 accelerate launch --config-file accelerate_config_singleGPU.yaml train_kitti360_sterep_depthsplat_revised_firstcam_as_reference.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}


TRAIN_KITTI360_VolumeFusion_Revised_FirstCAM_Ref(){
cd ../..
cd codes
configs_path="/home/2/ux04482/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_configs_firstasCam.py"
work_dir="/gs/FeedForwardGS/VolumeFusion/First_As_Input_FrstCAM_As_Ref/saved_models"
resume_from="None"

#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1  accelerate launch  --config-file accelerate_config.yaml train_kitti360_stereo_volumefusion_revised_firstcam_as_reference.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from
}


TRAIN_KITTI360_VolumeFusion_Revised_FirstCAM_Ref
TRAIN_KITTI360_DepthSplat_Revised_FirstCAM_Ref



#TRAIN_KITTI360_VolumeFusion
# TRAIN_KITTI360_DepthSplatRevised
# TRAIN_KITTI360_VolumeFusion
# TRAIN_KITTI360_DepthSplat
#TRAIN_KITTI360_OmniScene
# TRAIN_KITTI360_Unimatch_Only
