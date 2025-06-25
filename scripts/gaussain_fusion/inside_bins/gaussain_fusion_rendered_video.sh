GaussainFusion_Inside_BIN_DepthSplat_RenderVideos(){
cd ../../..
cd codes/playrgound/

configs_path="/home/zliu/Project2025/FeedStereoGS/codes/configs/DepthSplat/depthsplat_gs_revised.py"
work_dir="/data1/zliu/temp_for_0617/Vis_And_Evals/GaussainFusion/DepthSplat/BIN_INSIDE"
resume_from="/data1/zliu/temp_for_0617/DepthSplat/SH0_Version/checkpoint-99000/"
val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/demo.txt"
#val_filelist="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"

fusion_type="voxel_fusion" # "concat","None","voxel_fusion"
# demo.txt
# val_2013_05_28_drive_0000_sync_complete.txt
#configs 
# - Single GPU YAML: accelerate_config_singleGPU.yaml
# - Multi GPUs YAML: accelerate_config.yaml

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml GaussainFusion/inside_bin_gs_fusion_video_depthsplatSH0.py \
    --py-config $configs_path \
    --work-dir  $work_dir \
    --resume-from $resume_from \
    --val_filelist $val_filelist \
    --fusion_type $fusion_type \
    # --output_vis
}

GaussainFusion_Inside_BIN_DepthSplat_RenderVideos