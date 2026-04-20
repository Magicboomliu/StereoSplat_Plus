
Post_Fusion_UpperBound() {
# stereosplat/scripts/evaluation/stereosplat_plus/dev/ -> four parents up = stereosplat/ (pixi project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREOSPLAT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$STEREOSPLAT_ROOT" || exit 1


accelerate_config_path="${STEREOSPLAT_ROOT}/accelerate_configs/inference/gpu_1.yaml"
validator_script="validator/stereosplat_plus/posed_post_gaussain_fusion.py"
fusion_mode="stereo_fov_depth60_rpix1_repalce"

pseudo_ratio="0.5 1.0"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat_plus/dev/posed_post_gaussain_fusion/$fusion_mode/$pseudo_ratio"
configs_path="${STEREOSPLAT_ROOT}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
val_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${STEREOSPLAT_ROOT}/filenames/kitti360/train_complete/demo_more.txt"
ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"
pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
# the pre-trained the difix3d model
#pretrained_diffix_model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
pretrained_diffix_model_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
prompt="remove degradation"
timestep=199

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 \
pixi run -e cu118 accelerate launch --config-file "$accelerate_config_path" "$validator_script" \
--config_path "$configs_path" \
--output_folder "$output_folder" \
--val_filelist "$val_filelist" \
--demo_filelist "$demo_filelist" \
--ablation_type "$ablation_type" \
--dataset_type "$dataset_type" \
--pretrained_model_path "$pretrained_model_path" \
--pseudo_ratio $pseudo_ratio \
--timestep "$timestep" \
--pretrained_diffix_model_path "$pretrained_diffix_model_path" \
--prompt "$prompt" \
--use_diffix3d \
--use_ref \
--fusion_mode "$fusion_mode"
# --use_gt_view \
# Optional: --output_vis
}


Post_Fusion_UpperBound