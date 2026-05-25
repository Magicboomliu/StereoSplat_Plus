StereoSplat_2Views_Eval(){
cd ../../..

REPO="/home/zliu/IROS2026/Conf/StereoSplat_Plus/stereosplat"

accelerate_config_path="${REPO}/accelerate_configs/inference/gpu_0.yaml"
configs_path="${REPO}/src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
output_folder="/data1/zliu/IROS26/Compared_With_Others_Pixi/results/stereosplat/with_conf/"
val_filelist="${REPO}/filenames/kitti360/train_complete/val.txt"
demo_filelist="${REPO}/filenames/kitti360/train_complete/demo.txt"
ablation_type="NMRFStereo"
dataset_type="First_LiDAR_3_Uniform"

pretrained_model_path="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/with_conf/checkpoint-250000"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd):${PYTHONPATH}"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
pixi run -e cu118 python -m accelerate.commands.launch --config_file $accelerate_config_path validator/stereosplat/rendered_views_inside_bin.py \
    --config_path  $configs_path \
    --output_folder $output_folder \
    --val_filelist $val_filelist \
    --demo_filelist $demo_filelist \
    --ablation_type $ablation_type \
    --dataset_type $dataset_type \
    --pretrained_model_path $pretrained_model_path \
    # --output_vis

}

StereoSplat_2Views_Eval
