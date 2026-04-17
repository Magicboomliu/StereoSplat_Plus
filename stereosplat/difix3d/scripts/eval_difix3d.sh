EVAL_Finetuned_Difix3D(){
cd ..
dataset_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/difix3d/filenames/Validation_Set/all_results_dict.json"
pretrained_path="/data4/zliu/Difix3D_Output_Results/Refined_Vanilla_Difix3D_PSNR20/checkpoints/model_130001.pkl"
saved_json_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/difix3d/output/finetuned_vanilla_difix3d_psnr20_eval.json"

CUDA_VISIBLE_DEVICES=2 pixi run python evals/eval_difix_ref_pipeline.py \
  --dataset_path $dataset_path \
  --split test \
  --pretrained_path $pretrained_path \
  --device cuda \
  --save_json $saved_json_path

}

EVAL_Finetuned_Difix3D_Old_Scheme(){
cd ..
dataset_path="/home/zliu/IROS2026/StereoSplat_Latest/StereoSplat_Plus/stereosplat/difix3d/filenames/Validation_Set/all_results_dict.json"
pretrained_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
saved_json_path="/home/zliu/IROS2026/StereoSplat_Plus/difix3d/output/old_vanilla_difix3d_psnr20_eval.json"

CUDA_VISIBLE_DEVICES=1 pixi run python evals/eval_difix_ref_pipeline.py \
  --dataset_path $dataset_path \
  --split test \
  --pretrained_path $pretrained_path \
  --device cuda \
  --save_json $saved_json_path\
  --deterministic_vae_encode \
  --deterministic_scheduler_step

}
EVAL_Finetuned_Difix3D
# EVAL_Finetuned_Difix3D_Old_Scheme
# EVAL_Finetuned_Difix3D
