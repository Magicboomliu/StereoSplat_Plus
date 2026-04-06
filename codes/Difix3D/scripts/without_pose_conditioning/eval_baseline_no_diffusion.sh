#!/usr/bin/env bash
# Baseline eval（无 diffusion / 无模型增强）：
# 直接把 dataset 里的 input(image) 当 prediction，与 target_image 计算 PSNR/SSIM/LPIPS
RUN_BASELINE_NO_DIFFUSION() {
  cd ../..

  input_test_filename="/home/zliu/IROS2026/StereoSplat_Plus/codes/Difix3D/filenames/Validation_Set/all_results_dict.json"
  split="test"

  ablation_study_name="baseline_no_diffusion"
  seed=42
  output_folder="/data4/zliu/Difix3D_Output_Results/Eval/$ablation_study_name"
  height=112
  width=544

  python src/difix_baseline_evaluation.py \
    --input_test_filename "$input_test_filename" \
    --split "$split" \
    --ablation_study_name "$ablation_study_name" \
    --seed "$seed" \
    --output_folder "$output_folder" \
    --height "$height" \
    --width "$width" \
    --eval_mode original_res
    # --max_samples 100
    # --eval_mode train_like
}

RUN_BASELINE_NO_DIFFUSION

