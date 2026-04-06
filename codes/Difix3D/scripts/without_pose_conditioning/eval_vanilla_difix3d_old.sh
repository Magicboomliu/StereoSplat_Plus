

RUN_EVAL_VANILLA_DIFIX3D_OLD_STOCHASTIC() {
  cd ../..

  input_test_filename="/home/zliu/IROS2026/StereoSplat_Plus/codes/Difix3D/filenames/Validation_Set/all_results_dict.json"
  split="test"

  model_name="nvidia/difix_ref"
  model_path="/data1/zliu/KITTI360_Completed/checkpoints/model_50001.pkl"
  ablation_study_name="vanilla_difix3d_old_stochastic_eval"
  seed=42
  timestep=199
  use_model_type="local"
  output_folder="/data4/zliu/Difix3D_Output_Results/Eval/$ablation_study_name"
  height=112
  width=544

  python src/diffix_model_evaluation.py \
    --input_test_filename "$input_test_filename" \
    --split "$split" \
    --model_name "$model_name" \
    --model_path "$model_path" \
    --ablation_study_name "$ablation_study_name" \
    --output_folder "$output_folder" \
    --seed "$seed" \
    --timestep "$timestep" \
    --lora_rank_vae 4 \
    --use_model_type "$use_model_type" \
    --eval_mode original_res \
    --height "$height" \
    --width "$width" \
    --stochastic_forward \
    --save_predictions
    # --mv_unet
    # --max_samples 100
}

RUN_EVAL_VANILLA_DIFIX3D_OLD_STOCHASTIC

