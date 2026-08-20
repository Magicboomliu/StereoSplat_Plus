#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

Difix3D_Vanilla_Finetuning() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFIX_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DIFIX_ROOT" || exit 1

pretrained_model_name_or_path="nvidia/difix_ref"
pretrained_model_path="${DIFIX3D_WEIGHTS:-/path/to/model_130001.pkl}"
output_dir="${DIFIX_OUTPUT_DIR:-${DIFIX_ROOT}/outputs/difix_finetune}"
config_path="configs/train_difix_ref.yaml"
dataset_path="${DIFIX_DATASET_JSON:-${DIFIX_ROOT}/filenames/Validation_Set/all_results_dict.json}"

max_train_steps=120000
resolution_h=112
resolution_w=544
learning_rate=2e-5
train_batch_size=4
dataloader_num_workers=8
checkpointing_steps=10000
print_freq=10
eval_freq=5000
viz_freq=100
num_samples_eval=100
lambda_l2=1.0
lambda_ssim=0.5
report_to="wandb"
tracker_project_name="DIFIX3D_Finetuning"
tracker_run_name="vanilla_difix3d_psnr20"
timestep=199

if [ ! -f "$dataset_path" ]; then
  echo "[ERROR] dataset JSON not found: $dataset_path"
  echo "        Generate your dataset manifest or copy filenames/Validation_Set/all_results_dict.example.json"
  echo "        Set DIFIX_DATASET_JSON to your JSON file."
  exit 1
fi

pixi run accelerate launch --mixed_precision=bf16 --config-file gpu_configs/single_mode/gpu_config_0.yaml \
  trainer/train_difix_ref.py \
  --config "$config_path" \
  --pretrained_model_name_or_path "$pretrained_model_name_or_path" \
  --pretrained_model_path "$pretrained_model_path" \
  --output_dir "$output_dir" \
  --dataset_path "$dataset_path" \
  --max_train_steps "$max_train_steps" \
  --resolution_h "$resolution_h" \
  --resolution_w "$resolution_w" \
  --learning_rate "$learning_rate" \
  --print_freq "$print_freq" \
  --train_batch_size "$train_batch_size" \
  --dataloader_num_workers "$dataloader_num_workers" \
  --checkpointing_steps "$checkpointing_steps" \
  --eval_freq "$eval_freq" \
  --viz_freq "$viz_freq" \
  --num_samples_eval "$num_samples_eval" \
  --lambda_l2 "$lambda_l2" \
  --lambda_ssim "$lambda_ssim" \
  --report_to "$report_to" \
  --tracker_project_name "$tracker_project_name" \
  --tracker_run_name "$tracker_run_name" \
  --timestep "$timestep" \
  --enable_xformers_memory_efficient_attention

}

Difix3D_Vanilla_Finetuning
