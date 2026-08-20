#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

EVAL_Finetuned_Difix3D() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFIX_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DIFIX_ROOT" || exit 1

dataset_path="${DIFIX_DATASET_JSON:-${DIFIX_ROOT}/filenames/Validation_Set/all_results_dict.json}"
pretrained_path="${DIFIX3D_WEIGHTS:-/path/to/model_130001.pkl}"
saved_json_path="${DIFIX_EVAL_JSON:-${DIFIX_ROOT}/output/finetuned_vanilla_difix3d_psnr20_eval.json}"

if [ ! -f "$dataset_path" ]; then
  echo "[ERROR] dataset JSON not found: $dataset_path"
  echo "        Generate your dataset manifest or copy filenames/Validation_Set/all_results_dict.example.json"
  echo "        Set DIFIX_DATASET_JSON to your JSON file."
  exit 1
fi
if [ ! -f "$pretrained_path" ]; then
  echo "[ERROR] Difix3D checkpoint not found: $pretrained_path"
  exit 1
fi

pixi run python evals/eval_difix_ref_pipeline.py \
  --dataset_path "$dataset_path" \
  --pretrained_path "$pretrained_path" \
  --saved_json_path "$saved_json_path"

}

EVAL_Finetuned_Difix3D
