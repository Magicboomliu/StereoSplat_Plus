# Difix3D (Pixi project)

This folder is a standalone **pixi project** for training and evaluating Difix3D (image restoration).

## Installation (Pixi)

Create/sync the environment from this folder:

```bash
cd difix3d
pixi install
```

Notes:
- This pixi project targets **Python 3.11** (`requires-python >= 3.11`).
- It pins the core HF stack for Difix3D:
  - `diffusers==0.25.1`
  - `transformers==4.38.0`
  - `huggingface-hub==0.25.1`
  - `peft==0.9.0`

## Training

Entrypoint script:
- `scripts/train_difix3d.sh`

The script launches training via `accelerate`:
- training code: `trainer/train_difix_ref.py`
- config: `configs/train_difix_ref.yaml`
- accelerate config examples: `gpu_configs/single_mode/gpu_config_*.yaml`

Run:

```bash
cd difix3d
bash scripts/train_difix3d.sh
```

Before running, edit `scripts/train_difix3d.sh` to set:
- dataset JSON path (e.g. `filenames/Validation_Set/all_results_dict.json`)
- pretrained checkpoint path (optional)
- output directory
- GPU config (`gpu_configs/...`)

## Inference / Evaluation

Entrypoint script:
- `scripts/eval_difix3d.sh`

This script evaluates a finetuned checkpoint using:
- `evals/eval_difix_ref_pipeline.py`

Run:

```bash
cd difix3d
bash scripts/eval_difix3d.sh
```

Edit `scripts/eval_difix3d.sh` to set:
- `dataset_path`
- `pretrained_path` (e.g. `model_*.pkl`)
- output JSON path
