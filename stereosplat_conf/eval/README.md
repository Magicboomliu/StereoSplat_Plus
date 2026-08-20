# Evaluation

Evaluation logic lives in `eval/` (`run_multi_gpu.py` for shells, `run.py` for single-GPU / advanced use).  
All default flows use **15D confidence Gaussians** and the custom rasterizer (`rendered_conf`).

Setup and training → [`../README.md`](../README.md). Root repo guide → [`../../README.md`](../../README.md).

---

## Paper ↔ `eval_mode`

Paper: [StereoSplat+ (arXiv:2607.08808)](https://arxiv.org/abs/2607.08808)

| Paper | `eval_mode` | Shell |
|-------|-------------|-------|
| StereoSplat (2-view baseline) | `stereosplat` | `scripts/evaluation/evaluations/stereosplat.sh` |
| **StereoSplat+** (progressive + Difix3D + conf fusion) | **`pixel_fusion`** + `--conf_pixel_level_fusion` | **`scripts/evaluation/evaluations/stereosplat_plus.sh`** |
| S+ progressive, no conf fusion (ablation) | `stereosplat_plus` | CLI only |

`run_multi_gpu.py` defaults to `eval_mode=pixel_fusion`.  
The script name `stereosplat_plus.sh` is historical — it runs **`pixel_fusion`**, not `stereosplat_plus`.

---

## Quick start

```bash
cd stereosplat_conf
pixi install -e cu118 && pixi run -e cu118 setup

export STEREOSPLAT_CHECKPOINT=/path/to/stage2_checkpoint
export DIFIX3D_WEIGHTS=/path/to/model_130001.pkl

bash scripts/evaluation/evaluations/stereosplat_plus.sh   # paper StereoSplat+
bash scripts/evaluation/evaluations/stereosplat.sh          # 2-view baseline

# ablation val split
bash scripts/evaluation/ablations/stereosplat_plus.sh
bash scripts/evaluation/ablations/stereosplat.sh
```

Use **`bash`**, not `sh`. Override paths in the shell or via `STEREOSPLAT_CHECKPOINT` / `DIFIX3D_WEIGHTS`.

Bundled S+ flags: `--use_ref --conf_pixel_level_fusion --fusion_mode soft --self_pseudo`.

---

## Modes (short)

**`stereosplat`** — 2 GT views → one forward → render novel views.

**`pixel_fusion`** — 2-view GS render + pseudo-multiview GS render → per-pixel confidence fusion (paper StereoSplat+). Uses Difix3D on pseudo views when enabled.

**`stereosplat_plus`** — same progressive pipeline as `pixel_fusion`, but **no** 2D conf fusion (ablation only).

**`architecture`**: `whole` (single checkpoint, default) or `separated` (frozen Stage1 + Stage2; CLI only).

---

## Advanced CLI

```bash
pixi run -e cu118 accelerate launch \
  --config_file accelerate_configs/inference/multi_gpu.yaml \
  eval/run_multi_gpu.py \
  --eval_mode pixel_fusion \
  --config_path src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py \
  --output_folder outputs/eval/my_run \
  --val_filelist filenames/kitti360/train_complete/val.txt \
  --pretrained_model_path /path/to/checkpoint \
  --pretrained_diffix_model_path /path/to/model_130001.pkl \
  --use_ref --conf_pixel_level_fusion --fusion_mode soft --self_pseudo
```

```bash
pixi run -e cu118 python eval/run.py --help
pixi run -e cu118 python eval/run_multi_gpu.py --help
```

Call chain: `run*.py` → `routes.py` → `stereosplat.py` (`infer_*`).

---

## Output

Metrics are written to `--output_folder/metric.json` (RGB, depth, conf sections depend on mode).

Add `--output_vis` on **`eval/run.py`** (single-GPU) to save images under each bin folder; uses `--demo_filelist` instead of full val.

---

## Useful flags

| Flag | Notes |
|------|-------|
| `--eval_mode` | `stereosplat` / `stereosplat_plus` / `pixel_fusion` |
| `--architecture` | `whole` / `separated` |
| `--pseudo_ratio` | e.g. `0.5 1.0` (center + last stereo; default if omitted) |
| `--self_pseudo` | Match self-pseudo trained checkpoints |
| `--conf_pixel_level_fusion` | Required for paper StereoSplat+ |
| `--fusion_mode` | `soft` (shell default) / `legacy` / `per_view_adaptive` |
| `--no_difix3d` | Skip Difix3D |
| `--output_vis` | Visualize only (`run.py`) |
