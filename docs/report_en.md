# StereoSplat+ Research Progress Report

**Project**: StereoSplat+: Feed-Forward Gaussian Splatting with Diffusion-Assisted Progressive Inference  
**Author**: Liu Zihua  
**Date**: 2026-06-08  
**Evaluation Dataset**: KITTI-360 val set (5485 bins)

---

## 1. Background and Problem Definition

### 1.1 StereoSplat Baseline

StereoSplat is a feed-forward 3D Gaussian Splatting method for autonomous driving (KITTI-360). It takes a single stereo pair (first-frame left+right) as input and directly predicts 3D Gaussians covering the entire trajectory via a single forward pass.

### 1.2 StereoSplat+: Progressive Inference

StereoSplat+ extends the baseline through **progressive inference**:
1. Step 1: Generate initial 3DGS (G_base) from first-frame stereo pair
2. Step 2: Render pseudo stereo pairs at center/last trajectory positions from G_base
3. Step 3: Inject pseudo pairs back into the model to produce G_plus

### 1.3 Core Problem: Performance Degradation from Pseudo View Injection

| Method | PSNR | SSIM | Notes |
|--------|------|------|-------|
| StereoSplat (baseline) | 20.42 | 0.67 | 2-view GT only |
| StereoSplat+ (with GT leakage bug) | 21.03 | 0.73 | Center GT leaked |
| StereoSplat+ (fixed, with Difix3D) | 19.89 | 0.66 | Degradation: -0.53 |
| StereoSplat+ (fixed, without Difix3D) | 19.78 | 0.65 | Degradation: -0.64 |

**Conclusion**: After fixing the GT leakage bug, pseudo view injection **hurts** performance. Root cause: distribution shift between GT inputs (seen during training) and pseudo inputs (rendered, lower quality).

---

## 2. Three Improvement Paths

| Path | Strategy | Goal |
|------|----------|------|
| **Path 1** | Train stronger image enhancer (Difix3D) | Reduce pseudo/GT distribution gap |
| **Path 2** | Two-stage pseudo-GT mixed training | Make model robust to pseudo inputs |
| **Path 3** | Design effective G_base/G_plus fusion strategy | Robust inference |

**This report focuses on Path 3**: confidence-based fusion strategies.

---

## 3. Technical Implementation: Per-Gaussian Confidence Rendering

### 3.1 System Components

To support confidence-based fusion, the following modifications were made to the 3DGS pipeline:

- **15D Gaussians**: Added 1D confidence to the original 14D (xyz, rotation, scale, opacity, SH)
- **Custom Rasterizer**: Modified diff-gaussian-rasterization CUDA kernel for `rendered_conf` output
- **Confidence Supervision**: Self-supervised conf loss during training
- **Fusion Modules**: Pixel-level fusion, GS voxel fusion, joint fusion, Oracle upper bound

### 3.2 Gaussian Prediction Head Architecture

The model outputs 15D Gaussian parameters:

| Parameter | Dims | Activation | Meaning |
|-----------|------|------------|---------|
| xyz offset | 0:3 | exp (delta) | 3D position offset |
| opacity | 3:4 | sigmoid | Opacity |
| scale | 4:7 | exp(x) * 0.01 | 3D scale |
| rotation | 7:11 | normalize | Quaternion rotation |
| RGB | 11:14 | sigmoid | Color |
| **confidence** | **14:15** | **sigmoid** | **Quality confidence ∈ (0,1)** |

Network architecture (`custom_gs_head.py`):

```
Input: concat(image[3], depth[1], match_prob[1], upsampled_features[C])
  → gaussian_regressor: Conv2d(in, 64) → GELU → Conv2d(64, 64)
  → concat(regressor_out[64], image[3], features[C], match_prob[1])
  → gaussian_aggregator: Conv2d(in, 128) → GELU → Conv2d(128, 128)
  → gaussian_head: Conv2d(128, 15) → GELU → Conv2d(15, 15)
  → Apply respective activations per parameter group
```

Key point: **confidence shares the same prediction head** as other Gaussian parameters; the last dimension is mapped to (0,1) via sigmoid.

### 3.3 Confidence Rendering

Confidence is rendered identically to RGB color via **alpha-blending** (volume rendering):

```
rendered_conf(pixel) = Σ_i conf_i × α_i × T_i
```

Where:
- `conf_i`: per-Gaussian confidence value of the i-th Gaussian
- `α_i`: opacity of the i-th Gaussian at this pixel (after 2D Gaussian falloff)
- `T_i = Π_{j<i} (1 - α_j)`: accumulated transmittance (occlusion from preceding Gaussians)

CUDA implementation (`forward.cu`):
```cuda
float CONF = 0;
// Front-to-back accumulation for each Gaussian:
CONF += geom_confs[collected_id[j]] * alpha * T;
```

Output: `rendered_conf [B, V, 1, H, W]`, same resolution as rendered_image.

### 3.4 Confidence Supervision

#### Supervision Signal: Self-supervised Photometric Soft Label

```python
# stereosplat.py, lines 1358-1371
with torch.no_grad():
    l1_err = torch.abs(rendered_image.detach() - gt_image)  # [B,V,3,H,W]
    l1_err = l1_err.mean(dim=2, keepdim=True)               # [B,V,1,H,W] channel mean
    conf_gt = torch.exp(-conf_lambda * l1_err)              # (0, 1]
```

**Semantics**:
- Where render matches GT perfectly → `l1_err ≈ 0` → `conf_gt ≈ 1` (high confidence)
- Where render is poor → large `l1_err` → `conf_gt → 0` (low confidence)
- Gradient detach (`torch.no_grad()`): conf_gt is a pseudo-label, not backpropagated through

#### Loss Function

```python
conf_loss = MSE(rendered_conf, conf_gt)
fusion_branch_loss += weight_conf * conf_loss
```

#### Hyperparameters

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `use_conf_loss` | `True` | Enable conf supervision |
| `conf_lambda` | `10.0` | Exponential decay sharpness: higher = conf drops faster with error |
| `weight_conf` | `0.1` | Conf loss weight within fusion branch |
| `fusion_branch_weight` | `1.0` | Fusion branch weight in total loss |

**Loss hierarchy**:
```
total_loss = cv_branch_loss × 1.0
           + volume_branch_loss × 1.0
           + fusion_branch_loss × 1.0        ← conf_loss lives here
           + depth_est_loss × 0.05

where fusion_branch_loss = recon_loss × 1.0
                         + perceptual_loss × 0.05
                         + depth_abs_loss × 0.01
                         + conf_loss × 0.1       ← effective weight = 0.1
```

---

## 4. Experimental Design

### 4.1 Model Configurations

| Model | Training | G_base Source | G_plus Source | Notes |
|-------|----------|--------------|--------------|-------|
| **Stage1 (S1)** | All GT views + conf supervision | S1 processes 2-view GT | S1 processes pseudo views | Single model, dual use |
| **Stage2 whole** | Pseudo-GT mixed training | S2 processes 2-view GT | S2 processes pseudo views | Single model, dual use |
| **Stage2 separated (S2 sep)** | Frozen S1 + dedicated S2 | **S1 processes 2-view GT** | **S2 processes pseudo views** | Each model handles its own distribution |

Stage2 training: freeze S1 to produce pseudo views, then train S2 on pseudo+GT mixed inputs. Therefore **S2 sep's inference exactly mirrors its training setup** (S1 produces pseudo, S2 processes them)—this is the fundamental reason it works best.

### 4.2 Evaluation Metrics

- **Primary**: `all_view_psnr_average` (average of first + center + last stereo PSNR)
- Secondary: SSIM, LPIPS, Abs_Rel (depth)
- Per-view analysis: first_view / center_view / last_view reported separately

### 4.3 Fusion Methods

| Method | Logic | Granularity |
|--------|-------|-------------|
| **Progressive only** | Use G_plus rendering directly (no base/plus selection) | — |
| **Pixel-level fusion** | Per-pixel selection by rendered conf | Pixel-level |
| **Pixel fusion + margin** | `conf_plus > conf_base + margin` to select plus (tie → base) | Pixel-level |
| **GS voxel fusion** | Within 3D voxels, select base or plus Gaussians by mean conf | Voxel-level (0.1m) |
| **GS + Pixel** | GS fusion first, then pixel fusion | Joint |
| **Oracle** | Per-pixel selection using GT L1 error (theoretical upper bound) | Pixel-level |

---

## 5. Experimental Results

### 5.1 Impact of Confidence Training on Baseline Performance

**Motivation**: Verify that adding the confidence head does not hurt original performance.

| Model | PSNR | SSIM | LPIPS | Abs_Rel |
|-------|------|------|-------|---------|
| S1 (without conf, 14D) | 20.424 | 0.674 | 0.195 | 0.071 |
| S1 (with conf, 15D) | 20.436 | 0.668 | 0.199 | 0.075 |
| S2 (with conf, 15D) | 20.425 | 0.673 | 0.205 | 0.076 |

**Conclusion**: The confidence head has negligible impact on PSNR (+0.01). The 15D model is a valid extension of the 14D baseline.

---

### 5.2 Oracle Upper Bound Analysis

**Motivation**: Determine the theoretical best performance achievable by confidence-based fusion.

**Design**: Per-pixel comparison of G_base and G_plus renders against GT, selecting whichever is closer.

| Component | all_view PSNR | all_view SSIM |
|-----------|---------------|---------------|
| G_base (2-view only) | 20.408 | 0.665 |
| G_plus (pseudo multiview) | 19.689 | 0.637 |
| **Oracle Fusion (GT selection)** | **21.096** | **0.706** |

**Per-view Oracle analysis**:

| View | G_base | G_plus | Oracle | Oracle Gain |
|------|--------|--------|--------|-------------|
| First | 25.24 | 23.61 | 26.24 | +1.00 |
| Center | 19.73 | 19.30 | 20.22 | +0.49 |
| Last | 17.59 | 17.27 | 18.11 | +0.52 |
| **All** | **20.41** | **19.69** | **21.10** | **+0.66** |

**Key conclusions**:
1. G_plus is worse overall (20.41 vs 19.69), but **locally** G_plus has better pixels
2. Oracle achieves 21.10, which is **+0.66 PSNR above baseline**—proving significant fusion potential
3. First frame has the largest gain (+1.00): even where G_base excels (25.24), G_plus contributes some better pixels

---

### 5.3 Progressive Performance Across Three Model Configurations

**Motivation**: Compare S1, S2 whole, and S2 sep when using G_plus directly (no fusion).

| View | S1 (prog+difix) | S2 whole (prog) | S2 sep (prog) | Baseline |
|------|-----------------|-----------------|---------------|----------|
| First | 23.61 | 24.28 | 24.65 | 25.24 |
| Center | 19.30 | 19.47 | 19.79 | 19.73 |
| Last | 17.27 | 17.53 | **17.79** | 17.59 |
| **All** | **19.76** | **20.07** | **20.37** | **20.44** |

**Conclusions**:
1. S2 sep progressive (20.37) far exceeds S1 (19.76)—pseudo-GT mixed training is effective
2. S2 sep center/last views approach or exceed baseline—pseudo views genuinely help at distant viewpoints
3. All configurations show first frame degradation—pseudo input quality most harms the first frame

---

### 5.4 Pixel Fusion Performance Across Three Model Configurations

**Motivation**: Compare confidence-based pixel fusion effectiveness across model configurations.

| View | S1 pixel fusion | S2 whole pixel fusion | S2 sep pixel fusion | Baseline |
|------|-----------------|----------------------|---------------------|----------|
| First | 24.07 | 24.80 | **25.36** | 25.24 |
| Center | 19.38 | 19.67 | **19.84** | 19.73 |
| Last | 17.31 | 17.63 | 17.72 | 17.59 |
| **All** | **19.93** | **20.31** | **20.53** | **20.44** |

**Comparison against baseline**:

| Configuration | Progressive vs BL | Pixel Fusion vs BL | Fusion Effective? |
|---------------|-------------------|--------------------|--------------------|
| S1 | -0.68 | -0.51 | No (still below baseline) |
| S2 whole | -0.37 | -0.13 | Partially (close but below) |
| **S2 sep** | -0.07 | **+0.10** | **Yes (only config above baseline)** |

**Conclusion**: Only S2 separated + pixel fusion exceeds the baseline, with improvements across all three view groups.

---

### 5.5 Core Finding: Confidence Bias

**Motivation**: Understand why S1 fusion fails while S2 sep succeeds.

#### Confidence Statistics Comparison

| Configuration | G_base conf (all) | G_plus conf (all) | Bias (plus - base) |
|---------------|--------------------|--------------------|---------------------|
| **S1** | 0.660 | **0.744** | **+0.084** |
| **S2 whole** | 0.663 | 0.668 | +0.005 |
| **S2 sep** | 0.660 | 0.668 | +0.008 |

#### Per-View Confidence Comparison

| Config | G_base: first / center / last | G_plus: first / center / last |
|--------|-------------------------------|-------------------------------|
| S1 | 0.743 / 0.649 / 0.597 | 0.752 / **0.758** / **0.749** |
| S2 whole | 0.749 / 0.650 / 0.597 | 0.742 / 0.657 / 0.614 |
| S2 sep | 0.743 / 0.649 / 0.597 | 0.743 / 0.657 / 0.614 |

#### Root Cause Analysis

**Why S1 has large conf bias (+0.084)**:

S1 was trained only on GT inputs. During inference, G_plus receives pseudo inputs (GT + pseudo mix), producing more Gaussians covering the same regions. After alpha-blending, rendered confidence is naturally higher—the model misinterprets "more multiview coverage" as "higher quality."

**Why S2 sep has minimal conf bias (+0.008)**:

S2 was trained with the separated architecture—frozen S1 produces pseudo views, S2 receives them. Therefore **S2 has already seen the pseudo view distribution during training**. Its confidence predictions for pseudo inputs are naturally calibrated. There is no distribution shift at inference time.

**Why S2 whole is worse than S2 sep**:

S2 whole uses the same model for both GT (producing G_base) and pseudo (producing G_plus). But S2 was trained on pseudo-GT mix—pure GT input is not its optimal operating point. S2 sep assigns each model to its native distribution: S1 handles GT → G_base, S2 handles pseudo → G_plus.

---

### 5.6 S1 Pixel Fusion: Per-View Loss Analysis

**Motivation**: Quantify exactly where and why S1 pixel fusion fails.

| View | Baseline | S1 Pixel Fusion | Loss | Oracle | Oracle Gain | Cause |
|------|----------|-----------------|------|--------|-------------|-------|
| First | 25.24 | 24.07 | **-1.17** | 26.24 | +1.00 | G_plus conf 0.752 > G_base 0.743 → massive incorrect pixel replacement |
| Center | 19.73 | 19.38 | -0.35 | 20.22 | +0.49 | G_plus conf 0.758 >> G_base 0.649 → largest bias view |
| Last | 17.59 | 17.31 | -0.28 | 18.11 | +0.52 | G_plus conf 0.749 >> G_base 0.597 → same as above |
| **All** | **20.44** | **19.93** | **-0.51** | **21.10** | **+0.66** | |

**Core issue**: S1's G_plus confidence is systematically higher than G_base across **all views**, causing massive incorrect pixel replacement. First frame suffers the most (-1.17) as it's the primary source of the total all-view loss (-0.51).

---

### 5.7 S2 Separated Pixel Fusion: Detailed Analysis

**Motivation**: Understand why S2 sep works and where room for improvement remains.

| View | Baseline | S2 sep Progressive | S2 sep Pixel Fusion | Fusion vs Prog | Fusion vs BL |
|------|----------|--------------------|--------------------|----------------|--------------|
| First | 25.24 | 24.65 | **25.36** | +0.71 | **+0.12** |
| Center | 19.73 | 19.79 | **19.84** | +0.05 | **+0.11** |
| Last | 17.59 | **17.79** | 17.72 | -0.07 | **+0.13** |
| **All** | **20.44** | **20.37** | **20.53** | **+0.16** | **+0.10** |

**Key observations**:

1. **First frame: fusion massively improves progressive (+0.71)**. Confidence correctly selects G_base at the first frame (G_base far better: 25.24 vs 24.65)
2. **Center frame: fusion slightly improves progressive (+0.05)**. Confidence partially discriminates base vs plus
3. **Last frame: fusion is slightly worse than progressive (-0.07)**. Some regions where G_plus is better (progressive 17.79 > baseline 17.59) are incorrectly replaced back to base by fusion

**Remaining headroom**: Compared to S1 Oracle (21.10), there is still 0.57 PSNR of potential improvement (S2 sep oracle needs to be computed to confirm its own ceiling).

---

### 5.8 S1 Fusion Strategy Ablation

**Motivation**: Compare different fusion granularities and methods under S1.

| Method | PSNR | SSIM | vs Baseline |
|--------|------|------|-------------|
| Baseline (2-view) | 20.436 | 0.668 | — |
| Progressive + Difix3D | 19.762 | 0.656 | -0.67 |
| Pixel fusion (no margin) | 19.925 | 0.660 | -0.51 |
| Pixel fusion + margin 0.05 | 20.007 | 0.662 | -0.43 |
| GS voxel fusion | 19.842 | 0.656 | -0.59 |
| GS + Pixel joint | 20.071 | 0.661 | -0.36 |
| **Oracle** | **21.096** | **0.706** | **+0.66** |

**Conclusions**:
1. All S1 fusion methods are below baseline—root cause is systematic conf bias
2. Margin helps incrementally (+0.08 over no-margin); GS+Pixel is better than GS-only (-0.36 vs -0.59)
3. GS voxel fusion is worst—coarser granularity amplifies conf bias impact
4. Oracle shows +0.66 headroom exists; the problem is purely **selection accuracy**

---

### 5.9 Effect of Margin Strategy

| Config | Margin | PSNR | vs no-margin |
|--------|--------|------|--------------|
| S1 pixel fusion | 0 | 19.925 | — |
| S1 pixel fusion | 0.05 | 20.007 | +0.08 |

**Principle**: `conf_plus > conf_base + 0.05` required to select plus; ties favor base. Since G_base has better average quality, defaulting to base when uncertain is a sound conservative strategy.

---

## 6. Core Conclusions

### 6.1 Method Effectiveness

| Conclusion | Evidence |
|------------|----------|
| Oracle proves conf fusion has +0.66 PSNR potential | Oracle = 21.10 vs Baseline = 20.44 |
| S2 sep + pixel fusion is the only method exceeding baseline | 20.53 vs 20.44 (+0.10) |
| Conf bias is the root cause of S1 fusion failure | S1 bias +0.084 vs S2 sep bias +0.008 |
| Train-inference distribution alignment is the key | S2 sep trained in separated mode → conf naturally calibrated |

### 6.2 Model Configuration Comparison

| Configuration | Strengths | Weaknesses | Recommendation |
|---------------|-----------|------------|----------------|
| S1 whole | Simple; highest Oracle ceiling | Large conf bias → fusion fails | Needs bias correction |
| S2 whole | Small conf bias | G_base quality lower (S2 suboptimal for pure GT) | Not recommended |
| **S2 sep** | **Small bias + high G_base quality (from S1)** | Requires two models | **Recommended final approach** |

### 6.3 Confidence Supervision Analysis

| Aspect | Analysis |
|--------|----------|
| Supervision method | Self-supervised `exp(-10 × L1_error)`: reasonable but imperfect |
| Issue 1 | Model conflates "more multiview coverage" with "higher quality" → G_plus conf inflated (S1) |
| Issue 2 | S2 naturally resolves this by training on pseudo distribution |
| Issue 3 | `conf_lambda=10` provides limited discrimination in common error ranges |
| Issue 4 | `weight_conf=0.1` is relatively low, limiting conf learning signal strength |

---

## 7. Progress Summary

### 7.1 Completed Work

| Item | Status |
|------|--------|
| 15D Gaussian + conf custom Rasterizer | Done |
| Stage1 conf model training | Done |
| Stage2 pseudo-GT mixed training (whole + separated) | Done |
| Pixel-level conf fusion implementation & evaluation | Done |
| GS voxel conf fusion implementation & evaluation | Done |
| GS + Pixel joint fusion | Done |
| Oracle upper bound analysis (S1) | Done |
| Full ablation study (13 experiments, 3 model configs) | Done |
| Unified evaluation system (eval/run.py) | Done |
| Root cause analysis: conf bias & train-test distribution alignment | Done |

### 7.2 Current Best Result

```
Method: Stage2 Separated + Pixel-level Confidence Fusion
PSNR: 20.53 (vs baseline 20.44, +0.10)
SSIM: 0.670
LPIPS: 0.194
```

---

## 8. Future Plan (2 Weeks)

### 8.1 Core Strategy

Based on the analysis, apply differentiated optimization strategies for S1 and S2 sep:

| Track | Bottleneck | Strategy | Target |
|-------|-----------|----------|--------|
| S1 whole | Conf bias +0.084 → massive pixel mis-selection | Per-view margin + conf calibration | 19.93 → ~20.5-20.7 |
| S2 sep | Small bias +0.008 → selection precision limited | Soft blending + fine-grained margin | 20.53 → ~20.6-20.7 |

### 8.2 Specific Approaches

**Approach A: Per-View Adaptive Fusion (primarily for S1)**
- Force base selection for first frame (S1 first frame severely damaged: 25.24→24.07)
- Use conf fusion with calibrated conf for center/last frames
- Expected: S1 all_view from 19.93 up to ~20.5

**Approach B: Conf Calibration (primarily for S1)**
- Per-image z-score normalization: remove global conf bias between G_base/G_plus
- Make comparison based on relative conf ranking rather than absolute values

**Approach C: Soft Blending (primarily for S2 sep)**
- `weight_plus = sigmoid((conf_plus - conf_base) × temperature)` for continuous weighting
- Reduces cost of hard selection errors when conf values are close
- Expected: S2 sep from 20.53 up to ~20.6

### 8.3 Items to Confirm

- S2 separated Oracle upper bound (confirm its ceiling)
- Per-view fusion V-dimension index layout (confirm first/center/last positions in code)

---

*End of report*
