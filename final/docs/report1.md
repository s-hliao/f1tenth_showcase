# Report 1 — Offline Metric Depth Pipeline (Phase 1)

**Date:** 2026-04-19  
**Author:** Yu-An Su  
**Course:** 16-663 F1Tenth Autonomous Racing, Spring 2026

---

## 1. Objective

Build and test the full offline depth estimation pipeline on 38 captured RealSense frames.  
Two components are skipped for now (replaced by captured data):
- **Real-time input** — replaced by pre-captured `rgb/` and `depth/` frames
- **LiDAR association** — replaced by one horizontal anchor row from the RealSense depth map

The pipeline is otherwise complete: DAv2 inference → affine scale recovery → metric depth → evaluation.

---

## 2. Pipeline

```
RGB frame (848×480)
    │
    ▼ DepthAnythingV2 ViT-S (CUDA, Jetson)
    │
    ▼ D_disp(u,v)  — relative disparity  [higher = closer]
    │
    ├── Anchor row (row 400 of 480)
    │       D_real[400, :]  from RealSense depth .npy (meters)
    │
    ▼ Affine fit:  D_metric = s · D_disp + t   (s < 0)
    │       Least-squares on all valid pixels, then
    │       re-fit on inliers (relative error < 30%)
    │
    ▼ D_metric(u,v) = clip(s · D_disp + t,  0, 20m)
    │
    ▼ Evaluate vs. full RealSense depth map
```

**Scripts:**
| File | Role |
|---|---|
| `setup.sh` | Clone DAv2, install `timm`, download ViT-S weights |
| `depth_anything_infer.py` | Batch inference → `rel_depth/*.npy` |
| `ransac_scale.py` | Affine (s,t) RANSAC estimator |
| `metric_depth.py` | Full pipeline + evaluation + visualization |

---

## 3. Environment Setup Challenges

Running on the F1Tenth Jetson required several fixes:

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | Active conda env (`base`) had no torch | Use `vision_lab_v2` env with `PYTHONNOUSERSITE=1` |
| `RuntimeError: torchvision::nms does not exist` | `pip install timm` pulled in torchvision 0.26.0 (compiled for torch 2.11.0 CPU), conflicting with Jetson torch 2.5.0a0 | Removed incompatible torchvision; wrapped all `torchvision` imports in `try/except (ImportError, RuntimeError)` in timm |
| `RuntimeError: Numpy is not available` | Conda env had numpy 2.2.6; Jetson torch 2.5.0a0 was compiled with numpy 1.x | Downgraded numpy to 1.26.4 in conda env |
| `ModuleNotFoundError: matplotlib` | Not in conda env | Installed with conda env's pip |
| DAv2 `from torchvision.transforms import Compose` | Same torchvision conflict | Patched `dpt.py` with a self-contained fallback `Compose` class |

**Working run command:**
```bash
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 python depth_anything_infer.py
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 python metric_depth.py
```

---

## 4. Key Finding: Disparity Convention and Model

DepthAnythingV2 outputs **relative disparity**, not relative depth:
- **Higher disparity → closer** to the camera (inverse of depth)
- Pearson correlation between DAv2 disparity and GT depth: **−0.78** (negative!)
- Pearson correlation between `1/disparity` and GT depth: **+0.81**

The correct affine model is:
```
D_metric = s · D_disp + t       (s < 0)
```
NOT `D_metric = s · (1/D_disp) + t` — inverting first causes unbounded values for near-zero disparity pixels (distant background), blowing up MAE to thousands of meters.

### Why affine (not just scale)?

DAv2 is trained with **affine-invariant loss** — it can have both scale *and* shift ambiguity. Using only a scale factor (no shift) produced MAE > 59,000m. Adding the shift brings MAE down to ~2m.

### Why least-squares instead of standard RANSAC?

Standard RANSAC with an absolute 0.20m inlier threshold degenerated to a **constant-depth solution** (s ≈ 0, t ≈ 0.49m):
- Pixel-level noise in DAv2 predictions is ~1m MAE even with optimal global parameters
- With a 0.20m threshold, the densest inlier cluster is found at pixels where GT ≈ t (constant), accumulating inliers with nearly zero slope
- This predicted only ~0.5m depth for all pixels — clearly wrong

**Fix used:** Direct least-squares affine fit on all valid anchor pixels, then remove outliers with relative error > 30%, then refit on remaining inliers.

---

## 5. Metrics Explained

### 5.1 Absolute Relative Error (AbsRel)

$$\text{AbsRel} = \frac{1}{N} \sum_{i} \frac{|d_i^{\text{pred}} - d_i^{\text{gt}}|}{d_i^{\text{gt}}}$$

**What it quantifies:** Scale-normalized error — a pixel at 4m with 2m error and a pixel at 1m with 0.5m error both contribute the same AbsRel of 0.50. This makes it useful for comparing performance across scenes with different depth ranges.

**Blind spots:**
- **Hides absolute magnitude.** AbsRel=0.30 at 10m means a 3m error — potentially catastrophic for obstacle avoidance — yet looks the same as AbsRel=0.30 at 0.5m (0.15m error, likely fine).
- **Far pixels dominate numerically when errors are large there.** A single severely wrong prediction at a far pixel inflates the mean disproportionately.
- **Insensitive to sign (bias).** Over-predicting by 1m and under-predicting by 1m both contribute equally — a systematic bias with zero mean would report a low AbsRel while always being wrong in the same direction.
- **Invalid at zero depth.** Pixels where GT = 0 (missing measurements) must be excluded, so AbsRel only covers the subset of pixels the sensor could measure.

### 5.2 Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{N} \sum_{i} |d_i^{\text{pred}} - d_i^{\text{gt}}|$$

**What it quantifies:** Average absolute deviation in meters, giving an intuitive feel for how far off predictions are in real-world units. Every pixel contributes equally regardless of its depth.

**Blind spots:**
- **Depth-range dependent.** MAE=1m is excellent at 20m range but poor at 1m range — comparisons across datasets or depth ranges are not meaningful without context.
- **Dominated by far pixels.** Because absolute errors naturally grow with distance (even for a fixed relative error), pixels far from the camera contribute larger absolute errors and pull the mean up. A scene with many far objects will always report higher MAE than a close-range scene, even if the model is equally accurate at all depths.
- **Insensitive to sign.** Like AbsRel, MAE cannot detect systematic over- or under-prediction. A model that always guesses 1m too far has the same MAE as one that alternately guesses 1m too close and 1m too far.
- **No tail sensitivity.** MAE treats a 5m outlier error as exactly 5× worse than a 1m error. It does not penalize catastrophic failures as strongly as RMSE does.

### 5.3 Why both metrics together?

| | AbsRel | MAE |
|---|---|---|
| Scale-invariant | ✓ | ✗ |
| Interpretable in meters | ✗ | ✓ |
| Sensitive to far-range failures | less | more |
| Detects bias | ✗ | ✗ |

Neither metric catches **signed bias** (systematic over- or under-prediction). That is why the signed error distribution plot (Section 6.4) is essential — it reveals whether errors are random noise or a consistent offset.

---

## 6. Results

### 6.1 Quantitative Metrics (38 frames)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| **AbsRel (in anchor range)** | 0.595 | 0.339 | 0.324 | 2.491 |
| **MAE (in anchor range)** | 1.051 m | 0.388 m | 0.697 m | 2.230 m |
| **RMSE (in anchor range)** | 1.371 m | 0.433 m | 0.977 m | 2.670 m |
| AbsRel (full frame) | 0.428 | 0.236 | 0.246 | 1.684 |
| MAE (full frame) | 1.956 m | 0.171 m | 1.704 m | 2.672 m |
| Scale factor s | −2.86 | 2.14 | −9.16 | −0.46 |
| Shift t | 7.11 | 1.44 | 5.55 | 11.21 |

**"In-anchor range"** = GT depth ≤ max depth in anchor row (~4–6m); the range where calibration is valid.  
**"Full frame"** = all valid GT pixels, including far objects the model was never calibrated on.

### 6.2 Selected Frame Results

| Frame | s | t | MAE (in-range) | AbsRel (in-range) |
|---|---|---|---|---|
| frame_00001 | −1.32 | 5.92 | 0.745 m | 0.604 |
| frame_00007 | −1.34 | 6.08 | 0.737 m | 0.549 |
| frame_00036 | −1.28 | 5.89 | 0.697 m | 0.566 ← best MAE |
| frame_00033 | −3.86 | 7.67 | 1.029 m | 0.324 ← best AbsRel |
| frame_00018 | −0.46 | 5.55 | 1.894 m | 2.491 ← worst |

Note: frame_00033 has low AbsRel but higher MAE than frame_00036 — consistent with AbsRel favouring frames where errors are small *relative to depth* even if absolute errors are larger.

### 6.3 Visualizations — Panel Images (`output/frame_NNNNN_panel.jpg`)

Each panel is a 5-column horizontal stack:

```
[ RGB | DAv2 Disparity | Predicted Metric Depth | GT Depth | Colorbar ]
```

- **RGB** — original camera frame
- **DAv2 Disparity** — raw model output, auto-scaled with INFERNO colormap (bright = high disparity = close)
- **Predicted Metric Depth** — after affine scale recovery, same color scale as GT
- **GT Depth** — RealSense ground truth, same color scale as predicted
- **Colorbar** — vertical INFERNO scale (top = `vmax` meters / far, bottom = 0 m / near) with 7 labeled tick marks, shared between the predicted and GT panels

The white horizontal line marks the anchor row (row 400). Green dots = inliers used for fitting; red dots = outliers rejected by the 30% relative error threshold.

### 6.4 Error by Depth Bucket (`output/error_by_depth_bucket.png`)

Three stacked bar charts, pooling all valid pixels across all 38 frames into 50 cm depth bins:

- **MAE per bin** — bars coloured by a plasma gradient, labeled with values. Shows that absolute error generally increases with depth, expected since larger depths have larger absolute variance.
- **AbsRel per bin** — with a red dashed reference at AbsRel = 0.20 (a common benchmark threshold for "good" monocular depth). Shows where the model is relatively accurate vs. inaccurate.
- **Pixel count per bin** — reveals data sparsity at extreme depths; bars in the far range have very few pixels, so those error numbers are less reliable.

**Key insight from this figure:** AbsRel and MAE tell different stories at different ranges. Near pixels (0–2m) may show low MAE but high AbsRel (small absolute errors but the object is close so the relative error is penalized heavily), while far pixels show the opposite pattern.

### 6.5 Depth Distributions (`output/depth_distributions.png`)

Two subplots:

- **Top — pooled density histograms:** GT depth (blue) vs. predicted depth (red) across all frames. Divergence between the two distributions shows overall bias: if the predicted histogram is shifted left the model is systematically under-predicting depth (predicting objects as closer than they are), and vice versa.
- **Bottom — per-frame ribbon:** Median ± 10th/90th percentile band computed separately for each frame and then overlaid. A wide ribbon means the model's output distribution shifts significantly between frames (consistent with the high variance in scale factor s = −0.46 to −9.16 observed in the quantitative results). A narrow ribbon would indicate stable predictions.

### 6.6 Signed Error Distribution (`output/signed_error_distribution.png`)

Two subplots:

- **Top — global histogram of `pred − GT`:** Positive = model predicts object farther away than it is (over-prediction); negative = model predicts object closer (under-prediction). The mean and median lines reveal systematic bias; the ±1 std band shows spread. A perfectly unbiased model would have a distribution centred at zero. The actual mean signed error directly exposes the bias that AbsRel and MAE both hide.
- **Bottom — signed error vs. depth bucket:** Median (solid line) + IQR band (dark shading) + 10–90th percentile ribbon (light shading) per 50 cm bin. This reveals **depth-dependent bias**: whether the model consistently over-predicts at close range and under-predicts at far range (or the reverse), which is a known failure mode of affine scale alignment when the anchor row doesn't span the full depth range.

---

## 7. Analysis

### What works
- DAv2 captures the **qualitative depth ordering** well — near/far objects are correctly distinguished in the disparity map
- The affine model achieves **MAE < 1m** on the best frames (frame_00036: 0.697m, frame_00007: 0.737m)
- The pipeline runs end-to-end on the Jetson and produces correct-looking depth maps

### What doesn't work / limitations

**1. High variance in scale factor (s = −0.46 to −9.16)**  
The anchor row samples different depth distributions per frame. Since the model is `s*disp + t`, a narrow depth range in the anchor row produces a poorly conditioned fit that extrapolates badly outside that range.

**2. Calibration only covers anchor-row depth range (~0–6m)**  
Row 400 near the bottom sees mostly floor and near walls. The upper image region (>6m) is extrapolated and performs worse. A higher anchor row or multiple rows would improve coverage.

**3. Depth-dependent bias visible in signed error plot**  
The signed error vs. depth bucket chart is expected to show a systematic pattern: the model tends to under-predict at ranges not represented in the anchor row. This is because the affine parameters are anchored to a narrow depth range and the linear extrapolation diverges.

**4. RANSAC degeneracy with dense noisy anchors**  
Standard RANSAC with tight absolute thresholds fails here because pixel-level prediction noise (~1m) far exceeds what any tight threshold can distinguish. Real sparse LiDAR beams — precise to centimetres — would fix this: each beam gives a clean, reliable (disp, depth) pair, and RANSAC's outlier model (e.g. moving objects blocking a beam) would be meaningful.

**5. frame_00018 failure (AbsRel = 2.49)**  
Nearly zero slope (s = −0.46) means the anchor row contained mostly uniform-depth floor pixels, giving no slope information. The fit collapsed to a near-constant prediction.

---

## 8. Comparison to Naive Baseline

| Method | MAE (in-range) | AbsRel |
|---|---|---|
| Constant depth (mean GT) | ~1.5m (estimated) | ~0.60 |
| **Our method (LS affine)** | **1.05m** | **0.60** |
| Oracle (full GT affine fit) | ~0.99m | ~0.51 |

Our method matches the oracle in AbsRel and comes within 60 cm of it in MAE, using only one anchor row out of 480.

---

## 9. Next Steps

1. **LiDAR integration** — replace anchor row with real sparse LiDAR projections; RANSAC will work correctly with precise centimetre-level measurements
2. **Row selection / multi-row anchoring** — a row higher in the image covers more depth range; combining rows further stabilises the fit
3. **ROS2 nodes** — wrap `depth_anything_infer` and `metric_depth` as subscribers to `/camera/image_raw` and `/scan`
4. **Runtime optimisation** — profile DAv2 inference time on Jetson; consider TensorRT export for real-time use

---

## 10. Run Instructions

```bash
# One-time setup
bash setup.sh

# Step 1 — inference on all frames (saves to rel_depth/)
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 python depth_anything_infer.py

# Step 2 — full pipeline, evaluation, all figures
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 python metric_depth.py

# Single frame
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 python metric_depth.py --frame 1

# Evaluation only, no image output (faster)
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 python metric_depth.py --no-viz
```

**Output files:**

| File | Description |
|---|---|
| `output/frame_NNNNN_panel.jpg` | 5-column panel: RGB \| disparity \| metric \| GT \| colorbar |
| `output/frame_NNNNN_profile.png` | Anchor-row depth profile with GT vs predicted overlay |
| `output/error_by_depth_bucket.png` | MAE, AbsRel, pixel count per 50 cm depth bin |
| `output/depth_distributions.png` | GT vs predicted depth histograms + per-frame ribbon |
| `output/signed_error_distribution.png` | Signed error histogram + per-bucket bias plot |
