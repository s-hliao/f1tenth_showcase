# Team 10 — Metric Depth Estimation: Todo List

Owner: Yu-An Su (RGB inference + RealSense anchor-row scale recovery + RANSAC)

Pipeline recap:
1. RGB frame → DepthAnythingV2 → `D_disp(u,v)` (relative disparity, higher = closer)
2. Auto-select anchor row: row with the most GT depth spread within `max_gt_depth`
3. Affine fit: `D_metric = s·D_disp + t`  (s < 0), RANSAC-refined on inliers
4. Metric depth: `D_metric(u,v) = clip(s·D_disp + t, 0, max_depth)`

---

## Phase 0 — Environment & Setup

- [x] `bash setup/setup.sh` — clone DAv2, install timm, download ViT-S weights
- [x] Confirmed: torch, CUDA, numpy, opencv available on Jetson; depth frames are float32 metres (480×848), ~20% invalid zeros
- [ ] Confirm camera intrinsics K (needed for LiDAR-to-pixel projection)
- [ ] Confirm LiDAR-to-camera extrinsic T_lc (needed for ROS2 LiDAR integration)

---

## Phase 1 — Offline Validation

- [x] `depth_anything_infer.py` — load DAv2 ViT-S, run on `rgb/` frames, save disparity to `rel_depth/`
- [x] `ransac_scale.py` — RANSAC affine estimator (`D_metric = s·D_disp + t`), self-test passing
- [x] `metric_depth.py` — full offline pipeline with CLI flags:
  - `--data-dir DIR` — point at any folder with `rgb/`, `depth/`, `rel_depth/`
  - `--max-gt-depth M` — cap RealSense GT at M m; output to `output_<M>m/`
  - `--anchor-row R` — fix anchor row; omit for auto-select (recommended)
  - `--frame N` — run on single frame
- [x] Pipeline run on `testing_data/` (38 frames); results in `output_3m/`
- [x] `docs/ransac_illustration.py` — figure comparing RANSAC vs naïve LS with injected outliers; saved to `docs/ransac_vs_no_ransac_3m.png`
- [ ] Review output panels in `output_3m/` — check colorised depth maps and anchor row profile plots

---

## Phase 2 — ROS2 Node (depth_fusion)

- [x] `depth_fusion/depth_fusion/depth_fusion_node.py` — real-time node:
  - pyrealsense2 capture + DAv2 inference + anchor-row affine fit + dual OpenCV viz
  - ROS2 params: `anchor_row` (-1 = auto), `max_gt_depth` (3.0 m), `max_depth`, `inlier_rel_tol`, `cam_*`
- [x] `run_depth_fusion.sh` — launches node inside `conda vision_lab_v2` with ROS2 env; DISPLAY fix applied
- [x] Node confirmed running on Jetson (~1 fps, CUDA, anchor row auto-selected per frame)
- [ ] Build / colcon install after node edits: `colcon build --packages-select depth_fusion`
- [ ] `depth_fusion.launch.py` — wire parameters into launch file
- [ ] LiDAR integration: `lidar_assoc_node.py` — project `/scan` beams to image plane using K + T_lc
- [ ] Scale fusion: replace anchor-row GT method with LiDAR-derived scale samples

---

## Phase 3 — Hardware Testing

- [ ] Deploy on physical F1Tenth car with live `/scan` + RealSense
- [ ] Static scene evaluation: AbsRel, MAE, RMSE vs RealSense GT (≤ 3 m)
- [ ] Dynamic scene evaluation (moving objects)

---

## Phase 4 — Polish & Report

- [ ] Evaluation script for final report metrics table (compare anchor-row vs LiDAR scale)
- [ ] Final report figures: pipeline diagram, RANSAC illustration, quantitative results
- [ ] Tune RANSAC hyperparameters if needed

---

## Known Issues / Decisions

| Issue | Decision |
|-------|----------|
| RealSense noisy beyond ~3 m | Cap GT at `max_gt_depth=3.0` for fitting and evaluation |
| Fixed anchor row 400 degenerate with 3 m cap (only 9 cm GT range) | Auto-select row with max GT depth spread per frame |
| `s ≈ 0` degenerate fits | Guard: skip frame if `\|s\| < 0.01` |
| DISPLAY env var broken in run script | Fixed: `export DISPLAY="${DISPLAY:-:0}"` |

---

## Quick Reference — Run Commands

```bash
# Offline pipeline, all frames, auto anchor, 3 m GT cap
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 \
    python metric_depth.py --data-dir testing_data --max-gt-depth 3.0

# Single frame, fixed anchor
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 \
    python metric_depth.py --data-dir testing_data --frame 0 --anchor-row 430

# Live ROS2 node (auto anchor, 3 m GT cap by default)
bash run_depth_fusion.sh

# Live node with fixed anchor row
bash run_depth_fusion.sh --ros-args -p anchor_row:=430

# RANSAC illustration figure
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 \
    python docs/ransac_illustration.py
```

---

## Progress Log

| Date       | Update |
|------------|--------|
| 2026-04-19 | Project scaffold, pipeline design, setup.sh, depth_anything_infer.py, ransac_scale.py, metric_depth.py written |
| 2026-04-20 | depth_fusion_node.py running on Jetson (CUDA, ~1 fps); DISPLAY bug fixed; degenerate s≈0 guard added |
| 2026-04-20 | 3 m GT cap added (--max-gt-depth); dynamic anchor row auto-selection implemented in both offline and live node |
| 2026-04-20 | RANSAC illustration figure generated (docs/ransac_vs_no_ransac_3m.png); all scripts ported to clean CLI args |
