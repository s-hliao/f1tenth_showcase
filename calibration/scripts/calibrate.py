#!/usr/bin/env python3
"""
Offline targetless LiDAR-camera extrinsic calibration using natural-scene edges.

Algorithm (Geiger et al.-style gradient-alignment)
  For each synchronized (image, scan) pair:
    1. Undistort the image with known camera intrinsics.
    2. Compute Sobel gradient-magnitude map G(u,v).
    3. Project LiDAR scan points into the image with candidate T_cam_lidar.
    4. Sum G at projected pixel positions  →  frame score.
  Objective: maximise total score over all frames.
  Optimiser: Nelder-Mead simplex (two-phase: coarse then fine).
  Result: 4×4 T_cam_lidar  saved as .npz + .yaml.

Usage
  python3 calibrate.py
  python3 calibrate.py --data-dir ~/calibration_data --intrinsics ~/f1tenth_ws/src/camera_intrinsics_color.npz
  python3 calibrate.py --tx 0.0 --ty 0.0 --tz 0.0 --rx -1.5708 --ry 0.0 --rz -1.5708

Tips
  • Collect ≥ 15 pairs from diverse car positions (vary distance and angle to walls).
  • The initial guess for rx/rz should match the physical sensor orientation.
  • If the score does not improve, try a different initial guess for the rotations.
"""

import os
import sys
import argparse
import glob

import numpy as np
import cv2

# Allow running the script directly from the scripts/ directory
_PKG = os.path.join(os.path.dirname(__file__), "..")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from calibration.calib_utils import (
    scan_to_points,
    project_points,
    compute_gradient,
    sample_gradient,
    params_to_transform,
    nelder_mead,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_pairs(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "pair_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No pair_*.npz files found in {data_dir}\n"
            "Run 'ros2 run calibration collect_node' first.")
    print(f"[load]  {len(files)} pairs from {data_dir}")
    pairs = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        pairs.append({
            "image":           d["image"],
            "ranges":          d["ranges"],
            "angle_min":       float(d["angle_min"]),
            "angle_increment": float(d["angle_increment"]),
            "range_min":       float(d["range_min"]),
            "range_max":       float(d["range_max"]),
        })
    return pairs


def load_intrinsics(path: str):
    if path.endswith(".npz"):
        d = np.load(path)
        K    = d["camera_matrix"].astype(np.float64)
        dist = d["dist_coeffs"].astype(np.float64).ravel()
    else:
        raise ValueError(f"Unsupported intrinsics format: {path}")
    print(f"[intrinsics]  K:\n{K}\n  dist: {dist}")
    return K, dist


def save_result(out_path: str, T: np.ndarray, params: np.ndarray, score: float):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, T_cam_lidar=T, params=params, gradient_score=score)
    print(f"[save]  {out_path}")

    yaml_path = out_path.replace(".npz", ".yaml")
    tx, ty, tz, rx, ry, rz = params
    with open(yaml_path, "w") as fh:
        fh.write("# Extrinsic calibration — T_cam_lidar (camera frame FROM lidar frame)\n")
        fh.write(f"# gradient_score: {score:.4f}\n")
        fh.write(f"tx: {tx:.8f}   # metres\n")
        fh.write(f"ty: {ty:.8f}   # metres\n")
        fh.write(f"tz: {tz:.8f}   # metres\n")
        fh.write(f"rx: {rx:.8f}   # roll  (rad, ZYX Euler)\n")
        fh.write(f"ry: {ry:.8f}   # pitch (rad)\n")
        fh.write(f"rz: {rz:.8f}   # yaw   (rad)\n")
        fh.write("T_cam_lidar:  # row-major 4x4\n")
        for row in T:
            fh.write(f"  - [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}]\n")
    print(f"[save]  {yaml_path}")


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

def precompute(pairs, K: np.ndarray, dist: np.ndarray):
    """Undistort images, compute gradients, and extract scan points once."""
    grads = []
    pts_list = []

    h, w = pairs[0]["image"].shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (w, h), cv2.CV_16SC2)

    for p in pairs:
        img_u = cv2.remap(p["image"], map1, map2, cv2.INTER_LINEAR)
        grads.append(compute_gradient(img_u, sigma=2.0))
        pts_list.append(scan_to_points(
            p["ranges"], p["angle_min"], p["angle_increment"],
            p["range_min"], p["range_max"],
        ))

    return grads, pts_list, new_K


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(grads, pts_list, K: np.ndarray):
    """Return a closure that evaluates -total_gradient_score for a param vector."""
    K = K.astype(np.float64)

    def objective(params):
        T = params_to_transform(params)
        total = 0.0
        for grad, pts in zip(grads, pts_list):
            if pts.shape[0] == 0:
                continue
            u, v, _ = project_points(pts, K, T)
            total += sample_gradient(grad, u, v)
        return -total   # minimise negative score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Targetless LiDAR-camera calibration via gradient alignment")
    parser.add_argument("--data-dir",   default=os.path.expanduser("~/calibration_data"))
    parser.add_argument("--intrinsics", default=os.path.expanduser(
        "~/f1tenth_ws/src/camera_intrinsics_color.npz"))
    parser.add_argument("--output",     default=os.path.expanduser(
        "~/calibration_data/extrinsic.npz"))
    # Initial guess
    parser.add_argument("--tx",  type=float, default=0.0,
                        help="initial tx (m): LiDAR→camera lateral offset")
    parser.add_argument("--ty",  type=float, default=0.0,
                        help="initial ty (m): LiDAR→camera vertical offset")
    parser.add_argument("--tz",  type=float, default=0.0,
                        help="initial tz (m): LiDAR→camera depth offset")
    parser.add_argument("--rx",  type=float, default=1.5708,
                        help="initial roll  (rad)  default=π/2")
    parser.add_argument("--ry",  type=float, default=-1.5708,
                        help="initial pitch (rad)  default=-π/2")
    parser.add_argument("--rz",  type=float, default=0.0,
                        help="initial yaw   (rad)")
    # Optimiser tuning
    parser.add_argument("--coarse-t-step", type=float, default=0.05,
                        help="coarse translation step (m)")
    parser.add_argument("--coarse-r-step", type=float, default=0.10,
                        help="coarse rotation step (rad)")
    parser.add_argument("--fine-t-step",   type=float, default=0.005)
    parser.add_argument("--fine-r-step",   type=float, default=0.01)

    args = parser.parse_args()

    pairs   = load_pairs(args.data_dir)
    K, dist = load_intrinsics(args.intrinsics)

    print("\n[precompute]  undistorting images and computing gradients …")
    grads, pts_list, K_new = precompute(pairs, K, dist)

    total_pts = sum(p.shape[0] for p in pts_list)
    print(f"[precompute]  total valid scan points across all frames: {total_pts}")
    if total_pts == 0:
        print("ERROR: no valid scan points — check range_min/range_max in collected data")
        return

    objective = make_objective(grads, pts_list, K_new)

    x0 = np.array([args.tx, args.ty, args.tz, args.rx, args.ry, args.rz])
    initial_score = -objective(x0)
    print(f"\n[init]  params: {x0}")
    print(f"[init]  gradient score: {initial_score:.2f}")

    # ---- Phase 1: coarse ------------------------------------------------
    step_coarse = np.array([
        args.coarse_t_step, args.coarse_t_step, args.coarse_t_step,
        args.coarse_r_step, args.coarse_r_step, args.coarse_r_step,
    ])
    print(f"\n[phase 1]  coarse optimisation  step={step_coarse[:3]}m / {step_coarse[3:]}rad …")
    x1, f1 = nelder_mead(objective, x0, step_coarse,
                         max_iter=600, tol=1e-5, verbose=True)
    print(f"[phase 1]  score={-f1:.2f}  params={x1}")

    # ---- Phase 2: fine --------------------------------------------------
    step_fine = np.array([
        args.fine_t_step, args.fine_t_step, args.fine_t_step,
        args.fine_r_step, args.fine_r_step, args.fine_r_step,
    ])
    print(f"\n[phase 2]  fine optimisation  step={step_fine[:3]}m / {step_fine[3:]}rad …")
    x_opt, f_opt = nelder_mead(objective, x1, step_fine,
                               max_iter=3000, tol=1e-7, verbose=True)

    final_score = -f_opt
    T_opt = params_to_transform(x_opt)

    print(f"\n{'='*60}")
    print(f"RESULT   gradient score: {initial_score:.2f}  →  {final_score:.2f}"
          f"  (improvement: {final_score - initial_score:+.2f})")
    print(f"  tx={x_opt[0]:.6f}  ty={x_opt[1]:.6f}  tz={x_opt[2]:.6f}")
    print(f"  rx={x_opt[3]:.6f}  ry={x_opt[4]:.6f}  rz={x_opt[5]:.6f}")
    print(f"T_cam_lidar:\n{T_opt}")
    print(f"{'='*60}")

    save_result(args.output, T_opt, x_opt, final_score)
    print(f"\nDone.  Verify with:  python3 visualize.py --extrinsic {args.output}")


if __name__ == "__main__":
    main()
