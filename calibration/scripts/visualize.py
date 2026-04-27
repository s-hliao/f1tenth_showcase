#!/usr/bin/env python3
"""
Interactive visualiser: project LiDAR scan onto camera image.

Useful for verifying calibration quality or manually tuning the initial guess.

Usage
  python3 visualize.py                           # use saved extrinsic
  python3 visualize.py --extrinsic ~/calibration_data/extrinsic.npz
  python3 visualize.py --tx 0 --ty 0 --tz 0 --rx -1.5708 --rz -1.5708
  python3 scripts/visualize.py --tx 0.05 --ty 0.1 --tz 0.3 ## why is using this extrinsic directly works better than doing calibration?

Controls
  N / D    next pair
  P / A    previous pair
  +/-      increase / decrease point radius
  Q / Esc  quit
"""

import os
import sys
import argparse
import glob

import numpy as np
import cv2

_PKG = os.path.join(os.path.dirname(__file__), "..")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from calibration.calib_utils import (
    scan_to_points, project_points, params_to_transform
)

_DEFAULT_INTRINSICS = os.path.expanduser(
    "~/f1tenth_ws/src/camera_intrinsics_color.npz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def depth_to_color(depths: np.ndarray) -> np.ndarray:
    """Map depth values to JET BGR colors (near=blue, far=red)."""
    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min < 1e-3:
        norm = np.zeros(len(depths), dtype=np.uint8)
    else:
        norm = ((depths - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    colors = cv2.applyColorMap(norm[:, None], cv2.COLORMAP_JET)[:, 0, :]
    return colors  # Nx3 BGR


def overlay(image: np.ndarray, pts3d: np.ndarray,
            K: np.ndarray, T: np.ndarray, radius: int = 4) -> np.ndarray:
    out = image.copy()
    if pts3d.shape[0] == 0:
        return out

    h, w = out.shape[:2]
    u, v, mask = project_points(pts3d, K, T)
    if u.size == 0:
        return out

    # Depth in camera frame for colormap
    pts_h  = np.hstack([pts3d[mask], np.ones((mask.sum(), 1))])
    pts_cam = (T @ pts_h.T).T
    colors = depth_to_color(pts_cam[:, 2])

    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

    for i in np.where(ok)[0]:
        cv2.circle(out, (int(ui[i]), int(vi[i])),
                   radius, tuple(int(c) for c in colors[i]), -1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default=os.path.expanduser("~/calibration_data"))
    parser.add_argument("--intrinsics", default=_DEFAULT_INTRINSICS)
    parser.add_argument("--extrinsic",  default=None)
    # Manual extrinsic override
    parser.add_argument("--tx",  type=float, default=0.0)
    parser.add_argument("--ty",  type=float, default=0.0)
    parser.add_argument("--tz",  type=float, default=0.0)
    parser.add_argument("--rx",  type=float, default=1.5708)
    parser.add_argument("--ry",  type=float, default=-1.5708)
    parser.add_argument("--rz",  type=float, default=0.0)
    args = parser.parse_args()

    # Intrinsics
    d  = np.load(args.intrinsics)
    K  = d["camera_matrix"].astype(np.float64)
    dt = d["dist_coeffs"].astype(np.float64).ravel()

    # Extrinsic
    if args.extrinsic and os.path.exists(args.extrinsic):
        e = np.load(args.extrinsic)
        T = e["T_cam_lidar"].astype(np.float64)
        score = float(e.get("gradient_score", np.nan))
        print(f"Loaded extrinsic from {args.extrinsic}  (score={score:.2f})")
        print(f"T_cam_lidar:\n{T}")
    else:
        T = params_to_transform([args.tx, args.ty, args.tz,
                                 args.rx, args.ry, args.rz])
        print(f"Using manual extrinsic  tx={args.tx} ty={args.ty} tz={args.tz} "
              f"rx={args.rx} ry={args.ry} rz={args.rz}")

    # Data pairs
    files = sorted(glob.glob(os.path.join(args.data_dir, "pair_*.npz")))
    if not files:
        print(f"No pair_*.npz in {args.data_dir} — run collect_node first")
        return
    print(f"{len(files)} pairs found.  N/P to navigate, Q to quit, +/- for dot size.")

    # Undistortion maps (compute once)
    sample = np.load(files[0])
    h, w = sample["image"].shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dt, (w, h), 0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dt, None, new_K, (w, h), cv2.CV_16SC2)

    idx    = 0
    radius = 4

    while True:
        p = np.load(files[idx], allow_pickle=False)
        img_u = cv2.remap(p["image"], map1, map2, cv2.INTER_LINEAR)
        pts3d = scan_to_points(
            p["ranges"],
            float(p["angle_min"]), float(p["angle_increment"]),
            float(p["range_min"]), float(p["range_max"]),
        )

        vis = overlay(img_u, pts3d, new_K, T, radius)
        n_proj = pts3d.shape[0]

        label = (f"Pair {idx+1}/{len(files)}   pts={n_proj}   "
                 f"N/P=navigate  +/-=size  Q=quit")
        cv2.putText(vis, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("LiDAR-Camera Calibration", vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), ord("d")):
            idx = (idx + 1) % len(files)
        elif key in (ord("p"), ord("a")):
            idx = (idx - 1) % len(files)
        elif key == ord("+"):
            radius = min(radius + 1, 12)
        elif key == ord("-"):
            radius = max(radius - 1, 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
