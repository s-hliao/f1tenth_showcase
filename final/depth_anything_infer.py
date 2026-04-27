"""
depth_anything_infer.py

Loads DepthAnythingV2 and runs relative depth inference on all RGB frames.
Outputs are saved to rel_depth/ as float32 .npy files (values in arbitrary scale,
larger = farther in DepthAnythingV2's convention — we handle sign/inversion in
metric_depth.py).

Usage:
    PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 python depth_anything_infer.py
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# ---- Config ----
RGB_DIR    = Path("rgb")
OUT_DIR    = Path("rel_depth")
MODEL_TYPE = "vits"   # vits | vitb | vitl
CHECKPOINT = Path("checkpoints/depth_anything_v2_vits.pth")

# Point Python at the cloned repo
DAV2_PATH = os.environ.get("DAV2_PATH", str(Path.home() / "Depth-Anything-V2"))
sys.path.insert(0, DAV2_PATH)


def load_model():
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**configs[MODEL_TYPE])
    state = torch.load(str(CHECKPOINT), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model


def infer(model, rgb_bgr: np.ndarray) -> np.ndarray:
    """
    Run DepthAnythingV2 on a BGR image (H×W×3 uint8).
    Returns H×W float32 relative disparity (higher = closer in raw output;
    we do NOT invert here — caller decides).
    """
    # DepthAnythingV2's infer_image expects BGR uint8
    depth = model.infer_image(rgb_bgr)   # returns H×W float32 numpy
    return depth.astype(np.float32)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    rgb_files = sorted(RGB_DIR.glob("frame_*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB frames found in {RGB_DIR}/")

    print(f"Found {len(rgb_files)} RGB frames. Running inference...")
    model = load_model()

    for rgb_path in rgb_files:
        stem = rgb_path.stem                          # e.g. frame_00000
        out_path = OUT_DIR / f"{stem}.npy"

        if out_path.exists():
            print(f"  [skip] {stem} already done")
            continue

        img = cv2.imread(str(rgb_path))
        if img is None:
            print(f"  [warn] Could not read {rgb_path}, skipping")
            continue

        rel_depth = infer(model, img)
        np.save(str(out_path), rel_depth)
        print(f"  {stem}: shape={rel_depth.shape}, "
              f"min={rel_depth.min():.3f}, max={rel_depth.max():.3f}")

    print(f"\nDone. Relative depth maps saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
