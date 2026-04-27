#!/usr/bin/env python3
"""
export_trt.py

Export the DAv2 ViT-S checkpoint to a TensorRT engine.

Steps:
  1. Load depth_anything_v2_vits.pth on CPU
  2. Export to ONNX  (checkpoints/depth_anything_v2_vits.onnx)
  3. Build TRT engine via trtexec  (checkpoints/depth_anything_v2_vits.engine)

Usage:
    conda run -n vision_lab_v2 python export_trt.py
    conda run -n vision_lab_v2 python export_trt.py --workspace 4096
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

SCRIPT_DIR   = os.path.dirname(os.path.realpath(__file__))
DAV2_PATH    = os.environ.get("DAV2_PATH", os.path.expanduser("~/Depth-Anything-V2"))
CKPT_DIR     = os.path.join(SCRIPT_DIR, "checkpoints")
PTH_PATH     = os.path.join(CKPT_DIR, "depth_anything_v2_vits.pth")
ONNX_PATH    = os.path.join(CKPT_DIR, "depth_anything_v2_vits.onnx")
ENGINE_PATH  = os.path.join(CKPT_DIR, "depth_anything_v2_vits.engine")
TRTEXEC      = "/usr/src/tensorrt/bin/trtexec"
INPUT_SIZE   = 518   # fixed square — matches TRTInferencer in trt_infer.py


def export_onnx(pth_path: str, onnx_path: str) -> None:
    sys.path.insert(0, DAV2_PATH)
    from depth_anything_v2.dpt import DepthAnythingV2

    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    state = torch.load(pth_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    print(f"Exporting ONNX → {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=14,
            input_names=["image"],
            output_names=["depth"],
            do_constant_folding=True,
        )
    print("ONNX export done.")


def build_engine(onnx_path: str, engine_path: str, workspace_mb: int) -> None:
    if not os.path.isfile(TRTEXEC):
        sys.exit(f"trtexec not found at {TRTEXEC}")

    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--verbose",
    ]
    print("Building TRT engine (this takes a few minutes)…")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("trtexec failed — see output above.")
    print(f"Engine saved → {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Export DAv2 ViT-S to TensorRT")
    parser.add_argument("--pth",       default=PTH_PATH,    help="Source .pth checkpoint")
    parser.add_argument("--onnx",      default=ONNX_PATH,   help="Intermediate ONNX path")
    parser.add_argument("--engine",    default=ENGINE_PATH, help="Output .engine path")
    parser.add_argument("--workspace", default=2048, type=int, metavar="MB",
                        help="trtexec workspace in MiB (default 2048)")
    parser.add_argument("--skip-onnx", action="store_true",
                        help="Skip ONNX export (reuse existing .onnx)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.engine), exist_ok=True)

    if not args.skip_onnx:
        export_onnx(args.pth, args.onnx)
    else:
        print(f"Skipping ONNX export, using {args.onnx}")

    build_engine(args.onnx, args.engine, args.workspace)


if __name__ == "__main__":
    main()
