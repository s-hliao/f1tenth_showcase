#!/bin/bash
# Setup script for DepthAnythingV2
# Run once from the project directory: bash setup.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DAV2_DIR="$HOME/Depth-Anything-V2"
CKPT_DIR="$PROJECT_DIR/checkpoints"

echo "=== Step 1: Clone DepthAnythingV2 ==="
if [ ! -d "$DAV2_DIR" ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$DAV2_DIR"
else
    echo "Already cloned at $DAV2_DIR"
fi

echo "=== Step 2: Install Python dependencies ==="
pip install timm einops

echo "=== Step 3: Download ViT-S checkpoint ==="
mkdir -p "$CKPT_DIR"
CKPT_PATH="$CKPT_DIR/depth_anything_v2_vits.pth"
if [ ! -f "$CKPT_PATH" ]; then
    # Official HuggingFace release
    wget -O "$CKPT_PATH" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
else
    echo "Checkpoint already exists at $CKPT_PATH"
fi

echo ""
echo "=== Setup complete ==="
echo "DAV2_PATH=$DAV2_DIR"
echo "Checkpoint: $CKPT_PATH"
echo ""
echo "Run the pipeline with:"
echo "  DAV2_PATH=$DAV2_DIR python3 metric_depth.py"
