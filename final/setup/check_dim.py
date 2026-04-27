import numpy as np
import cv2
import os

def check_dimensions():
    rgb_dir = "./rgb"
    depth_dir = "./depth"

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])


    img = cv2.imread(os.path.join(rgb_dir, rgb_files[0]))
    depth = np.load(os.path.join(depth_dir, depth_files[0]))

    print("-" * 30)
    print(f"Checking: {rgb_files[0]} vs {depth_files[0]}")
    print(f"RGB Shape:   {img.shape}")
    print(f"Depth Shape: {depth.shape}")
    print("-" * 30)


if __name__ == "__main__":
    check_dimensions()