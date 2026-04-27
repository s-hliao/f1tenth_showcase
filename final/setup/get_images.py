import os
import numpy as np
import cv2
import pyrealsense2 as rs

def main():
    rgb_dir = "./rgb"
    depth_npy_dir = "./depth"
    depth_img_dir = "./depth_images"
    for d in (rgb_dir, depth_npy_dir, depth_img_dir):
        os.makedirs(d, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    width, height, fps = 848, 480, 60
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale (m/unit): {depth_scale}")

    align = rs.align(rs.stream.color)

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())          # HxWx3 uint8 (BGR)
            depth_raw   = np.asanyarray(depth_frame.get_data())          # HxW   uint16 (raw units)
            depth_meters = depth_raw.astype(np.float32) * depth_scale    # HxW   float32 (meters)

            name = f"frame_{frame_count:05d}"

            cv2.imwrite(os.path.join(rgb_dir, name + ".png"), color_image)

            np.save(os.path.join(depth_npy_dir, name + ".npy"), depth_meters)

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03), 
                cv2.COLORMAP_JET,
            )
            cv2.imwrite(os.path.join(depth_img_dir, name + ".png"), depth_vis)

            frame_count += 1

            # preview = np.hstack((color_image, depth_vis))

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()