#!/usr/bin/env python3
"""
Data collection node for LiDAR-camera extrinsic calibration.

Subscribes to /scan (LaserScan) and captures RGB frames from a RealSense camera
(via pyrealsense2) or a ROS Image topic.  Synchronized pairs are saved to disk.

Usage
-----
  Terminal 1:  ros2 launch f1tenth_stack bringup_launch.py   # or equivalent
  Terminal 2:  ros2 run calibration collect_node

Controls (OpenCV window)
  SPACE  — save current (image, scan) pair
  Q/Esc  — quit

Saved files
  ~/calibration_data/pair_NNNN.npz
    keys: image, ranges, angle_min, angle_increment, range_min, range_max

ROS parameters
  scan_topic          /scan
  output_dir          ~/calibration_data
  use_realsense       true   (false → subscribe to camera_topic)
  camera_topic        /camera/color/image_raw
  cam_width/height    1280 / 720
  cam_fps             30
"""

import os
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    _HAS_RS = False

try:
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    _HAS_BRIDGE = True
except ImportError:
    _HAS_BRIDGE = False

try:
    from .calib_utils import scan_to_points, project_points, params_to_transform
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from calibration.calib_utils import scan_to_points, project_points, params_to_transform

# Hardcoded from camera_intrinsics_color.npz
_K = np.array([[895.21928782, 0., 638.37371331],
               [0., 895.77576320, 350.15291942],
               [0., 0., 1.]], dtype=np.float64)
# Nominal extrinsic: LiDAR(x=fwd,y=left,z=up) → Camera(x=right,y=down,z=fwd)
_T = params_to_transform([0., 0., 0., 1.5708, -1.5708, 0.])


class CollectNode(Node):

    def __init__(self):
        super().__init__("calibration_collect")

        self.declare_parameter("scan_topic",    "/scan")
        self.declare_parameter("output_dir",    os.path.expanduser("~/calibration_data"))
        self.declare_parameter("use_realsense", True)
        self.declare_parameter("camera_topic",  "/camera/color/image_raw")
        self.declare_parameter("cam_width",     1280)
        self.declare_parameter("cam_height",    720)
        self.declare_parameter("cam_fps",       30)

        self._scan_topic    = self.get_parameter("scan_topic").value
        self._output_dir    = self.get_parameter("output_dir").value
        self._use_rs        = self.get_parameter("use_realsense").value and _HAS_RS
        self._cam_topic     = self.get_parameter("camera_topic").value
        self._cam_w         = self.get_parameter("cam_width").value
        self._cam_h         = self.get_parameter("cam_height").value
        self._cam_fps       = self.get_parameter("cam_fps").value

        os.makedirs(self._output_dir, exist_ok=True)

        self._lock        = threading.Lock()
        self._latest_scan = None
        self._latest_img  = None
        self._pair_count  = self._count_existing_pairs()
        self._stop        = threading.Event()

        # LiDAR subscription (best-effort QoS matches the driver)
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(LaserScan, self._scan_topic,
                                 self._scan_cb, scan_qos)

        # Camera
        if self._use_rs:
            self.get_logger().info("Camera: pyrealsense2 direct capture")
            threading.Thread(target=self._rs_loop, daemon=True).start()
        elif _HAS_BRIDGE:
            self.get_logger().info(f"Camera: ROS topic {self._cam_topic}")
            self._bridge = CvBridge()
            self.create_subscription(Image, self._cam_topic,
                                     self._image_cb, 10)
        else:
            self.get_logger().error("Neither pyrealsense2 nor cv_bridge is available")

        self.get_logger().info(f"Saving pairs to: {self._output_dir}")
        self.get_logger().info(f"Scanning {self._scan_topic}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _scan_cb(self, msg: LaserScan):
        with self._lock:
            self._latest_scan = msg

    def _image_cb(self, msg):
        with self._lock:
            self._latest_img = self._bridge.imgmsg_to_cv2(msg, "bgr8")

    # ------------------------------------------------------------------
    # RealSense capture thread
    # ------------------------------------------------------------------

    def _rs_loop(self):
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color,
                              self._cam_w, self._cam_h,
                              rs.format.bgr8, self._cam_fps)
            pipeline.start(cfg)
            self.get_logger().info(
                f"RealSense pipeline started ({self._cam_w}x{self._cam_h}@{self._cam_fps}fps)")
            try:
                while not self._stop.is_set():
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                    f = frames.get_color_frame()
                    if f:
                        with self._lock:
                            self._latest_img = np.asanyarray(f.get_data())
            finally:
                pipeline.stop()
        except Exception as e:
            self.get_logger().error(f"RealSense error: {e}")

    # ------------------------------------------------------------------
    # Status check (called from main thread between captures)
    # ------------------------------------------------------------------

    def status(self) -> str:
        with self._lock:
            has_img  = self._latest_img  is not None
            has_scan = self._latest_scan is not None
            n_pts = 0
            if has_scan:
                s = self._latest_scan
                r = np.asarray(s.ranges, dtype=np.float32)
                n_pts = int(np.isfinite(r).sum())
        parts = []
        parts.append("cam=OK"   if has_img  else "cam=MISSING")
        parts.append(f"scan={n_pts}pts" if has_scan else "scan=MISSING")
        parts.append(f"saved={self._pair_count}")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    # Save one pair + write a preview JPEG for visual review
    # ------------------------------------------------------------------

    def _capture(self):
        with self._lock:
            img  = self._latest_img
            scan = self._latest_scan

        if img is None:
            self.get_logger().warn("No camera frame — is the camera topic publishing?")
            return False
        if scan is None:
            self.get_logger().warn("No LiDAR scan — is /scan publishing?")
            return False

        base = os.path.join(self._output_dir, f"pair_{self._pair_count:04d}")
        np.savez_compressed(
            base + ".npz",
            image           = img,
            ranges          = np.array(scan.ranges,        dtype=np.float32),
            angle_min       = np.float64(scan.angle_min),
            angle_increment = np.float64(scan.angle_increment),
            range_min       = np.float64(scan.range_min),
            range_max       = np.float64(scan.range_max),
        )
        # Save JPEG preview so the user can verify what was captured
        cv2.imwrite(base + ".jpg", img)

        r = np.asarray(scan.ranges, dtype=np.float32)
        n_pts = int(np.isfinite(r).sum())
        self._pair_count += 1
        self.get_logger().info(
            f"Saved pair {self._pair_count:04d}  "
            f"image={img.shape[1]}x{img.shape[0]}  scan_pts={n_pts}  → {base}.npz")
        return True

    def _count_existing_pairs(self) -> int:
        import glob as _glob
        return len(_glob.glob(os.path.join(self._output_dir, "pair_*.npz")))


# ---------------------------------------------------------------------------

def _build_frame(img, scan):
    """Overlay projected LiDAR dots and status text onto camera frame."""
    display = img.copy()
    h, w = display.shape[:2]

    if scan is not None:
        pts3d = scan_to_points(scan.ranges, scan.angle_min,
                               scan.angle_increment,
                               scan.range_min, scan.range_max)
        u, v, _ = project_points(pts3d, _K, _T)
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        for uu, vv in zip(ui[ok], vi[ok]):
            cv2.circle(display, (int(uu), int(vv)), 3, (0, 255, 0), -1)

    return display


def main(args=None):
    _gui = True
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyAllWindows()
    except cv2.error:
        _gui = False

    rclpy.init(args=args)
    node = CollectNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    if _gui:
        print("Window: SPACE=capture  Q=quit")
    else:
        print("No GUI — ENTER=capture  Ctrl+C=quit")

    try:
        while rclpy.ok() and not node._stop.is_set():
            with node._lock:
                img  = node._latest_img
                scan = node._latest_scan

            if _gui:
                if img is not None:
                    frame = _build_frame(img, scan)
                    scan_txt = f"scan: {int(np.isfinite(np.asarray(scan.ranges)).sum())}pts" if scan else "scan: MISSING"
                    cv2.putText(frame,
                                f"Pairs: {node._pair_count}  {scan_txt}  SPACE=capture  Q=quit",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 0) if scan else (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Calibration — Collect", frame)
                else:
                    blank = np.zeros((240, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for camera...", (160, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    cv2.imshow("Calibration — Collect", blank)

                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    node._capture()
                elif key in (ord("q"), 27):
                    break
            else:
                import select, sys
                st = node.status()
                print(f"\r[{st}]  ENTER=capture  Ctrl+C=quit  ", end="", flush=True)
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r:
                    line = sys.stdin.readline()
                    if not line or line.strip().lower() == "q":
                        break
                    print()
                    node._capture()

    except KeyboardInterrupt:
        pass
    finally:
        if _gui:
            cv2.destroyAllWindows()
        else:
            print()
        node._stop.set()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=3.0)
        try:
            node.destroy_node()
        except Exception:
            pass
        print(f"Done. {node._pair_count} pairs saved to {node._output_dir}")
        os._exit(0)


if __name__ == "__main__":
    main()
