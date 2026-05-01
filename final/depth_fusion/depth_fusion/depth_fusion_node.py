#!/usr/bin/env python3
"""
depth_fusion_node.py

Real-time ROS2 node: RealSense RGB + depth → DAv2 metric depth, dual-window viz.

Data acquisition uses pyrealsense2 directly (no realsense2_camera ROS driver needed),
mirroring get_images.py.  The node is still a proper rclpy node so it can be
composed with other nodes later.

Two OpenCV windows, updated each frame:
  "GT Depth (RealSense)"   — aligned RealSense depth, INFERNO colormap
  "Predicted Depth (DAv2)" — DAv2 disparity after anchor-row affine fit, same scale

Each window shows:  [ RGB | depth / predicted | colorbar ]

Pipeline (same math as offline metric_depth.py):
  RGB → DAv2 ViT-S → D_disp(u,v)
  Best anchor row: least-squares affine  D_metric = s·D_disp + t  (s < 0)
  Refit on inliers with relative error < 30 %
  D_metric = clip(s·D_disp + t, 0, max_depth)

Run (recommended — handles conda + ROS2 env automatically):
  bash run_depth_fusion.sh
  bash run_depth_fusion.sh --max-gt-depth 3.0 --anchor-row 430
  bash run_depth_fusion.sh --cam-fps 15 --display-vmax 5.0

Run via launch file (conda env must already be active):
  ros2 launch depth_fusion depth_fusion.launch.py
  ros2 launch depth_fusion depth_fusion.launch.py anchor_row:=430 max_gt_depth:=3.0

Parameters — bash flag → ROS2 param → type / default / description:
  --anchor-row R      anchor_row      int    -1     Scale-fit row.
                                                    -1 = auto-select the row with the
                                                    most GT depth spread within
                                                    max_gt_depth each frame.
  --max-gt-depth M    max_gt_depth    float  3.0    Ignore RealSense GT beyond M m.
                                                    RealSense is unreliable past ~3 m.
  --max-depth M       max_depth       float  20.0   Clip predicted depth at M m.
  --inlier-rel-tol T  inlier_rel_tol  float  0.30   RANSAC inlier relative-error threshold.
  --display-vmax V    display_vmax    float  0.0    Colormap max in m (0 = auto, GT 95th pct).
  --cam-fps N         cam_fps         int    30     RealSense stream FPS.
  --cam-width W       cam_width       int    848    RealSense stream width.
  --cam-height H      cam_height      int    480    RealSense stream height.
"""

import os
import sys
import threading
import time
import collections
import csv
from datetime import datetime

import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ---- Locate DAv2 --------------------------------------------------------
DAV2_PATH = os.environ.get("DAV2_PATH", os.path.expanduser("~/Depth-Anything-V2"))
sys.path.insert(0, DAV2_PATH)

CHECKPOINT = os.environ.get(
    "DAV2_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "../../checkpoints/depth_anything_v2_vits.pth"),
)
CHECKPOINT = os.path.realpath(CHECKPOINT)

CHECKPOINT_TRT = os.environ.get(
    "DAV2_CHECKPOINT_TRT",
    os.path.join(os.path.dirname(__file__), "../../checkpoints/depth_anything_v2_vits.engine"),
)
CHECKPOINT_TRT = os.path.realpath(CHECKPOINT_TRT)

# ---- ROS2 imports -------------------------------------------------------
import rclpy
from rclpy.node import Node

from math import cos, sin

EPS = 1e-6


# =========================================================================
class DepthFusionNode(Node):

    def __init__(self):
        super().__init__("depth_fusion")

        # ROS2 parameters
        self.declare_parameter("anchor_row",     -1)
        self.declare_parameter("max_depth",      20.0)
        self.declare_parameter("max_gt_depth",   3.0)   # RealSense unreliable beyond 3 m
        self.declare_parameter("inlier_rel_tol", 0.30)
        self.declare_parameter("display_vmax",   0.0)   # 0 = auto
        self.declare_parameter("cam_width",      848)
        self.declare_parameter("cam_height",     480)
        self.declare_parameter("cam_fps",        30)
        self.declare_parameter("use_trt",        False)
        self.declare_parameter("guidance_mode", "camera")  # "camera" | "lidar"
        self.declare_parameter("scan_topic",    "/scan")
        self.declare_parameter("tx", 0.0)   # LiDAR→camera translation (m)
        self.declare_parameter("ty", 0.0)
        self.declare_parameter("tz", 0.0)
        self.declare_parameter("rx",  1.5708)  # LiDAR→camera rotation, ZYX Euler (rad)
        self.declare_parameter("ry", -1.5708)  # R = Rz·Ry·Rx → maps x-fwd/y-left/z-up to x-right/y-down/z-fwd
        self.declare_parameter("rz",  0.0)
        self.declare_parameter("benchmark_csv", "")
        self.declare_parameter("benchmark_warmup_frames", 10)

        self.anchor_row     = self.get_parameter("anchor_row").value
        self.max_depth      = self.get_parameter("max_depth").value
        self.max_gt_depth   = self.get_parameter("max_gt_depth").value
        self.inlier_rel_tol = self.get_parameter("inlier_rel_tol").value
        self.display_vmax   = self.get_parameter("display_vmax").value
        self.cam_width      = self.get_parameter("cam_width").value
        self.cam_height     = self.get_parameter("cam_height").value
        self.cam_fps        = self.get_parameter("cam_fps").value
        self.use_trt        = self.get_parameter("use_trt").value
        self.guidance_mode  = self.get_parameter("guidance_mode").value
        self.scan_topic     = self.get_parameter("scan_topic").value
        self.tx = self.get_parameter("tx").value
        self.ty = self.get_parameter("ty").value
        self.tz = self.get_parameter("tz").value
        self.rx = self.get_parameter("rx").value
        self.ry = self.get_parameter("ry").value
        self.rz = self.get_parameter("rz").value
        self.benchmark_csv = str(self.get_parameter("benchmark_csv").value)
        self.benchmark_warmup_frames = int(self.get_parameter("benchmark_warmup_frames").value)

        self._R_lc, self._t_lc = self._build_T_lc(
            self.rx, self.ry, self.rz, self.tx, self.ty, self.tz)
        self._K = None   # set from RealSense profile once pipeline starts

        self._scan_lock   = threading.Lock()
        self._latest_scan = None

        self._model    = None
        self._model_ev = threading.Event()   # set when model is ready
        self._stop_ev  = threading.Event()

        # Latest frame shared between RealSense thread → inference thread
        self._frame_lock    = threading.Lock()
        self._latest_rgb    = None
        self._latest_gt     = None
        self._latest_frame_t = None
        self._frame_ready   = threading.Event()

        # Metrics history (time_s, abs_rel, mae) — keep last 300 frames
        self._metrics_lock    = threading.Lock()
        self._metrics_history = collections.deque(maxlen=300)
        self._t_start         = time.monotonic()

        # Benchmark CSV logger
        self._benchmark_file = None
        self._benchmark_writer = None
        self._benchmark_frame_id = 0
        self._benchmark_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._benchmark_last_row_t = None
        self._init_benchmark_logger()

        anchor_desc = str(self.anchor_row) if self.anchor_row >= 0 else "auto"
        backend = "TensorRT" if self.use_trt else "PyTorch"
        self.get_logger().info(
            f"DepthFusionNode starting — backend={backend}  "
            f"guidance={self.guidance_mode}  anchor_row={anchor_desc}  "
            f"max_gt_depth={self.max_gt_depth}m  "
            f"resolution={self.cam_width}x{self.cam_height}@{self.cam_fps}fps"
        )

        if self.guidance_mode == "lidar":
            from sensor_msgs.msg import LaserScan
            from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                                   DurabilityPolicy, HistoryPolicy)
            _qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._scan_sub = self.create_subscription(
                LaserScan, self.scan_topic, self._scan_callback, _qos)
            self.get_logger().info(
                f"LiDAR guidance — topic={self.scan_topic}  "
                f"t=[{self.tx:.3f}, {self.ty:.3f}, {self.tz:.3f}]  "
                f"r=[{self.rx:.3f}, {self.ry:.3f}, {self.rz:.3f}] rad"
            )

        # Start background threads
        threading.Thread(target=self._load_model,   daemon=True).start()
        threading.Thread(target=self._realsense_loop, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Extrinsic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_T_lc(rx, ry, rz, tx, ty, tz):
        """Build R (3×3) and t (3,) for the LiDAR→camera transform.
        rx/ry/rz: ZYX Euler angles in radians (roll, pitch, yaw).
        R = Rz·Ry·Rx so that P_cam = R @ P_lidar + t."""
        cr, sr = cos(rx), sin(rx)
        cp, sp = cos(ry), sin(ry)
        cy, sy = cos(rz), sin(rz)
        Rx = np.array([[1,  0,   0 ], [0,  cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0,  sp ], [0,  1,   0 ], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0 ], [sy, cy,   0], [0,   0,  1]])
        return Rz @ Ry @ Rx, np.array([tx, ty, tz])

    def _scan_callback(self, msg):
        with self._scan_lock:
            self._latest_scan = msg

    # ------------------------------------------------------------------
    # Benchmark CSV logging
    # ------------------------------------------------------------------
    def _init_benchmark_logger(self):
        """Open benchmark CSV if benchmark_csv param is set."""
        if not self.benchmark_csv:
            return

        path = os.path.expanduser(self.benchmark_csv)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        fieldnames = [
            "run_id",
            "frame_id",
            "timestamp_wall",
            "elapsed_s",

            "guidance_mode",
            "use_trt",
            "cam_width",
            "cam_height",
            "cam_fps_requested",
            "max_gt_depth",
            "max_depth",
            "inlier_rel_tol",
            "tx",
            "ty",
            "tz",
            "rx",
            "ry",
            "rz",
            "scan_topic",

            "capture_age_ms",
            "inference_ms",
            "scale_fit_ms",
            "metric_compute_ms",
            "viz_build_ms",
            "total_loop_ms",
            "fps_from_total_loop",
            "actual_output_fps",

            "mae",
            "absrel",
            "rmse",
            "mae_ls",
            "absrel_ls",
            "rmse_ls",
            "num_eval_pixels",

            "s",
            "t",
            "s_ls",
            "t_ls",
            "n_valid",
            "n_inliers",
            "inlier_ratio",
            "anchor_row",

            "lidar_projected_points",
            "lidar_no_disp_points",
            "lidar_inlier_points",
            "lidar_outlier_points",
        ]

        self._benchmark_file = open(path, "w", newline="")
        self._benchmark_writer = csv.DictWriter(
            self._benchmark_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        self._benchmark_writer.writeheader()
        self._benchmark_file.flush()

        self.get_logger().info(
            f"Benchmark CSV enabled: {path} "
            f"(warmup_frames={self.benchmark_warmup_frames})"
        )

    def _write_benchmark_row(self, row: dict):
        """Write one benchmark row after warmup frames."""
        frame_id = self._benchmark_frame_id
        self._benchmark_frame_id += 1

        if self._benchmark_writer is None:
            return

        if frame_id < self.benchmark_warmup_frames:
            return

        row["frame_id"] = frame_id
        self._benchmark_writer.writerow(row)
        self._benchmark_file.flush()

    def _close_benchmark_logger(self):
        if self._benchmark_file is not None:
            self._benchmark_file.flush()
            self._benchmark_file.close()
            self._benchmark_file = None
            self._benchmark_writer = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        if self.use_trt:
            self._load_model_trt()
        else:
            self._load_model_pth()

    def _load_model_pth(self):
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        model = DepthAnythingV2(**cfg)
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = model.to(device)
        self._model_ev.set()
        self.get_logger().info(f"DAv2 (.pth) loaded on {device}")

    def _load_model_trt(self):
        import sys
        sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../..")))
        from trt_infer import TRTInferencer

        self._model = TRTInferencer(CHECKPOINT_TRT)
        self._model_ev.set()
        self.get_logger().info(f"DAv2 (.engine) loaded via TensorRT")

    # ------------------------------------------------------------------
    # RealSense capture thread
    # ------------------------------------------------------------------
    def _realsense_loop(self):
        pipeline = rs.pipeline()
        cfg      = rs.config()
        cfg.enable_stream(rs.stream.color, self.cam_width, self.cam_height,
                          rs.format.bgr8, self.cam_fps)
        cfg.enable_stream(rs.stream.depth, self.cam_width, self.cam_height,
                          rs.format.z16,  self.cam_fps)

        profile = pipeline.start(cfg)
        sensor  = profile.get_device().first_depth_sensor()
        depth_scale = sensor.get_depth_scale()          # m per raw unit
        align = rs.align(rs.stream.color)

        color_intr = (profile.get_stream(rs.stream.color)
                      .as_video_stream_profile().get_intrinsics())
        self._K = np.array([
            [color_intr.fx, 0.0,           color_intr.ppx],
            [0.0,           color_intr.fy, color_intr.ppy],
            [0.0,           0.0,           1.0],
        ], dtype=np.float64)

        self.get_logger().info(
            f"RealSense started — depth_scale={depth_scale:.6f} m/unit  "
            f"K: fx={color_intr.fx:.1f} fy={color_intr.fy:.1f} "
            f"cx={color_intr.ppx:.1f} cy={color_intr.ppy:.1f}"
        )

        try:
            while not self._stop_ev.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                aligned = align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                rgb_bgr = np.asanyarray(color_frame.get_data())          # H×W×3 uint8
                depth_m = (np.asanyarray(depth_frame.get_data())
                           .astype(np.float32) * depth_scale)            # H×W float32 m

                with self._frame_lock:
                    self._latest_rgb = rgb_bgr
                    self._latest_gt  = depth_m
                    self._latest_frame_t = time.perf_counter()
                self._frame_ready.set()
        finally:
            pipeline.stop()
            self.get_logger().info("RealSense pipeline stopped")

    # ------------------------------------------------------------------
    # Inference + visualization thread
    # ------------------------------------------------------------------
    def _inference_loop(self):
        self._model_ev.wait()
        self.get_logger().info("Inference loop started")

        while not self._stop_ev.is_set():
            if not self._frame_ready.wait(timeout=1.0):
                continue
            self._frame_ready.clear()

            with self._frame_lock:
                rgb_bgr = self._latest_rgb
                d_gt    = self._latest_gt
                frame_t = self._latest_frame_t

            if rgb_bgr is None or d_gt is None:
                continue

            t_loop0 = time.perf_counter()
            capture_age_ms = (
                (t_loop0 - frame_t) * 1000.0
                if frame_t is not None else float("nan")
            )

            # ---- DAv2 inference ----
            t_infer0 = time.perf_counter()
            try:
                if not self.use_trt:
                    import torch
                    torch.cuda.empty_cache()
                d_disp = self._model.infer_image(rgb_bgr).astype(np.float32)
            except (RuntimeError, Exception) as e:
                self.get_logger().error(f"Inference failed: {e} — skipping frame")
                continue
            t_infer1 = time.perf_counter()

            # ---- Scale fit ----
            lidar_pts = []
            t_fit0 = time.perf_counter()
            try:
                if self.guidance_mode == "lidar":
                    with self._scan_lock:
                        scan = self._latest_scan
                    if scan is None:
                        self.get_logger().warn(
                            "No LiDAR scan received yet", throttle_duration_sec=2.0)
                        continue
                    if self._K is None:
                        self.get_logger().warn(
                            "Camera intrinsics not ready", throttle_duration_sec=2.0)
                        continue
                    s, t, s_ls, t_ls, n_inliers, n_valid, row, lidar_pts = \
                        self._fit_scale_lidar(d_disp, scan)
                else:
                    s, t, s_ls, t_ls, n_inliers, n_valid, row = \
                        self._fit_scale(d_disp, d_gt)
            except Exception as _e:
                import traceback as _tb
                self.get_logger().error(
                    f"Scale fit error: {_e}\n{_tb.format_exc()}")
                continue

            t_fit1 = time.perf_counter()

            if s is None:
                self.get_logger().warn("Scale fit failed — insufficient anchor pixels")
                continue

            # ---- Metric depth + error metrics ----
            t_metric0 = time.perf_counter()

            d_metric    = np.clip(s    * d_disp + t,    0.0, self.max_depth).astype(np.float32)
            d_metric_ls = np.clip(s_ls * d_disp + t_ls, 0.0, self.max_depth).astype(np.float32)

            abs_rel = mae = rmse = float("nan")
            abs_rel_ls = mae_ls = rmse_ls = float("nan")

            eval_mask = (d_gt > 0) & (d_gt <= self.max_gt_depth)
            num_eval_pixels = int(eval_mask.sum())

            if num_eval_pixels > 0:
                gt_v       = d_gt[eval_mask]
                pred_v     = d_metric[eval_mask]
                pred_ls_v  = d_metric_ls[eval_mask]

                err        = pred_v - gt_v
                err_ls     = pred_ls_v - gt_v

                abs_rel    = float(np.mean(np.abs(err)    / (gt_v + EPS)))
                mae        = float(np.mean(np.abs(err)))
                rmse       = float(np.sqrt(np.mean(err ** 2)))

                abs_rel_ls = float(np.mean(np.abs(err_ls) / (gt_v + EPS)))
                mae_ls     = float(np.mean(np.abs(err_ls)))
                rmse_ls    = float(np.sqrt(np.mean(err_ls ** 2)))

                elapsed = time.monotonic() - self._t_start
                with self._metrics_lock:
                    self._metrics_history.append((elapsed, abs_rel, mae, abs_rel_ls, mae_ls))

            t_metric1 = time.perf_counter()

            # Keep this dt for the existing on-screen annotation/log behavior.
            dt = t_metric1 - t_loop0
            t_viz0 = time.perf_counter()
            # ---- Shared colormap scale ----
            valid_gt = (d_gt > 0) & (d_gt <= self.max_gt_depth)
            if self.display_vmax > 0:
                vmax = self.display_vmax
            else:
                vmax = float(np.percentile(d_gt[valid_gt], 95)) if valid_gt.any() else 8.0

            # ---- Filter visualization: mask out pixels at/beyond depth limits ----
            # Estimated: pixels >= max_depth are invalid (clipped boundary)
            d_metric_viz = d_metric.copy()
            d_metric_viz[d_metric_viz >= self.max_depth] = 0.0

            # GT: pixels beyond max_gt_depth are unreliable — show as invalid
            d_gt_viz = d_gt.copy()
            d_gt_viz[d_gt_viz > self.max_gt_depth] = 0.0

            # ---- Colorize ----
            c_estimated = self._colorize(d_metric_viz, vmax)
            c_gt        = self._colorize(d_gt_viz,     vmax)
            colorbar    = self._make_colorbar(rgb_bgr.shape[0], 0.0, vmax)

            # Anchor row marker (camera guidance only)
            w = c_estimated.shape[1]
            if row >= 0:
                cv2.line(c_estimated, (0, row), (w - 1, row), (255, 255, 255), 1)
                cv2.line(c_gt,        (0, row), (w - 1, row), (255, 255, 255), 1)

            # Panel titles
            guidance_label = "LiDAR-Guided" if self.guidance_mode == "lidar" \
                             else "Anchor-Row Guided"
            cv2.putText(c_estimated, f"Calibrated Depth ({guidance_label})",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(c_gt, "RealSense Ground Truth",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Fit stats on estimated panel
            if row >= 0:
                ann = (f"row={row} s={s:.3f} t={t:.3f}  inliers={n_inliers}/{n_valid}  "
                       f"{1/dt:.1f}fps  vmax={vmax:.1f}m")
            else:
                ann = (f"[LiDAR] s={s:.3f} t={t:.3f}  beams={n_inliers}/{n_valid}  "
                       f"{1/dt:.1f}fps  vmax={vmax:.1f}m")
            cv2.putText(c_estimated, ann, (8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

            # LiDAR beam overlay — draw on both panels; add legend on estimated
            if lidar_pts:
                try:
                    # status → BGR color
                    _DOT = {0: (30, 165, 255),   # orange/yellow — no disparity
                            1: (0,  230,   0),    # green         — RANSAC inlier
                            2: (0,   40, 220)}    # red           — RANSAC outlier
                    for pu, pv, status in lidar_pts:
                        clr = _DOT[status]
                        cv2.circle(c_estimated, (pu, pv), 4, (0,   0,  0), -1)  # black bg
                        cv2.circle(c_estimated, (pu, pv), 3, clr,        -1)
                        cv2.circle(c_gt,        (pu, pv), 4, (0,   0,  0), -1)
                        cv2.circle(c_gt,        (pu, pv), 3, clr,        -1)
                    # Legend (y=54, below the two existing text lines at y=18,36)
                    for x_off, clr, label in [
                        (8,   (0, 230,   0), "inlier"),
                        (72,  (0,  40, 220), "outlier"),
                        (148, (30, 165, 255), "no-disp"),
                    ]:
                        cv2.circle(c_estimated, (x_off, 54), 4, (0, 0, 0), -1)
                        cv2.circle(c_estimated, (x_off, 54), 3, clr,       -1)
                        cv2.putText(c_estimated, label, (x_off + 7, 58),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1, cv2.LINE_AA)
                except Exception as _e:
                    import traceback as _tb
                    self.get_logger().error(
                        f"LiDAR overlay draw error: {_e}\n{_tb.format_exc()}")

            depth_win = np.hstack([c_estimated, c_gt, colorbar])
            metrics_win = self._make_metrics_plot()
            zoom_win    = self._make_zoom_plot()
            t_viz1 = time.perf_counter()

            with self._frame_lock:
                self._viz_depth   = depth_win
                self._viz_metrics = metrics_win
                self._viz_zoom    = zoom_win

            t_loop1 = time.perf_counter()

            total_loop_ms = (t_loop1 - t_loop0) * 1000.0
            inference_ms = (t_infer1 - t_infer0) * 1000.0
            scale_fit_ms = (t_fit1 - t_fit0) * 1000.0
            metric_compute_ms = (t_metric1 - t_metric0) * 1000.0
            viz_build_ms = (t_viz1 - t_viz0) * 1000.0

            fps_from_total_loop = (
                1000.0 / total_loop_ms if total_loop_ms > 0 else float("nan")
            )

            if self._benchmark_last_row_t is None:
                actual_output_fps = float("nan")
            else:
                frame_period = t_loop1 - self._benchmark_last_row_t
                actual_output_fps = 1.0 / frame_period if frame_period > 0 else float("nan")
            self._benchmark_last_row_t = t_loop1

            lidar_projected_points = len(lidar_pts)
            lidar_no_disp_points = sum(1 for _, _, status in lidar_pts if status == 0)
            lidar_inlier_points = sum(1 for _, _, status in lidar_pts if status == 1)
            lidar_outlier_points = sum(1 for _, _, status in lidar_pts if status == 2)

            inlier_ratio = (
                float(n_inliers) / float(n_valid)
                if n_valid > 0 else float("nan")
            )

            self._write_benchmark_row({
                "run_id": self._benchmark_run_id,
                "timestamp_wall": time.time(),
                "elapsed_s": time.monotonic() - self._t_start,

                "guidance_mode": self.guidance_mode,
                "use_trt": self.use_trt,
                "cam_width": self.cam_width,
                "cam_height": self.cam_height,
                "cam_fps_requested": self.cam_fps,
                "max_gt_depth": self.max_gt_depth,
                "max_depth": self.max_depth,
                "inlier_rel_tol": self.inlier_rel_tol,
                "tx": self.tx,
                "ty": self.ty,
                "tz": self.tz,
                "rx": self.rx,
                "ry": self.ry,
                "rz": self.rz,
                "scan_topic": self.scan_topic,

                "capture_age_ms": capture_age_ms,
                "inference_ms": inference_ms,
                "scale_fit_ms": scale_fit_ms,
                "metric_compute_ms": metric_compute_ms,
                "viz_build_ms": viz_build_ms,
                "total_loop_ms": total_loop_ms,
                "fps_from_total_loop": fps_from_total_loop,
                "actual_output_fps": actual_output_fps,

                "mae": mae,
                "absrel": abs_rel,
                "rmse": rmse,
                "mae_ls": mae_ls,
                "absrel_ls": abs_rel_ls,
                "rmse_ls": rmse_ls,
                "num_eval_pixels": num_eval_pixels,

                "s": s,
                "t": t,
                "s_ls": s_ls,
                "t_ls": t_ls,
                "n_valid": n_valid,
                "n_inliers": n_inliers,
                "inlier_ratio": inlier_ratio,
                "anchor_row": row,

                "lidar_projected_points": lidar_projected_points,
                "lidar_no_disp_points": lidar_no_disp_points,
                "lidar_inlier_points": lidar_inlier_points,
                "lidar_outlier_points": lidar_outlier_points,
            })

            if row >= 0:
                self.get_logger().info(
                    f"row={row} s={s:.3f} t={t:.3f} inliers={n_inliers}/{n_valid} "
                    f"dt={dt*1e3:.0f}ms",
                    throttle_duration_sec=2.0,
                )
            else:
                self.get_logger().info(
                    f"[lidar] s={s:.3f} t={t:.3f} beams={n_inliers}/{n_valid} "
                    f"dt={dt*1e3:.0f}ms",
                    throttle_duration_sec=2.0,
                )

    # ------------------------------------------------------------------
    # Shared helper: render a 2-row matplotlib figure to a BGR numpy array
    # Uses the OO API (Figure + FigureCanvasAgg) — no shared pyplot state.
    # ------------------------------------------------------------------
    @staticmethod
    def _render_metrics_figure(
        times, abs_rels, maes, abs_rels_ls, maes_ls,
        xlim, title_suffix="",
    ) -> np.ndarray:
        fig = Figure(figsize=(8, 4), dpi=80, facecolor="#1a1a1a")
        FigureCanvasAgg(fig)          # attaches an Agg canvas; no display needed
        ax1, ax2 = fig.subplots(2, 1)

        for ax in (ax1, ax2):
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#cccccc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#444444")

        ax1.plot(times, abs_rels,    color="#00d4ff", linewidth=1.2,
                 label=f"RANSAC ({abs_rels[-1]:.4f})")
        ax1.plot(times, abs_rels_ls, color="#ff5f5f", linewidth=1.2,
                 label=f"No RANSAC ({abs_rels_ls[-1]:.4f})")
        ax1.set_title(f"AbsRel{title_suffix}", color="#ffffff", fontsize=9, pad=3)
        ax1.set_ylabel("AbsRel", color="#cccccc", fontsize=9)
        ax1.legend(fontsize=8, facecolor="#2a2a2a", labelcolor="#cccccc", framealpha=0.7)
        ax1.grid(True, alpha=0.2, color="#555555")
        ax1.set_xlim(*xlim)

        ax2.plot(times, maes,    color="#ff9f40", linewidth=1.2,
                 label=f"RANSAC ({maes[-1]:.4f} m)")
        ax2.plot(times, maes_ls, color="#c87fff", linewidth=1.2,
                 label=f"No RANSAC ({maes_ls[-1]:.4f} m)")
        ax2.set_title(f"MAE{title_suffix}", color="#ffffff", fontsize=9, pad=3)
        ax2.set_ylabel("MAE (m)", color="#cccccc", fontsize=9)
        ax2.set_xlabel("Time (s)", color="#cccccc", fontsize=9)
        ax2.legend(fontsize=8, facecolor="#2a2a2a", labelcolor="#cccccc", framealpha=0.7)
        ax2.grid(True, alpha=0.2, color="#555555")
        ax2.set_xlim(*xlim)

        fig.tight_layout(pad=0.8)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())   # H×W×4 RGBA
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    @staticmethod
    def _blank_plot(msg="Waiting for metrics...") -> np.ndarray:
        blank = np.zeros((320, 640, 3), dtype=np.uint8)
        cv2.putText(blank, msg, (140, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return blank

    # ------------------------------------------------------------------
    # Full-history metrics plot
    # ------------------------------------------------------------------
    def _make_metrics_plot(self) -> np.ndarray:
        with self._metrics_lock:
            history = list(self._metrics_history)
        if not history:
            return self._blank_plot()

        times       = [h[0] for h in history]
        abs_rels    = [h[1] for h in history]
        maes        = [h[2] for h in history]
        abs_rels_ls = [h[3] for h in history]
        maes_ls     = [h[4] for h in history]
        xlim = (times[0], max(times[-1], times[0] + 1))
        return self._render_metrics_figure(
            times, abs_rels, maes, abs_rels_ls, maes_ls, xlim)

    # ------------------------------------------------------------------
    # Zoomed metrics plot — sliding 5-second window
    # x-axis is relative: right edge = window_s ("now"), left edge = 0 ("5 s ago")
    # ------------------------------------------------------------------
    def _make_zoom_plot(self, window_s: float = 5.0) -> np.ndarray:
        with self._metrics_lock:
            history = list(self._metrics_history)
        if not history:
            return self._blank_plot()

        t_now    = history[-1][0]
        t_left   = t_now - window_s
        windowed = [h for h in history if h[0] >= t_left] or [history[-1]]

        # Normalize: rightmost point is always at x = window_s; left edge = 0
        times       = [h[0] - t_now + window_s for h in windowed]
        abs_rels    = [h[1] for h in windowed]
        maes        = [h[2] for h in windowed]
        abs_rels_ls = [h[3] for h in windowed]
        maes_ls     = [h[4] for h in windowed]
        xlim = (0.0, window_s)
        return self._render_metrics_figure(
            times, abs_rels, maes, abs_rels_ls, maes_ls, xlim,
            title_suffix="  [last 5 s]")

    # ------------------------------------------------------------------
    # Scale fitting (anchor-row affine, same as offline pipeline)
    # ------------------------------------------------------------------
    def _best_anchor_row(self, d_disp: np.ndarray, d_gt: np.ndarray) -> int:
        """Return anchor_row if fixed; otherwise pick the row with the greatest
        GT depth range within max_gt_depth (requires ≥ 20 valid pixels)."""
        if self.anchor_row >= 0:
            return self.anchor_row
        best_row, best_range = 400, -1.0
        for row in range(d_gt.shape[0]):
            ag = d_gt[row, :]
            ad = d_disp[row, :]
            v  = (ad > 0) & (ag > 0) & (ag <= self.max_gt_depth)
            if v.sum() < 20:
                continue
            gt_range = float(ag[v].max() - ag[v].min())
            if gt_range > best_range:
                best_range = gt_range
                best_row   = row
        return best_row

    def _fit_scale(self, d_disp: np.ndarray, d_gt: np.ndarray):
        row = self._best_anchor_row(d_disp, d_gt)

        anchor_disp = d_disp[row, :]
        anchor_gt   = d_gt  [row, :]

        # Ignore GT pixels beyond max_gt_depth — RealSense is noisy there
        anchor_gt = anchor_gt.copy()
        anchor_gt[anchor_gt > self.max_gt_depth] = 0.0

        valid   = (anchor_disp > 0) & (anchor_gt > 0)
        n_valid = int(valid.sum())
        if n_valid < 2:
            return None, None, None, None, 0, n_valid, row

        A = np.stack([anchor_disp[valid], np.ones(n_valid)], axis=1)
        st, _, _, _ = np.linalg.lstsq(A, anchor_gt[valid], rcond=None)
        s_ls, t_ls = float(st[0]), float(st[1])

        pred_ls = np.clip(s_ls * anchor_disp + t_ls, 0, self.max_depth)
        rel_err = np.abs(pred_ls - anchor_gt) / (anchor_gt + EPS)
        inliers = valid & (rel_err < self.inlier_rel_tol)
        n_inliers = int(inliers.sum())

        if n_inliers >= 2:
            A_in = np.stack([anchor_disp[inliers], np.ones(n_inliers)], axis=1)
            st_in, _, _, _ = np.linalg.lstsq(A_in, anchor_gt[inliers], rcond=None)
            s, t = float(st_in[0]), float(st_in[1])
        else:
            s, t = s_ls, t_ls

        # Degenerate: anchor row has no disparity variation — scale is meaningless
        if abs(s) < 0.01:
            return None, None, None, None, 0, n_valid, row

        return s, t, s_ls, t_ls, n_inliers, n_valid, row

    def _fit_scale_lidar(self, d_disp: np.ndarray, scan):
        """Project 2D LiDAR beams to camera pixels via T_lc, then fit affine
        D_metric = s·D_disp + t on (D_disp_i, d_lidar_i) pairs with RANSAC.

        Returns 8-tuple: (s, t, s_ls, t_ls, n_inliers, n_valid, -1, pts)
        pts is a list of (u, v, status) for every in-image beam:
          status 0 = no disparity (yellow)  — projected but DAv2 gave 0
          status 1 = RANSAC inlier (green)  — used in final fit
          status 2 = RANSAC outlier (red)   — projected, sampled, but rejected
        """
        K, R, t_lc = self._K, self._R_lc, self._t_lc
        H, W = d_disp.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        n_rays = len(scan.ranges)
        angles = (scan.angle_min
                  + np.arange(n_rays, dtype=np.float64) * scan.angle_increment)
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        valid  = (np.isfinite(ranges)
                  & (ranges >= scan.range_min)
                  & (ranges <= scan.range_max))
        angles = angles[valid];  ranges = ranges[valid]
        if len(ranges) == 0:
            return None, None, None, None, 0, 0, -1, []

        # 3D in LiDAR frame (horizontal scan plane: z_L = 0)
        P_L = np.stack([ranges * np.cos(angles),
                        ranges * np.sin(angles),
                        np.zeros_like(ranges)], axis=1)   # N×3

        # Transform to camera frame
        P_C       = (R @ P_L.T).T + t_lc          # N×3
        front     = P_C[:, 2] > 0.05
        P_C       = P_C[front]
        d_beam_all = P_C[:, 2].astype(np.float32)  # z-depth in camera frame

        # Project to image; keep all in-image coords for the overlay
        u_all = (fx * P_C[:, 0] / P_C[:, 2] + cx).astype(int)
        v_all = (fy * P_C[:, 1] / P_C[:, 2] + cy).astype(int)
        in_img = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
        u_img  = u_all[in_img];  v_img  = v_all[in_img]
        d_beam = d_beam_all[in_img]

        if len(u_img) == 0:
            return None, None, None, None, 0, 0, -1, []

        # Sample disparity; remember which beams have valid (>0) disparity
        d_disp_raw = d_disp[v_img, u_img].astype(np.float64)
        nz         = d_disp_raw > EPS
        d_disp_s   = d_disp_raw[nz];  d_beam_fit = d_beam[nz]
        n_valid    = int(nz.sum())

        if n_valid < 2:
            pts = [(int(u_img[i]), int(v_img[i]), 0) for i in range(len(u_img))]
            return None, None, None, None, 0, n_valid, -1, pts

        # Least-squares affine fit
        A = np.stack([d_disp_s, np.ones(n_valid)], axis=1)
        st, _, _, _ = np.linalg.lstsq(A, d_beam_fit, rcond=None)
        s_ls, t_ls  = float(st[0]), float(st[1])

        # RANSAC inlier rejection
        pred_ls  = s_ls * d_disp_s + t_ls
        rel_err  = np.abs(pred_ls - d_beam_fit) / (d_beam_fit + EPS)
        inliers  = rel_err < self.inlier_rel_tol
        n_inliers = int(inliers.sum())

        if n_inliers >= 2:
            A_in = np.stack([d_disp_s[inliers], np.ones(n_inliers)], axis=1)
            st_in, _, _, _ = np.linalg.lstsq(A_in, d_beam_fit[inliers], rcond=None)
            s, t = float(st_in[0]), float(st_in[1])
        else:
            s, t = s_ls, t_ls

        # Build per-beam status list for overlay (0=no-disp, 1=inlier, 2=outlier)
        pts = []
        fit_i = 0
        for i in range(len(u_img)):
            if not nz[i]:
                pts.append((int(u_img[i]), int(v_img[i]), 0))
            else:
                pts.append((int(u_img[i]), int(v_img[i]), 1 if inliers[fit_i] else 2))
                fit_i += 1

        if abs(s) < 0.01:
            return None, None, None, None, 0, n_valid, -1, pts

        return s, t, s_ls, t_ls, n_inliers, n_valid, -1, pts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _colorize(depth: np.ndarray, vmax: float) -> np.ndarray:
        d = np.clip(depth / (vmax + EPS), 0.0, 1.0)
        colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        colored[depth <= 0] = 0
        return colored

    @staticmethod
    def _make_colorbar(height: int, vmin: float, vmax: float,
                       bar_width: int = 28) -> np.ndarray:
        label_px = 52
        canvas   = np.zeros((height, bar_width + label_px, 3), dtype=np.uint8)
        gradient = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
        bar_col  = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)
        canvas[:, :bar_width] = np.repeat(bar_col, bar_width, axis=1)
        for i in range(7):
            frac = i / 6
            y    = int(round(frac * (height - 1)))
            dval = vmax - frac * (vmax - vmin)
            cv2.line(canvas, (bar_width, y), (bar_width + 5, y), (200, 200, 200), 1)
            cv2.putText(canvas, f"{dval:.1f}m", (bar_width + 8, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "far",  (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "near", (2, height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        return canvas

    def shutdown(self):
        self._stop_ev.set()
        self._close_benchmark_logger()
        if self.use_trt and self._model is not None:
            self._model.close()


# =========================================================================
def main(args=None):
    rclpy.init(args=args)
    node = DepthFusionNode()

    def _spin():
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            pass

    # Spin rclpy in a background thread; main thread drives the OpenCV GUI
    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    # imshow MUST be called from the main thread on most platforms
    node._viz_depth   = None
    node._viz_metrics = None
    node._viz_zoom    = None

    try:
        while rclpy.ok():
            with node._frame_lock:
                depth_win   = node._viz_depth
                metrics_win = node._viz_metrics
                zoom_win    = node._viz_zoom

            if depth_win is not None:
                cv2.imshow("Depth Comparison (Estimated | RealSense GT)", depth_win)
            if metrics_win is not None:
                cv2.imshow("Metrics over Time (AbsRel / MAE)", metrics_win)
            if zoom_win is not None:
                cv2.imshow("Metrics — Last 5 s (AbsRel / MAE)", zoom_win)

            key = cv2.waitKey(30)
            if key == ord("q") or key == 27:   # q or Esc
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
