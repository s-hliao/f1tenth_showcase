#!/usr/bin/env python3
"""
Frenet Frame Pure Pursuit.
F1Tenth Autonomous Racing — CMU 16-663

Architecture:
    Layer 1: Cartesian to Frenet Frame Transformation.
    Layer 2: Frenet Localization and Error mapping w.r.t. Frenet Raceline ref. path
    Layer 3: Steering Commands to minimize error
    Layer 4: Velocity profiling for the whole Frenet Frame Map.

================================================================
CHANGE LOG (this revision) — see inline [CHANGE N] markers
    [CHANGE 1] Curvature feedforward  delta_ff = atan(L * kappa)
    [CHANGE 2] Feedback uses RAW e_psi (removed the lookahead-heading
               hack that was hiding a feedforward inside KD)
    [CHANGE 3] Speed-scheduled gains: KP * (V_REF/v)^2 , KD * (V_REF/v)
    [CHANGE 4] Real per-segment arc length ds[] (killed WP_SPACING=0.1,
               which was 2x wrong vs the ~0.2 m actual spacing)
    [CHANGE 5] Smoothing split: smoothed kappa drives feedforward +
               velocity only; raw theta drives the feedback error.
               Plus a steering rate limiter to absorb PF jitter.
    [CHANGE 6] Waypoint tracking always moves to the NEXT waypoints in
               the CSV: every nearest-waypoint pick must face the car's
               heading (|heading diff| < 90 deg), and a lost-track reset
               searches forward from the current index before searching
               the whole lap. The opposite-direction return leg on the
               other side of a wall can no longer be picked.
    [CHANGE 7] Coarse raceline files (e.g. the IROS 2026 optimizer CSV:
               '#' header, 9 columns, ~0.67 m spacing) are resampled on
               load to RESAMPLE_DS with a closed (periodic) cubic spline
               through x, y. All index-based constants (LOOKBACK,
               MAX_SEARCH, LOOKAHEAD_WPS, REINIT_FORWARD_WPS, kappa median
               window) then keep the spacing they were tuned on. Files
               already at <= RESAMPLE_TRIGGER spacing load exactly as before.
    [CHANGE 8] Pose guard (guard_enable, default off): the particle-filter
               heading is checked against the EKF (/odometry/filtered,
               gyro heading). If it jumps or drifts more than 20 deg from
               the EKF motion, the PF is ignored and the pose is dead-
               reckoned on the EKF, with speed capped at guard_speed_cap,
               until the PF is steady again (or 10 s). /initialpose resets
               it. Stops the steering lock + spin when the PF loses its
               heading in the wide corner.
    [CHANGE 9] Barrier-state (BaS) obstacle avoidance (bas_enable, default
               off = bit-identical bv1). bas_layer.py turns the LiDAR scan
               into obstacles in Frenet and the barrier state into d_ref, a
               sideways shift of the tracking target; the PD law then tracks
               d - d_ref instead of d, and the speed is scaled down near
               obstacles. Wall distances come from the CSV (x, y, w_left,
               w_right). speed_max / speed_min are parameters (defaults =
               the old constants) so the same code runs at several speeds.
               Debug: /bas/debug (Float32MultiArray, layout in BAS_DEBUG_FIELDS)
               and /bas/markers (RViz).
================================================================
"""

import os
import numpy as np
import math
import time                                  # [CHANGE 8]
from collections import deque                # [CHANGE 8]
import sys                                   # [CHANGE 9]
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy   # [CHANGE 9]

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, PoseWithCovarianceStamped   # [CHANGE 8] /initialpose
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.ndimage import binary_dilation, uniform_filter1d
from std_msgs.msg import Float64, Float32MultiArray   # [CHANGE 9] Float32MultiArray: /bas/debug
from scipy.ndimage import median_filter   # add to imports
from scipy.interpolate import CubicSpline   # [CHANGE 7] resampling of coarse raceline CSVs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # [CHANGE 9] bas_layer.py next to this file
from bas_layer import BasParams, BasLine, BarrierLayer, BAS_PARAM_NAMES  # [CHANGE 9]

# [CHANGE 9] /bas/debug layout, one message per control cycle
BAS_DEBUG_FIELDS = ['s', 'd', 'd_ref', 'e_psi', 'v', 'speed_cmd', 'steer', 'n_obstacles', 'active',
                    'bt_ahead', 'h_obs_min', 'speed_scale', 'obs_ds', 'obs_d', 'obs_side', 'guard_reject']



# [CHANGE 8] ------------------------------------------------------------
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PoseGuard:
    """Controller pose = EKF motion (smooth, gyro-true heading) + slow correction toward the particle filter.

    Every PF pose is compared with the guard pose moved by the EKF motion:
      inside the gates  -> blend toward the PF (position tau_pos, heading tau_head)
      outside the gates -> REJECT: dead-reckon on the EKF, PF ignored
    While rejecting, the PF is accepted again (RESYNC) only once it is STEADY: its offset from the guard pose
    changes less than steady_pos / steady_head over steady_s while the car drives at least steady_travel
    (a relocalized PF moves exactly like the EKF; a PF that is still losing track keeps drifting). max_reject_s is the hard limit for dead reckoning.
    O(1) per update (the steady window holds ~steady_s * pose rate entries)."""

    def __init__(self, head_gate=0.35, lat_gate=0.4, long_gate=1.0, tau_pos=0.3, tau_head=1.0,
                 steady_s=1.0, steady_pos=0.15, steady_head=0.05, steady_travel=1.0, max_reject_s=10.0, guard_position=False, laser_x=0.26):
        self.head_gate, self.lat_gate, self.long_gate = head_gate, lat_gate, long_gate
        self.tau_pos, self.tau_head = tau_pos, tau_head
        self.steady_s, self.steady_pos, self.steady_head = steady_s, steady_pos, steady_head
        self.steady_travel = steady_travel
        self.guard_position = guard_position   # False: only the heading is guarded, position always from the PF
        self.max_reject_s, self.lx = max_reject_s, laser_x
        self.reset()

    def reset(self):
        self.pose = None
        self.odom_prev = None
        self.t_prev = None
        self.rejecting = False
        self.t_reject = 0.0
        self.offsets = deque()           # (t, ex, ey, eyaw, travel) while rejecting, map frame
        self.travel = 0.0                # EKF distance driven [m], running total
        self.n_reject = 0
        self.n_resync = 0

    def update(self, t, pf, odom):
        """t [s]; pf = (x, y, yaw) particle-filter pose (map, laser point);
        odom = (x, y, yaw) EKF pose (odom frame, base_link) or None.  Returns ((x, y, yaw), state)."""
        if self.pose is None or odom is None or self.odom_prev is None:
            self.pose, self.odom_prev, self.t_prev = list(pf), odom, t
            return tuple(pf), 'pass'
        dt = max(1e-3, min(0.2, t - self.t_prev))
        self.t_prev = t

        # 1) move the guard pose by the EKF motion since the last update (body frame, + laser lever arm)
        ox, oy, oyaw = self.odom_prev
        nx, ny, nyaw = odom
        self.odom_prev = odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        bx, by = c * (nx - ox) + s * (ny - oy), -s * (nx - ox) + c * (ny - oy)
        dyaw = wrap(nyaw - oyaw)
        bx += self.lx * (math.cos(dyaw) - 1.0)
        by += self.lx * math.sin(dyaw)
        self.travel += math.hypot(bx, by)
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw = x + c * bx - s * by, y + s * bx + c * by, wrap(yaw + dyaw)

        # 2) compare with the particle filter
        ex, ey, eyaw = pf[0] - x, pf[1] - y, wrap(pf[2] - yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        e_long, e_lat = c * ex + s * ey, -s * ex + c * ey
        inside = abs(eyaw) <= self.head_gate
        if self.guard_position:
            inside = inside and abs(e_lat) <= self.lat_gate and abs(e_long) <= self.long_gate

        if inside:
            self.rejecting = False
            self.offsets.clear()
            ap, ah = dt / (self.tau_pos + dt), dt / (self.tau_head + dt)
            x, y, yaw = x + ap * ex, y + ap * ey, wrap(yaw + ah * eyaw)
            state = 'blend'
        else:
            if not self.rejecting:
                self.rejecting, self.t_reject = True, t
                self.offsets.clear()
                self.n_reject += 1
            self.offsets.append((t, ex, ey, eyaw, self.travel))
            while self.offsets and t - self.offsets[0][0] > self.steady_s:
                self.offsets.popleft()
            t0, ex0, ey0, eyaw0, tr0 = self.offsets[0]
            # steady only means something while the car moves: a stopped car makes any PF look steady
            steady = (t - t0 >= 0.9 * self.steady_s and self.travel - tr0 >= self.steady_travel
                      and math.hypot(ex - ex0, ey - ey0) < self.steady_pos
                      and abs(wrap(eyaw - eyaw0)) < self.steady_head)
            if steady or t - self.t_reject > self.max_reject_s:
                x, y, yaw = pf                     # PF has settled (or dead reckoning too long): take it
                self.rejecting = False
                self.offsets.clear()
                self.n_resync += 1
                state = 'resync'
            else:
                state = 'reject'
        if not self.guard_position:
            # heading-gated guard: only the heading opens the gate, and while the gate is open the position
            # is taken straight from the PF. While rejecting, the whole pose (position too) is dead-reckoned
            # on the EKF, because a PF that lost its heading has usually shifted its position as well.
            # The steady test then also checks that the PF MOVES like the EKF (a flipped PF moves the wrong way).
            if state != 'reject':
                x, y = pf[0], pf[1]
        self.pose = [x, y, yaw]
        return (x, y, yaw), state
# [CHANGE 8] end ---------------------------------------------------------


class FrenetPursuit(Node):
    def __init__(self):
        super().__init__('frenet_pursuit_node')

        self.raceline_pub = self.create_publisher(Marker, '/raceline_viz', 10)
        self.create_timer(2.0, self._visualize_raceline)

        # --- Waypoint File (parameterized for sim/real portability) ---
        self.declare_parameter('waypoint_file',
            '~/F1TenthAR/src/frenet_pursuit/waypoints/centerline.csv')
        self.WAYPOINT_FILE = self.get_parameter('waypoint_file').value

        # [CHANGE 7] Spacing the index-based constants below are used at
        # (venue centerline = 0.1 m). Files coarser than RESAMPLE_TRIGGER
        # are resampled to RESAMPLE_DS on load.
        self.RESAMPLE_DS = 0.1
        self.RESAMPLE_TRIGGER = 0.3

        # Load waypoints
        self.waypoints = self._load_waypoints()
        if self.waypoints is None or len(self.waypoints) < 20:
            self.get_logger().error('Waypoints missing or too few — node will not drive')
            self.waypoints = None
            return

        self._precompute_raceline()

        # Cold start state
        self._frenet_initialized = False
        self._last_wp_idx = 0

        # --- PD Controller ---
        self.WHEELBASE = 0.33
        self.KP = 0.618
        # [CHANGE 2/1] With the lookahead-heading hack removed and a real
        # feedforward added, KD no longer carries curve anticipation.
        # It now only damps residual heading error, so it should be SMALLER.
        # Start here and retune downward if corners feel over-steered.
        self.KD = 0.530                                # was 0.590
        self.C_US = 0.0215
        self.STEER_LIMIT = 0.4

        # [CHANGE 3] Speed-gain schedule reference.
        # Set V_REF to the speed at which KP/KD above were actually tuned.
        # At v == V_REF the schedule is identity (your tune is preserved);
        # above it softens the gains, below it stiffens them.
        self.V_REF = 3.0
        self.V_MIN_SCHED = 1.0      # floor so the schedule can't blow up near v=0

        # [CHANGE 5] Steering rate limiter (rad per callback) — absorbs
        # particle-filter jitter without lagging real corner response.
        self.STEER_RATE = 0.18
        self._prev_steer = 0.0

        # --- Velocity Profiler ---
        # [CHANGE 9] parameters (defaults = the old constants 4.8 / 2.0)
        self.declare_parameter('speed_max', 4.8)
        self.declare_parameter('speed_min', 2.0)
        self.SPEED_MAX = float(self.get_parameter('speed_max').value)
        self.SPEED_MIN = min(float(self.get_parameter('speed_min').value), self.SPEED_MAX)
        self.A_LAT = 2.5
        self.A_BRAKE = 8.0
        # [CHANGE 4] WP_SPACING constant deleted. Per-segment ds[] computed
        # in _precompute_raceline is used everywhere instead (exact, and
        # robust to non-uniform spacing on real exported racelines).
        self.LOOKAHEAD_WPS = 150
        self.current_speed = 0.0

        # --- Active Braking ---
        self.BRAKE_CURRENT = 8000.0
        self.BRAKE_THRESHOLD = 0.5

        # --- Cold Start ---
        self.WARMUP_REQUIRED = 10
        self.HINT_RESET_DIST = 2.0
        self.LOOKBACK = 10
        self.REINIT_FORWARD_WPS = 100   # [CHANGE 6] reset searches this many waypoints ahead first
        self._warmup_count = 0

        # --- Sensor State ---
        self.latest_scan = None

        # --- Subscribers ---
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        odom_topic = self.get_parameter('odom_topic').value

        self.pose_sub = self.create_subscription(
            Odometry, odom_topic, self.pose_callback, 10)
        # [CHANGE 9] with BaS on, the scan is subscribed best-effort further down
        self.declare_parameter('bas_enable', False)
        self.BAS_ENABLE = bool(self.get_parameter('bas_enable').value)
        if not self.BAS_ENABLE:
            self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self.scan_callback, 10)

        # [CHANGE 8] Pose guard: PF heading checked against the EKF motion
        self.declare_parameter('guard_enable', False)
        self.declare_parameter('guard_odom_topic', '/odometry/filtered')
        self.declare_parameter('guard_speed_cap', 1.5)
        self.declare_parameter('guard_laser_x', 0.26)
        self.GUARD_ENABLE = bool(self.get_parameter('guard_enable').value)
        self.GUARD_SPEED_CAP = float(self.get_parameter('guard_speed_cap').value)
        self.GUARD_STALE_S = 0.2
        self.guard = PoseGuard(laser_x=float(self.get_parameter('guard_laser_x').value))
        self.guard_odom = None
        self.guard_odom_t = -1.0
        self.guard_state = 'off'
        if self.GUARD_ENABLE:
            self.create_subscription(Odometry, self.get_parameter('guard_odom_topic').value,
                                     self._guard_odom_callback, 1)
            self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                     lambda m: self.guard.reset(), 1)
            self.get_logger().info(
                f'Pose guard ON: EKF {self.get_parameter("guard_odom_topic").value}, '
                f'speed cap {self.GUARD_SPEED_CAP} m/s while the PF is ignored')

        # --- Publishers ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.brake_pub = self.create_publisher(Float64, '/commands/motor/brake', 10)
        self.wp_pub = self.create_publisher(MarkerArray, '/waypoints_viz', 10)
        self.target_pub = self.create_publisher(Marker, '/frenet_target', 10)

        # [CHANGE 9] Barrier-state obstacle avoidance
        self._bas_t_prev = None
        self._bas_active_prev = False
        self._bas_next_viz = 0.0
        if self.BAS_ENABLE:
            bp = BasParams()
            for name in BAS_PARAM_NAMES:
                default = getattr(bp, name)
                self.declare_parameter('bas.' + name, default)
                setattr(bp, name, type(default)(self.get_parameter('bas.' + name).value))
            w_left, w_right, src = self._load_widths(bp.TRACK_HALF_W)
            self.bas = BarrierLayer(bp, BasLine(self.waypoints, w_left, w_right))
            sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                    history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(LaserScan, '/scan', self._bas_scan_callback, sensor_qos)
            self.bas_debug_pub = self.create_publisher(Float32MultiArray, '/bas/debug', 10)
            self.bas_marker_pub = self.create_publisher(MarkerArray, '/bas/markers', 1)
            self.get_logger().info(
                f'BaS ON: walls {bp.BAS_WALLS} ({src}, min width L {np.min(w_left):.2f} R {np.min(w_right):.2f} m), '
                f'obstacles {bp.BAS_OBSTACLES}, H_ACT_OBS {bp.H_ACT_OBS}, T_LOOK {bp.T_LOOK}, K_B_OBS {bp.K_B_OBS}, '
                f'K_BV {bp.K_BV}, speed {self.SPEED_MIN}-{self.SPEED_MAX} m/s')

        self.get_logger().info(
            f'Frenet Pursuit Node initialized — {len(self.waypoints)} waypoints, '
            f'track length {self.track_length:.1f}m, odom: {odom_topic}')


    # ============================================================
    # WAYPOINT LOADING
    # ============================================================

    def _load_waypoints(self) -> Optional[np.ndarray]:
        """
        Loads waypoints from CSV.
        Format: x, y in the first two columns; extra columns ignored.
        [CHANGE 7] Lines starting with '#' are skipped (np.loadtxt default),
        and files with mean spacing > RESAMPLE_TRIGGER are resampled.
        """
        path = os.path.expanduser(self.WAYPOINT_FILE)
        try:
            data = np.loadtxt(path, delimiter=',')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            wp = data[:, :2]
            wp = self._resample_closed(wp)   # [CHANGE 7]

            if len(wp) < 20:
                self.get_logger().error(
                    f'Only {len(wp)} waypoints — need at least 20')
                return None

            self.get_logger().info(f'Loaded {len(wp)} waypoints from {path}')
            return wp
        except FileNotFoundError:
            self.get_logger().error(f'Waypoint file not found: {path}')
            return None
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
            return None


    # [CHANGE 7] Closed-loop resampling for coarse raceline files.
    def _resample_closed(self, wp: np.ndarray) -> np.ndarray:
        seg = np.linalg.norm(np.diff(wp, axis=0, append=wp[:1]), axis=1)
        if len(wp) < 4 or float(np.mean(seg)) <= self.RESAMPLE_TRIGGER:
            return wp                                  # fine file: unchanged
        if seg[-1] < 1e-6:                             # last row repeats the first
            wp, seg = wp[:-1], seg[:-1]
        s = np.concatenate(([0.0], np.cumsum(seg)))    # arc length, loop closed
        spline = CubicSpline(s, np.vstack([wp, wp[:1]]), bc_type='periodic', axis=0)
        n = int(round(s[-1] / self.RESAMPLE_DS))
        out = spline(np.linspace(0.0, s[-1], n, endpoint=False))
        self.get_logger().info(
            f'Resampled {len(wp)} waypoints (mean spacing {np.mean(seg):.3f} m) '
            f'to {n} at {s[-1] / n:.3f} m')
        return out


    # [CHANGE 9] Wall distances for the BaS layer, matched to the (possibly resampled) waypoints.
    def _load_widths(self, half_w: float):
        n = len(self.waypoints)
        try:
            data = np.loadtxt(os.path.expanduser(self.WAYPOINT_FILE), delimiter=',')
            data = data.reshape(1, -1) if data.ndim == 1 else data
        except Exception:
            data = None
        if data is None or data.shape[1] not in (4,) and data.shape[1] < 9:
            return np.full(n, half_w), np.full(n, half_w), f'fixed half width {half_w} m'
        cl, cr = (2, 3) if data.shape[1] == 4 else (7, 8)
        wl, wr = data[:, cl], data[:, cr]
        if len(data) == n:
            return wl.copy(), wr.copy(), 'widths from CSV'
        # resampled file: interpolate by arc-length fraction around the loop
        def frac(xy):
            seg = np.linalg.norm(np.diff(xy, axis=0, append=xy[:1]), axis=1)
            s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
            return s / (s[-1] + seg[-1])
        f_raw, f_new = frac(data[:, :2]), frac(self.waypoints)
        fx = np.concatenate([f_raw - 1.0, f_raw, f_raw + 1.0])
        return (np.interp(f_new, fx, np.tile(wl, 3)), np.interp(f_new, fx, np.tile(wr, 3)),
                'widths from CSV (resampled)')

    # ============================================================
    # RACELINE PRECOMPUTATION
    # ============================================================

    # [CHANGE 5] Wrap-aware circular smoother (used for curvature only).
    def _smooth_circular(self, arr: np.ndarray, window: int = 7) -> np.ndarray:
        n = len(arr)
        if n < window:
            return arr.copy()
        pad = window // 2
        ext = np.concatenate([arr[-pad:], arr, arr[:pad]])
        # [CHANGE] median (edge-preserving) instead of mean — keeps the U-turn
        # curvature plateau sharp so the feedforward doesn't fire early on entry
        return median_filter(ext, size=window)[pad:pad + n]

    def _precompute_raceline(self):
        """
        Precomputes per-waypoint arrays:
            wp_s[i]     — cumulative arc length at waypoint i
            ds[i]       — distance from waypoint i to i+1   [CHANGE 4]
            wp_theta[i] — RAW tangent angle (feedback geometry)   [CHANGE 5]
            wp_kappa[i] — SMOOTHED curvature (feedforward + velocity)  [CHANGE 5]
        """
        n = len(self.waypoints)
        pts = self.waypoints

        # Cumulative arc length
        self.wp_s = np.zeros(n)
        for i in range(1, n):
            self.wp_s[i] = self.wp_s[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])
        self.track_length = self.wp_s[-1] + np.linalg.norm(pts[0] - pts[-1])

        # [CHANGE 4] Real per-segment spacing. ds[i] = |wp[i+1] - wp[i]|.
        # Replaces the hard-coded WP_SPACING = 0.1 (your spacing is ~0.2 m,
        # so the old constant made every speed/braking number ~sqrt(2)x off).
        self.ds = np.empty(n)
        for i in range(n):
            self.ds[i] = np.linalg.norm(pts[(i + 1) % n] - pts[i])
        self.mean_ds = float(np.mean(self.ds))

        # RAW tangent angle at each waypoint — drives Frenet error + feedback.
        # [CHANGE 5] No smoothing applied here; smoothing the feedback
        # reference is exactly what caused the steering jitter earlier.
        self.wp_theta = np.zeros(n)
        for i in range(n):
            j = (i + 1) % n
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            self.wp_theta[i] = math.atan2(dy, dx)

        # Raw curvature from raw headings, then SMOOTH it.
        # [CHANGE 5] Curvature is continuous around the loop, so a circular
        # moving average is safe here (unlike raw-angle averaging, which
        # corrupts at the +/-pi wrap). Smoothed kappa feeds feedforward and
        # the velocity profiler ONLY.
        kappa_raw = np.zeros(n)
        for i in range(n):
            p = (i - 1) % n
            dtheta = self.wp_theta[i] - self.wp_theta[p]
            while dtheta > math.pi:
                dtheta -= 2 * math.pi
            while dtheta < -math.pi:
                dtheta += 2 * math.pi
            ds_prev = self.ds[p]
            kappa_raw[i] = dtheta / ds_prev if ds_prev > 1e-6 else 0.0

        self.wp_kappa = self._smooth_circular(kappa_raw, window=5)

        self.get_logger().info(
            f'Raceline precomputed: length={self.track_length:.1f}m, '
            f'mean_ds={self.mean_ds:.3f}m, '
            f'max_kappa={np.max(np.abs(self.wp_kappa)):.3f}')


    # ============================================================
    # CALLBACKS
    # ============================================================

    def scan_callback(self, scan_msg: LaserScan):
        """Stores latest scan for future obstacle avoidance."""
        self.latest_scan = scan_msg

    # [CHANGE 9] Scan for the BaS layer: store only, processed in the next pose cycle
    def _bas_scan_callback(self, scan_msg: LaserScan):
        self.latest_scan = scan_msg
        self.bas.set_scan(np.asarray(scan_msg.ranges, dtype=np.float32),
                          scan_msg.angle_min, scan_msg.angle_increment)

    # [CHANGE 8] EKF pose for the pose guard (store only)
    def _guard_odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self.guard_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                           math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        self.guard_odom_t = time.monotonic()

    def pose_callback(self, pose_msg: Odometry):
        """Main Callback triggered via Pose data publishing."""
        if self.waypoints is None:
            return

        cx, cy, psi, v = self._extract_pose(pose_msg)
        self.current_speed = v

        # [CHANGE 8] Pose guard: replace the PF pose while its heading disagrees with the EKF
        if self.GUARD_ENABLE:
            t_now = time.monotonic()
            fresh = t_now - self.guard_odom_t < self.GUARD_STALE_S
            (cx, cy, psi), state = self.guard.update(t_now, (cx, cy, psi), self.guard_odom if fresh else None)
            if state != self.guard_state and state in ('reject', 'resync'):
                self.get_logger().warn('Pose guard: PF heading disagrees with the EKF -> ignoring PF, speed capped'
                                       if state == 'reject' else 'Pose guard: PF accepted again')
            self.guard_state = state

        # Warmup: let particle filter converge
        if self._warmup_count < self.WARMUP_REQUIRED:
            self._warmup_count += 1
            self._stop_car()
            return

        # Cold start: full search with heading awareness (ONCE)
        if not self._frenet_initialized:
            self._last_wp_idx = self._find_nearest_waypoint(cx, cy, psi)
            self._frenet_initialized = True
            wp = self.waypoints[self._last_wp_idx]
            self.get_logger().info(
                f'Frenet initialized: waypoint {self._last_wp_idx} '
                f'at ({wp[0]:.2f}, {wp[1]:.2f}), '
                f's={self.wp_s[self._last_wp_idx]:.1f}m')

        # Advance waypoint index if next one is closer
        self._update_nearest_waypoint(cx, cy, psi)   # [CHANGE 6] heading passed in

        # Frenet conversion using 2-segment projection
        s, d, e_psi = self._to_frenet(cx, cy, psi)

        # [CHANGE 9] Barrier state: d_ref shifts the tracking target sideways (0.0 when off)
        d_ref, bas_scale = 0.0, 1.0
        if self.BAS_ENABLE:
            t_now_b = time.monotonic()
            dt_b = 0.025 if self._bas_t_prev is None else min(0.05, max(0.005, t_now_b - self._bas_t_prev))
            self._bas_t_prev = t_now_b
            n_wp = len(self.waypoints)
            i_s = max(0, int(np.searchsorted(self.wp_s, s % self.track_length, side='right')) - 1) % n_wp
            d_ref, bas_scale = self.bas.step(cx, cy, psi, s, d, e_psi, float(self.wp_kappa[i_s]),
                                             self.current_speed, self._last_wp_idx, dt_b)

        # [CHANGE 1/2/3] Steering now takes current speed (for the gain
        # schedule) and uses feedforward + raw-error feedback internally.
        steering = self._pd_steering(d - d_ref, e_psi, s, self.current_speed)   # [CHANGE 9] d - d_ref

        # Velocity profiling
        target_speed = self._get_target_speed(self._last_wp_idx)
        if self.BAS_ENABLE:                                               # [CHANGE 9] slow down near obstacles
            v_floor = self.bas.p.V_BAS_MIN if self.bas.active else self.SPEED_MIN
            target_speed = max(min(v_floor, target_speed), target_speed * bas_scale)
        if self.GUARD_ENABLE and self.guard_state == 'reject':          # [CHANGE 8]
            target_speed = min(target_speed, self.GUARD_SPEED_CAP)

        self.get_logger().info(
            f'wp={self._last_wp_idx} d={d:.3f} ePsi={e_psi:.3f} '
            f'steer={steering:.3f} speed={target_speed:.1f} v={self.current_speed:.1f}')

        # Drive or brake
        overspeed = self.current_speed - target_speed

        if overspeed > self.BRAKE_THRESHOLD:
            brake_fraction = min(overspeed / 3.0, 1.0)
            brake_current = 3000.0 + brake_fraction * (self.BRAKE_CURRENT - 3000.0)

            # Send target_speed (not 0.0) — prevents double-braking lurch
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering
            drive_msg.drive.speed = max(0.0, target_speed)
            self.drive_pub.publish(drive_msg)

            # Proportional brake assist
            brake_msg = Float64()
            brake_msg.data = brake_current
            self.brake_pub.publish(brake_msg)
        else:
            # Normal driving
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering
            drive_msg.drive.speed = max(0.0, target_speed)
            self.drive_pub.publish(drive_msg)

        # Visualization
        self._visualize_frenet_target(s)
        if self.BAS_ENABLE:                                               # [CHANGE 9]
            self._bas_publish(s, d, d_ref, e_psi, target_speed, steering, bas_scale)


    # ============================================================
    # COLD START — NEAREST WAYPOINT
    # ============================================================

    def _heading_ok(self, idx: int, psi: float) -> bool:
        """[CHANGE 6] True if waypoint idx points the same way as the car (|diff| < 90 deg)."""
        return math.cos(psi - self.wp_theta[idx]) > 0.0

    def _find_nearest_waypoint(self, cx: float, cy: float, psi: float,
                               idx_window: Optional[np.ndarray] = None) -> int:
        # [CHANGE 6] Search only a given index window (forward from the current
        # waypoint) when one is passed, and only waypoints facing the car's
        # heading, so a closer opposite-direction leg is never chosen.
        n = len(self.waypoints)
        idxs = np.arange(n) if idx_window is None else np.asarray(idx_window) % n
        dx = self.waypoints[idxs, 0] - cx
        dy = self.waypoints[idxs, 1] - cy
        dists_sq = np.full(n, np.inf)
        dists_sq[idxs] = dx**2 + dy**2
        facing = np.cos(psi - self.wp_theta) > 0.0
        if np.any(facing[idxs]):
            dists_sq[~facing] = np.inf

        top_k = np.argsort(dists_sq)[:5]

        best_idx = top_k[0]
        best_score = float('inf')

        for idx in top_k:
            if not np.isfinite(dists_sq[idx]):
                continue
            dist = float(dists_sq[idx])
            heading_diff = psi - self.wp_theta[idx]
            while heading_diff > math.pi:
                heading_diff -= 2 * math.pi
            while heading_diff < -math.pi:
                heading_diff += 2 * math.pi
            score = dist + 2.0 * abs(heading_diff)
            if score < best_score:
                best_score = score
                best_idx = idx

        return int(best_idx)


    # ============================================================
    # WAYPOINT INDEX UPDATE
    # ============================================================

    def _update_nearest_waypoint(self, cx: float, cy: float, psi: float):
        n = len(self.waypoints)
        MAX_SEARCH = 50

        start = (self._last_wp_idx - self.LOOKBACK) % n

        def dist_sq(i):
            wp = self.waypoints[i % n]
            return (cx - wp[0])**2 + (cy - wp[1])**2

        best_idx = start
        best_dist = dist_sq(start)
        prev_dist = best_dist
        consecutive_rises = 0

        total_steps = min(MAX_SEARCH + self.LOOKBACK, n)

        for step in range(1, total_steps):
            candidate = (start + step) % n
            d = dist_sq(candidate)

            if d < best_dist and self._heading_ok(candidate, psi):   # [CHANGE 6]
                best_dist = d
                best_idx = candidate

            past_current = (step >= self.LOOKBACK)

            if past_current:
                if d > prev_dist:
                    consecutive_rises += 1
                    if consecutive_rises >= 2:
                        break
                else:
                    consecutive_rises = 0

            prev_dist = d

        self._last_wp_idx = best_idx


    # ============================================================
    # FRENET CONVERSION
    # ============================================================

    def _to_frenet(self, x: float, y: float, psi: float,
                   _retry: bool = False) -> Tuple[float, float, float]:
        n = len(self.waypoints)
        idx = self._last_wp_idx

        seg_a = (idx - 1) % n
        seg_b = idx

        t_a, dist_a, proj_a = self._project_onto_segment(x, y, seg_a)
        t_b, dist_b, proj_b = self._project_onto_segment(x, y, seg_b)

        if dist_a < dist_b:
            seg_idx, t = seg_a, t_a
        else:
            seg_idx, t = seg_b, t_b

        min_dist = min(dist_a, dist_b)
        if min_dist > self.HINT_RESET_DIST:
            if _retry:
                self.get_logger().error(
                    f'Frenet projection failed — dist={min_dist:.2f}m')
                return 0.0, min_dist, 0.0
            # [CHANGE 6] Look forward along the CSV first; whole lap only if
            # nothing facing the car is within HINT_RESET_DIST there.
            forward = np.arange(self._last_wp_idx - self.LOOKBACK,
                                self._last_wp_idx + self.REINIT_FORWARD_WPS)
            cand = self._find_nearest_waypoint(x, y, psi, forward)
            wp = self.waypoints[cand]
            if (wp[0] - x)**2 + (wp[1] - y)**2 > self.HINT_RESET_DIST**2:
                cand = self._find_nearest_waypoint(x, y, psi)
            self._last_wp_idx = cand
            self.get_logger().warn(
                f'Frenet re-initialized at waypoint {self._last_wp_idx}')
            return self._to_frenet(x, y, psi, _retry=True)

        next_idx = (seg_idx + 1) % n
        seg_length = np.linalg.norm(
            self.waypoints[next_idx] - self.waypoints[seg_idx])
        s = self.wp_s[seg_idx] + t * seg_length

        theta_a = self.wp_theta[seg_idx]
        theta_b = self.wp_theta[next_idx]
        dtheta = theta_b - theta_a
        while dtheta > math.pi:
            dtheta -= 2 * math.pi
        while dtheta < -math.pi:
            dtheta += 2 * math.pi
        theta_ref = theta_a + t * dtheta

        proj_x, proj_y = self._get_projection_point(seg_idx, t)
        dx = x - proj_x
        dy = y - proj_y
        d = -dx * math.sin(theta_ref) + dy * math.cos(theta_ref)

        e_psi = psi - theta_ref
        while e_psi > math.pi:
            e_psi -= 2 * math.pi
        while e_psi < -math.pi:
            e_psi += 2 * math.pi

        return s, d, e_psi


    def _project_onto_segment(self, x: float, y: float,
                            seg_idx: int) -> Tuple[float, float, Tuple[float, float]]:
        n = len(self.waypoints)
        next_idx = (seg_idx + 1) % n

        ax, ay = self.waypoints[seg_idx]
        bx, by = self.waypoints[next_idx]

        abx = bx - ax
        aby = by - ay
        apx = x - ax
        apy = y - ay

        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-10:
            t = 0.0
        else:
            t = (apx * abx + apy * aby) / ab_sq
            t = max(0.0, min(1.0, t))

        proj_x = ax + t * abx
        proj_y = ay + t * aby
        dist = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)

        return t, dist, (proj_x, proj_y)


    def _get_projection_point(self, seg_idx: int, t: float) -> Tuple[float, float]:
        n = len(self.waypoints)
        next_idx = (seg_idx + 1) % n
        ax, ay = self.waypoints[seg_idx]
        bx, by = self.waypoints[next_idx]
        return (ax + t * (bx - ax), ay + t * (by - ay))


    # ============================================================
    # POSE EXTRACTION
    # ============================================================

    def _extract_pose(self, pose_msg: Odometry) -> Tuple[float, float, float, float]:
        cx = pose_msg.pose.pose.position.x
        cy = pose_msg.pose.pose.position.y
        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w
        psi = math.atan2(2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz))
        v = float(pose_msg.twist.twist.linear.x)
        return cx, cy, psi, v


    # ============================================================
    # PD STEERING CONTROLLER
    # ============================================================

    def _pd_steering(self, d: float, e_psi: float, s: float, v: float) -> float:
        """
        delta = feedforward(curvature) + scheduled feedback(d, e_psi), then
        rate-limited and clamped.

            [CHANGE 1] feedforward = atan(L * kappa_smoothed_ahead)
            [CHANGE 2] feedback uses RAW e_psi (no lookahead-heading hack)
            [CHANGE 3] kp = KP*(V_REF/v)^2 ,  kd = KD*(V_REF/v)
            [CHANGE 5] smoothed kappa for ff; steering rate limit at the end
        """
        n = len(self.waypoints)
        idx = int(np.searchsorted(
            self.wp_s, s % self.track_length, side='right')) - 1
        idx = max(0, idx) % n

        # [CHANGE 1] Curvature feedforward. Preview ~0.6 m ahead, converted
        # to a waypoint count using the REAL mean spacing (not a constant).
        preview = max(1, int(round(0.6 / self.mean_ds)))
        kappa_ff = self.wp_kappa[(idx + preview) % n]    # smoothed curvature
        delta_ff = math.atan(self.WHEELBASE * kappa_ff) + self.C_US * (v ** 2) * kappa_ff
        # [CHANGE 3] Speed-scheduled feedback gains.
        # Position loop (d) is two integration stages from steering -> v^2.
        # Heading loop (e_psi) is one stage -> v. So KP gets the square,
        # KD gets the linear ratio. At v == V_REF this is identity.
        v_eff = max(v, self.V_MIN_SCHED)
        kp = self.KP * (self.V_REF / v_eff) ** 2
        kd = self.KD * (self.V_REF / v_eff)

        # [CHANGE 2] Feedback on the TRUE Frenet errors, no disguised ff.
        delta_fb = -kp * d - kd * e_psi

        delta = delta_ff + delta_fb

        # [CHANGE 5] Rate limit — clamp change per callback to absorb PF jitter
        delta = max(self._prev_steer - self.STEER_RATE,
                    min(self._prev_steer + self.STEER_RATE, delta))
        delta = max(-self.STEER_LIMIT, min(self.STEER_LIMIT, delta))
        self._prev_steer = delta
        return delta


    # ============================================================
    # VELOCITY PROFILER
    # ============================================================

    def _get_target_speed(self, current_wp_idx: int) -> float:
        """
        Backward-pass velocity profiler.
            1. Corner speed at each lookahead waypoint from SMOOTHED curvature
            2. Backward pass enforces braking feasibility to each next point

        [CHANGE 4] Uses real ds[] per segment, and corner speed directly from
                   v = sqrt(A_LAT / |kappa|) — no WP_SPACING, no dtheta proxy.
        [CHANGE 5] Reads smoothed kappa, so straights don't jitter the speed.
        """
        n = len(self.waypoints)
        m = self.LOOKAHEAD_WPS

        # Step 1: corner speed cap at each lookahead waypoint
        speeds = np.full(m, self.SPEED_MAX)
        for i in range(m):
            k = abs(self.wp_kappa[(current_wp_idx + i) % n])
            if k > 1e-4:
                v_curve = math.sqrt(self.A_LAT / k)
                speeds[i] = max(self.SPEED_MIN, min(self.SPEED_MAX, v_curve))

        # Step 2: backward pass — must be able to brake into each next point
        for i in range(m - 2, -1, -1):
            seg = self.ds[(current_wp_idx + i) % n]
            max_entry = math.sqrt(speeds[i + 1] ** 2 + 2 * self.A_BRAKE * seg)
            speeds[i] = min(speeds[i], max_entry)

        return float(speeds[0])


    # ============================================================
    # [CHANGE 9] BaS DEBUG + RVIZ
    # ============================================================

    def _bas_publish(self, s, d, d_ref, e_psi, speed_cmd, steer, scale):
        b, L = self.bas, self.bas.L
        if b.active != self._bas_active_prev:
            self.get_logger().info(f'BaS {"ACTIVE" if b.active else "released"}: d_ref {d_ref:+.2f} m, '
                                   f'{len(b.obstacles)} obstacle(s)')
            self._bas_active_prev = b.active
        # nearest obstacle ahead (or beside) and its pass side
        obs_ds, obs_d, obs_side = 99.0, 0.0, 0.0
        half = 0.5 * L.length
        for ob in b.obstacles:
            dsj = (ob[0] - s + half) % L.length - half              # > 0: ahead
            if -b.p.OBS_BEHIND <= dsj < obs_ds:
                obs_ds, obs_d = dsj, ob[1]
                for e in b.sides:
                    if abs((ob[0] - e[0] + half) % L.length - half) < b.p.OBS_MATCH_DS:
                        obs_side = float(e[1])
        msg = Float32MultiArray()
        msg.data = [float(v) for v in (s, d, d_ref, e_psi, self.current_speed, speed_cmd, steer, len(b.obstacles),
                                       b.active, b.bt_ahead, min(b.h_obs_min, 99.0), scale, obs_ds, obs_d, obs_side,
                                       self.GUARD_ENABLE and self.guard_state == 'reject')]
        self.bas_debug_pub.publish(msg)

        t = time.monotonic()                                         # RViz at <= 10 Hz
        if t < self._bas_next_viz:
            return
        self._bas_next_viz = t + 0.1
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for k in range(6):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = 'bas_obstacles', k, Marker.CYLINDER, Marker.ADD
            m.pose.orientation.w = 1.0
            if k < len(b.obstacles):
                _, _, r_s, r_d, x_c, y_c, psi, _, _ = b.obstacles[k]
                bt = b.obs_bt[k] if k < len(b.obs_bt) else 0.0
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x_c), float(y_c), 0.1
                m.pose.orientation.z, m.pose.orientation.w = math.sin(0.5 * psi), math.cos(0.5 * psi)
                m.scale.x, m.scale.y, m.scale.z = 2.0 * r_s, 2.0 * r_d, 0.2
                m.color.r, m.color.g, m.color.b = (1.0, 0.1, 0.1) if bt > 0.0 else (0.6, 0.6, 0.6)
                m.color.a = min(1.0, 0.4 + 0.1 * bt) if bt > 0.0 else 0.5
            else:
                m.scale.x = m.scale.y = m.scale.z = 0.01
                m.color.a = 0.0
            arr.markers.append(m)
        # shifted target: the line point 0.6 m ahead, moved sideways by d_ref
        i_t = L.index_at(s + 0.6)
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = 'bas_target', 0, Marker.SPHERE, Marker.ADD
        m.pose.orientation.w = 1.0
        m.pose.position.x = float(L.x[i_t] - d_ref * math.sin(L.psi[i_t]))
        m.pose.position.y = float(L.y[i_t] + d_ref * math.cos(L.psi[i_t]))
        m.pose.position.z = 0.15
        m.scale.x = m.scale.y = m.scale.z = 0.18
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.6, 0.0, 1.0) if b.active else (0.0, 0.8, 1.0, 0.8)
        arr.markers.append(m)
        self.bas_marker_pub.publish(arr)

    # ============================================================
    # STOP CAR
    # ============================================================

    def _stop_car(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)


    # ============================================================
    # VISUALIZATION
    # ============================================================

    def _visualize_frenet_target(self, s: float):
        idx = int(np.searchsorted(
            self.wp_s, s % self.track_length, side='right')) - 1
        idx = max(0, idx) % len(self.waypoints)

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.waypoints[idx][0])
        m.pose.position.y = float(self.waypoints[idx][1])
        m.pose.position.z = 0.1
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 1.0
        self.target_pub.publish(m)

    def _visualize_raceline(self):
        if self.waypoints is None:
            return

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'raceline'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.pose.orientation.w = 1.0

        for i in range(len(self.waypoints)):
            p = Point()
            p.x = float(self.waypoints[i][0])
            p.y = float(self.waypoints[i][1])
            p.z = 0.0
            m.points.append(p)

        p = Point()
        p.x = float(self.waypoints[0][0])
        p.y = float(self.waypoints[0][1])
        p.z = 0.0
        m.points.append(p)

        self.raceline_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = FrenetPursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
