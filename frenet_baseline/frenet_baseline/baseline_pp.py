#!/usr/bin/env python3
"""
Baseline v2: Frenet-frame pure pursuit for F1TENTH (CMU, IROS 2026 Roboracer).

Data flow (one control cycle, triggered by every pose message):

    [1] POSE INPUT        pose_callback: x, y, yaw, v, yaw_rate, stamps
    [2] TRANSFORMS        pose -> rear axle -> Frenet (s, d, e_psi, kappa)
    [3] AVOIDANCE (BaS)   [3a] LiDAR obstacles (only when a new scan arrived)
                          [3b] barrier functions h, B~, z_dot for walls + obstacles
                          [3c] barrier state -> d_ref (sideways target shift) + speed scale
    [4] REFERENCE         lookahead point (shifted by d_ref), curvature preview, speed reference
    [5] PHYSICS           steering law (PURE_PURSUIT | FRENET_PD), understeer
                          correction, lateral-accel cap, steering-rate limit
    [6] SPEED + BRAKE     speed command, optional VESC brake current
    [7] OUTPUT            /drive (+ /commands/motor/brake)
    [8] TELEMETRY         race_msgs/Telemetry on /telemetry (record only)

File layout follows that order:
    SECTION 0  parameters
    SECTION 1  raceline load + precompute (runs once at startup, not in the loop)
    SECTION 2  control cycle, stages [2]..[6] as methods in data-flow order
    SECTION 3  ROS node: [1] input, calls SECTION 2, [7] output, [8] telemetry

Comment tags:
    # >>> TELEMETRY  field(s) written into the Telemetry message at this point
    # >>> METRIC     post-run metric that field feeds (computed offline on
                     manual trigger, never on the car)

Cost per cycle: O(k) time, O(1) extra memory.
    Point search runs ONCE (start-up). After that the car is tracked by arc
    length s: projection onto the current segment, walking k segments forward
    as the car moves (k = v*dt/ds, about 1-2 at 8 m/s and 50 Hz).
    Every other lookup (lookahead, curvature preview, speed) is s -> index in
    O(1): the raceline is uniformly spaced, so index = floor(s / ds).
    One output object is reused every cycle, so the loop allocates nothing
    except the ROS messages. A full O(N) search runs again only if the car is
    lost (|d| > RELOC_DIST_M), e.g. after a spin or a sim reset.
    Avoidance (AVOID_ENABLE): barriers O(K) per cycle, K = confirmed obstacles.
    Obstacle detection O(M*W) numpy per NEW scan only (M kept beams, W raceline
    points in the detection window), not per pose cycle.
"""

import glob
import math
import os
import time
from dataclasses import dataclass, fields
from typing import Callable

import numpy as np
from collections import deque

TWO_PI = 2.0 * math.pi
G = 9.81


def wrap(a: float) -> float:
    """Angle to [-pi, pi). O(1), no loops."""
    return (a + math.pi) % TWO_PI - math.pi


# ============================================================
# SECTION 0: PARAMETERS
# ============================================================

@dataclass
class Params:
    # --- raceline (make_racelines.py output, or a legacy 2-column x,y CSV) ---
    raceline_file: str = '~/F1TenthAR/tracks/iros_2026_sim/racelines/raceline_stable.csv'
    track_id: str = 'iros_2026_sim'          # recorded in the run registry only

    # --- vehicle (config/vehicle_params.yaml) ---
    WHEELBASE: float = 0.325                 # L [m]
    POSE_TO_REAR_AXLE_M: float = 0.0         # pose point -> rear axle, along -heading [m]
                                             # car: base_link = rear axle -> 0.0
    STEER_LIMIT: float = 0.44                # |delta| max [rad]
    STEER_RATE_MAX: float = 9.0              # steering slew limit [rad/s] (car-proven: 0.18 rad per 20 ms cycle)
    STEER_STEP_MAX: float = 0.0              # >0: slew limit in rad PER CYCLE instead (old node: 0.18)
    C_SF: float = 4.718                      # front cornering stiffness per unit load [1/rad]
    C_SR: float = 5.4562                     # rear  cornering stiffness per unit load [1/rad]
    K_US: float = -1.0                       # understeer gradient [rad per m/s^2]; <0 = auto from C_SF, C_SR
    A_Y_MAX: float = 4.0                     # lateral-accel cap on steering [m/s^2]; 0 = off
                                             # (off by default: in SIM-1 a 7 m/s^2 cap blocked the
                                             #  corrective steer and caused a wall hit at 8 m/s;
                                             #  friction is already respected by the raceline vx)

    # --- steering law selection ---
    STEER_LAW: str = 'FRENET_PD'          # 'PURE_PURSUIT' | 'FRENET_PD'

    # pure pursuit: lookahead = clamp(LA_TIME * v, LA_MIN, LA_MAX)
    LA_TIME: float = 1.0                    # [s]
    LA_MIN: float = 0.8                      # [m]
    LA_MAX: float = 2.0                      # [m]

    # Frenet PD: delta = atan(L*k_ff) + K_US*v^2*k_ff - kp*(d-d_ref) - kd*e_psi
    KP: float = 0.45                        # at V_REF, scheduled kp = KP*(V_REF/v)^2
    KD: float = 0.55                        # at V_REF, scheduled kd = KD*(V_REF/v)
    V_REF: float = 3.0
    V_MIN_SCHED: float = 1.0
    FF_PREVIEW_T: float = 0.12               # curvature preview = max(FF_PREVIEW_MIN, v*T) [s]
    FF_PREVIEW_MIN: float = 0.3              # [m]

    # reaction scale on the CORRECTION part of steering (both laws); the path-following part
    # (curvature of the line itself) is never scaled.  scale = clamp((REACT_V0 / v)^REACT_EXP, REACT_MIN, 1)
    # Full reaction below REACT_V0, less reaction as speed grows: the same steering moves the car
    # sideways faster at speed (~v^2) and a fixed delay is more metres late, so corrections must soften.
    REACT_V0: float = 0.0                    # [m/s] 0 = off (steering exactly as before)
    REACT_EXP: float = 2.0                   # 2 = same inverse-square shape as bv1's gain schedule
    REACT_MIN: float = 0.2                   # floor, so corrections never vanish at top speed

    # --- speed ---
    SPEED_SCALE: float = 0.6                 # x raceline vx (the 8 m/s ladder knob)
    SPEED_MAX: float = 8.0                   # [m/s]
    SPEED_MIN: float = 1.0                   # [m/s]
    SPEED_PREVIEW_T: float = 0.20            # read vx this far ahead to cover drive lag [s]
    RACELINE_XY_ONLY: bool = False           # true: use only x,y of any CSV and derive kappa + speed like the old node
    # x,y-only mode (legacy CSV or RACELINE_XY_ONLY): speed profile built once at load with these limits
    LEGACY_A_Y: float = 1.8
    LEGACY_A_BRAKE: float = 4.0
    LEGACY_A_ACCEL: float = 3.0
    LEGACY_KAPPA_MEDIAN: int = 5             # median window on derived curvature (old node: 5)

    # --- brake channel (VESC current); off in sim ---
    BRAKE_ENABLE: bool = True
    BRAKE_THRESHOLD: float = 0.5             # overspeed that triggers braking [m/s]
    BRAKE_CURRENT_MIN: float = 3000.0
    BRAKE_CURRENT_MAX: float = 8000.0

    # --- avoidance: barrier state (BaS) path shaping; off = baseline unchanged ---
    AVOID_ENABLE: bool = False
    BAS_WALLS: bool = True                   # wall barriers on d_ref and on the car's front bumper
    BAS_OBSTACLES: bool = True               # LiDAR obstacle barriers
    # car footprint relative to the rear axle (vehicle_params.yaml)
    CAR_HALF_WIDTH: float = 0.15
    CAR_FRONT: float = 0.414                 # rear axle -> front bumper [m]
    CAR_REAR: float = 0.135                  # rear axle -> rear bumper [m]
    BAS_MARGIN: float = 0.05                 # extra clearance [m]
    TRACK_HALF_W: float = 0.8                # walls for legacy x,y files (no width columns)
    # barrier shape
    H_ACT_WALL: float = 0.10                 # wall barrier switches on below this clearance [m]
    H_ACT_OBS: float = 2.0                   # obstacle barrier switches on below this ellipse level
    T_LOOK: float = 0.45                     # ellipse grows ahead by T_LOOK * v [s]
    T_LOOK_WALL: float = 0.30                # front bumper predicted this far ahead for the wall check [s]
    H_MIN: float = 1e-3
    B_MAX: float = 200.0
    ZDOT_MAX: float = 500.0
    DENOM_MIN: float = 0.1                   # floor for (1 - kappa*d)
    # barrier state -> d_ref dynamics:  d_ref_dot = sum side*max(0, K_B*B~ + K_DB*z_dot) - K_RELAX*d_ref
    K_B_WALL: float = 0.30
    K_DB_WALL: float = 0.0005
    K_B_OBS: float = 2.0
    K_DB_OBS: float = 0.10
    K_RELAX: float = 1.5                     # pull back to the raceline [1/s]
    DREF_RATE_MAX: float = 3.0               # |d_ref_dot| limit [m/s]
    DREF_EPS: float = 0.005                  # snap d_ref to exactly 0 below this when inactive [m]
    GAP_HYST: float = 0.05                   # side switch needs the other gap this much bigger [m]
    H_SWITCH: float = 1.0                    # side may only switch while the obstacle ellipse level h > this
    K_BV: float = 0.5                        # speed scale 1/(1 + K_BV * sum B~ ahead)
    K_BV_RETURN: float = 1.0                 # while back-stepping to the line: 1/(1 + K_BV_RETURN*|d_ref|)
    V_BAS_MIN: float = 1.0                   # never slower than this while avoiding [m/s]
    # LiDAR obstacle detection
    LASER_X: float = 0.26                    # laser position from the rear axle [m]
    LASER_Y: float = 0.0
    SCAN_STRIDE: int = 2                     # use every 2nd beam
    SCAN_FOV_DEG: float = 200.0              # beams used, centred ahead
    RANGE_MIN: float = 0.10
    OBS_RANGE: float = 6.0                   # detection distance [m]
    OBS_BEHIND: float = 0.6                  # keep obstacles this far behind the rear axle [m]
    WALL_GATE: float = 0.15                  # points closer than this to a wall are wall, not obstacle [m]
    CLUSTER_GAP: float = 0.25                # beam-to-beam jump that starts a new obstacle [m]
    MIN_PTS: int = 3                         # smallest cluster that counts
    OBS_R_MIN: float = 0.15                  # minimum obstacle half-size [m]
    OBS_PERSIST: int = 2                     # scans in a row before obstacles are trusted
    OBS_CLEAR: int = 4                       # scans without obstacles before they are dropped
    OBS_BIRTH_RANGE: float = 0.0             # [BIRTH] a NEW obstacle must be first seen farther than this [m];
                                             # closer clusters count only if they continue a known obstacle.
                                             # A real static obstacle is seen from metres away; a cluster that
                                             # appears right in front (floor bump, the car itself) is not one. 0 = off
    OBS_MATCH_DS: float = 1.0                # [BIRTH] same obstacle if within this along the line [m]...
    OBS_MATCH_DD: float = 0.5                # ...and this sideways [m]
    OBS_EDGE_JUMP: float = 0.0               # [EDGE] an obstacle stands in front of what is behind it: the beams just
                                             # outside the cluster must read at least this much farther [m] on
                                             # OBS_EDGE_SIDES side(s). A wall piece shifted into the track by a pose
                                             # error is continuous with the rest of the wall (no jump). 0 = off
    OBS_EDGE_SIDES: int = 1                  # [EDGE] 1 = one side is enough (obstacle may touch a wall), 2 = both

    # --- tracking / robustness ---
    MAX_WALK_SEGS: int = 50                  # segments walked per cycle before declaring lost
    RELOC_DIST_M: float = 1.5                # farther than this from the line = relocalize
    WARMUP_CYCLES: int = 10                  # publish stop while pose settles

    # --- pose guard: particle-filter heading checked against the EKF (gyro) motion ---
    # Controller heading = EKF heading change + slow pull toward the PF heading (GUARD_TAU_HEAD).
    # A PF heading that jumps/drifts more than GUARD_HEAD_GATE from that is ignored (dead-reckon on the EKF,
    # speed capped) until the PF is steady again or GUARD_MAX_REJECT_S passes. /initialpose resets it.
    GUARD_ENABLE: bool = True
    GUARD_HEAD_GATE: float = 0.35            # [rad] ~20 deg
    GUARD_TAU_HEAD: float = 1.0              # [s] pull toward the PF heading when they agree
    GUARD_POSITION: bool = False             # also guard position (off: position always from the PF)
    GUARD_LAT_GATE: float = 0.4              # [m] cross-track gate (GUARD_POSITION only)
    GUARD_LONG_GATE: float = 1.0             # [m] along-track gate (GUARD_POSITION only)
    GUARD_TAU_POS: float = 0.3               # [s] (GUARD_POSITION only)
    GUARD_STEADY_S: float = 1.0              # PF accepted again once its offset is steady this long...
    GUARD_STEADY_HEAD: float = 0.05          # [rad] ...changing less than this
    GUARD_STEADY_POS: float = 0.15           # [m]   ...and moving like the EKF within this
    GUARD_STEADY_TRAVEL: float = 1.0         # [m]   ...while the car drives at least this far
    GUARD_MAX_REJECT_S: float = 10.0         # hard limit on dead reckoning, then the PF is taken again
    GUARD_SPEED_CAP: float = 1.5             # [m/s] speed limit while the PF is being ignored
    GUARD_STALE_S: float = 0.2               # EKF older than this = not used (PF passes through)


# ============================================================
# SECTION 1: RACELINE LOAD + PRECOMPUTE (once, O(N))
# ============================================================

class Raceline:
    """Arrays as Python lists: per-element access in the loop is ~5x faster than numpy scalars."""

    def __init__(self, path: str, p: Params, log: Callable[[str], None]):
        data = np.loadtxt(os.path.expanduser(path), delimiter=',', comments='#', ndmin=2)
        x, y = data[:, 0], data[:, 1]
        seg_dx = np.roll(x, -1) - x                    # segment i: point i -> i+1 (closed loop)
        seg_dy = np.roll(y, -1) - y
        seg_len = np.hypot(seg_dx, seg_dy)
        s = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        psi = np.arctan2(seg_dy, seg_dx)

        # wall distances: raceline files (9 columns) or centerline.csv (x, y, w_left, w_right)
        w_left = np.full(len(x), p.TRACK_HALF_W)
        w_right = np.full(len(x), p.TRACK_HALF_W)
        if data.shape[1] >= 9:
            w_left, w_right = data[:, 7], data[:, 8]
        elif data.shape[1] == 4:
            w_left, w_right = data[:, 2], data[:, 3]

        if data.shape[1] >= 6 and not p.RACELINE_XY_ONLY:   # generator format: x,y,s,psi,kappa,vx,...
            kappa, vx = data[:, 4], data[:, 5]
            psi = data[:, 3]
            src = 'generator (kappa, vx from file)'
        else:                                          # x,y only (legacy file, or RACELINE_XY_ONLY): derive once
            dpsi = np.angle(np.exp(1j * (psi - np.roll(psi, 1))))
            # raw curvature = heading change / previous segment length (old node), then a
            # wrap-aware median filter: keeps corner plateaus sharp, removes single-point spikes
            kappa = dpsi / np.maximum(np.roll(seg_len, 1), 1e-6)
            w = int(p.LEGACY_KAPPA_MEDIAN)
            if w > 1 and len(kappa) > w:
                pad = w // 2
                ext = np.concatenate([kappa[-pad:], kappa, kappa[:pad]])
                kappa = np.median(np.lib.stride_tricks.sliding_window_view(ext, w), axis=1)
            vx = self._speed_profile(kappa, seg_len, p)
            src = 'legacy x,y (kappa, vx derived)'

        self.n = len(x)
        self.length = float(s[-1] + seg_len[-1])
        self.x, self.y = x.tolist(), y.tolist()
        self.s, self.psi = s.tolist(), psi.tolist()
        self.kappa, self.vx = kappa.tolist(), vx.tolist()
        self.seg_dx, self.seg_dy = seg_dx.tolist(), seg_dy.tolist()
        self.seg_len = seg_len.tolist()
        self.seg_inv_len2 = (1.0 / np.maximum(seg_len ** 2, 1e-12)).tolist()
        self.w_left, self.w_right = w_left.tolist(), w_right.tolist()   # wall distance from the line
        # numpy copies for the vectorised LiDAR -> Frenet step ([3a], once per scan)
        self.np_x, self.np_y, self.np_s = x.copy(), y.copy(), s.copy()
        self.np_cos, self.np_sin = np.cos(psi), np.sin(psi)
        self.np_wl, self.np_wr = np.asarray(w_left, float), np.asarray(w_right, float)
        # uniform spacing (generator output: 0.1 m +/- 0.0001) -> s to index is one division
        self.ds = float(np.mean(seg_len))
        self.inv_ds = 1.0 / self.ds
        self.uniform = float(np.max(np.abs(seg_len - self.ds))) < 0.01 * self.ds
        if not self.uniform:
            log('Raceline spacing is not uniform: s -> index falls back to a forward walk')
        log(f'Raceline {path}: {self.n} pts, {self.length:.1f} m, '
            f'max|kappa| {np.max(np.abs(kappa)):.2f}, vx {vx.min():.1f}-{vx.max():.1f} m/s, {src}')

    @staticmethod
    def _speed_profile(kappa, seg_len, p: Params) -> np.ndarray:
        """Friction-limited corner speed, then backward (braking) and forward (accel) passes."""
        v = np.clip(np.sqrt(p.LEGACY_A_Y / np.maximum(np.abs(kappa), 1e-4)), p.SPEED_MIN, p.SPEED_MAX)
        n = len(v)
        for _ in range(2):                              # two laps so the wrap-around settles
            for i in range(n - 1, -1, -1):
                j = (i + 1) % n
                v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * p.LEGACY_A_BRAKE * seg_len[i]))
            for i in range(n):
                j = (i - 1) % n
                v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * p.LEGACY_A_ACCEL * seg_len[j]))
        return v

    def index_at(self, s: float, hint: int = 0) -> int:
        """Segment index containing arc length s.
        Uniform spacing: O(1) (divide, then a +/-1 correction for rounding).
        Legacy non-uniform file: walk forward from hint (O(distance / ds))."""
        s = s % self.length
        n, S = self.n, self.s
        if self.uniform:
            i = min(int(s * self.inv_ds), n - 1)
        else:
            i = hint if S[hint] <= s else 0          # s wrapped past the finish line
        while S[i] > s:                                  # at most 1 step when uniform
            i -= 1
        while i + 1 < n and S[i + 1] <= s:
            i += 1
        return i

    def nearest_full(self, x: float, y: float, yaw: float, idxs=None) -> int:
        """Nearest point facing the car's heading (|diff| < 90 deg), so an opposite-direction
        return leg on the other side of a wall is never picked. idxs limits the search to an
        index window (forward along the CSV). O(N) or O(window); start-up and relocalization only."""
        n = self.n
        idxs = np.arange(n) if idxs is None else np.asarray(idxs) % n
        d2 = np.full(n, np.inf)
        d2[idxs] = (np.asarray(self.x)[idxs] - x) ** 2 + (np.asarray(self.y)[idxs] - y) ** 2
        facing = np.cos(yaw - np.asarray(self.psi)) > 0.0
        if np.any(facing[idxs]):
            d2[~facing] = np.inf
        k = min(5, int(np.isfinite(d2).sum()) - 1)
        cand = np.argpartition(d2, max(k, 0))[:max(k, 0) + 1]
        return int(min(cand, key=lambda i: d2[i] + 2.0 * abs(wrap(yaw - self.psi[i]))))


# ============================================================
# SECTION 1b: POSE GUARD (ROS-free; used by the node before stage [2] when GUARD_ENABLE)
# ============================================================

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


def ekf_at(buf, t: float, max_extrap: float):
    """[GUARD-SYNC] EKF pose at time t, so the guard compares the PF with the EKF at the SAME moment
    (a late EKF message during a fast turn otherwise looks like a heading disagreement).
    buf = deque of (stamp, x, y, yaw, vx, wz), oldest first.
      t inside the buffer : linear interpolation (heading wrap-aware)
      t newer than the last entry : moved forward with the last EKF speed and yaw rate, up to max_extrap [s]
      t older than the buffer : oldest entry
    Returns (x, y, yaw), or None if the buffer is empty or the EKF is older than max_extrap.
    O(k), k = entries newer than t (1-2 in practice)."""
    if not buf:
        return None
    s1, x1, y1, h1, v1, w1 = buf[-1]
    if t >= s1:
        dt = t - s1
        if dt > max_extrap:
            return None
        hm = h1 + 0.5 * w1 * dt                                   # mid-step heading for the position
        return x1 + v1 * dt * math.cos(hm), y1 + v1 * dt * math.sin(hm), wrap(h1 + w1 * dt)
    i = len(buf) - 1
    while i > 0 and buf[i - 1][0] > t:
        i -= 1
    if i == 0:
        return buf[0][1], buf[0][2], buf[0][3]
    s0, x0, y0, h0 = buf[i - 1][:4]
    s1, x1, y1, h1 = buf[i][:4]
    a = (t - s0) / max(s1 - s0, 1e-6)
    return x0 + a * (x1 - x0), y0 + a * (y1 - y0), wrap(h0 + a * wrap(h1 - h0))


# ============================================================
# SECTION 2: CONTROL CYCLE (stages [2]..[6], data-flow order)
# ============================================================

@dataclass
class CycleOut:
    """Everything one cycle decided. One instance, overwritten every cycle."""
    stop: bool = True
    # [2] transforms
    xr: float = 0.0                         # rear-axle position (RViz)
    yr: float = 0.0
    fx: float = 0.0                         # Frenet foot point: car projected on the raceline (RViz)
    fy: float = 0.0
    s: float = 0.0
    d: float = 0.0
    e_psi: float = 0.0
    kappa: float = 0.0
    # [4] reference
    lookahead_m: float = 0.0
    tx: float = 0.0                         # lookahead / preview point (for RViz)
    ty: float = 0.0
    v_ref: float = 0.0
    # [3] avoidance
    d_ref: float = 0.0
    speed_scale: float = 1.0                # x v_ref while an obstacle barrier is active
    speed_cap: float = math.inf
    avoid_active: bool = False
    n_obstacles: int = 0
    # [5] physics
    steer_raw: float = 0.0
    steer_cmd: float = 0.0
    react_scale: float = 1.0                # reaction scale used this cycle (RViz text; 1 = full)
    steer_saturated: bool = False
    ay_capped: bool = False
    # [6] speed + brake
    speed_cmd: float = 0.0
    speed_limited: bool = False
    brake_current: float = 0.0              # 0 = no brake command
    # stage timings [ms]
    t_frenet_ms: float = 0.0
    t_obstacles_ms: float = 0.0
    t_plan_ms: float = 0.0
    t_control_ms: float = 0.0


class Controller:
    """ROS-free control law: same code runs in the node, in SIM-1 and in offline tests."""

    def __init__(self, p: Params, log: Callable[[str], None] = print):
        self.p = p
        self.log = log
        self.line = Raceline(p.raceline_file, p, log)
        self.k_us = p.K_US if p.K_US >= 0.0 else (1.0 / p.C_SF - 1.0 / p.C_SR) / G
        self.use_pp = p.STEER_LAW.upper() == 'PURE_PURSUIT'
        self.seg = -1                        # current raceline segment; -1 = not localized yet
        self.prev_steer = 0.0
        self.warmup = 0
        self.guard_speed_cap = math.inf      # set by the node while the pose guard ignores the PF
        self.out = CycleOut()
        self._k_preview = 0.0                # curvature at the preview point, [4] -> [5]
        self._psi_s = 0.0                    # line heading at the car's s, [2] -> [4]
        self._k_path = 0.0                   # pure pursuit curvature if the car were on the line, [4] -> [5]
        # [3] avoidance state
        self.d_ref = 0.0                     # the barrier state: planned sideways offset from the raceline [m]
        self.d_ref_rate = 0.0
        self.side = 0                        # head-on latch: +1 pass left, -1 pass right, 0 none
        self.scan = None                     # latest scan (ranges, angle_min, angle_inc), set by the node
        self.scan_new = False
        self._beam_cache = None              # (key, beam index array, cos, sin) rebuilt only if scan geometry changes
        self.obstacles = []                  # confirmed: (s, d, r_s, r_d, x, y, psi, w_left, w_right) at the obstacle
        self._hits = 0
        self._misses = 0
        self._last_found = []                # [BIRTH] clusters of the previous scan
        self.obs_bt = []                     # B~ per obstacle this cycle (RViz colour)
        n_win = int(math.ceil((p.OBS_RANGE + p.LASER_X) / self.line.ds)) + 1
        self._win = np.arange(-int(math.ceil(p.OBS_BEHIND / self.line.ds)) - 2, n_win)
        log(f'Steer law {p.STEER_LAW}, K_US {self.k_us:.5f} rad/(m/s^2), '
            f'SPEED_SCALE {p.SPEED_SCALE}, SPEED_MAX {p.SPEED_MAX}, '
            + (f'reaction scale (REACT_V0 {p.REACT_V0} m/s, EXP {p.REACT_EXP}, MIN {p.REACT_MIN})'
               if p.REACT_V0 > 0.0 else 'reaction scale OFF'))

    def step(self, x: float, y: float, yaw: float, v: float, dt: float) -> CycleOut:
        """One cycle. dt = time since the previous pose [s] (pose-stamp difference)."""
        o = self.out
        if self.warmup < self.p.WARMUP_CYCLES:
            self.warmup += 1
            o.stop = True
            return o
        o.stop = False

        t0 = time.perf_counter()
        xr, yr = self.to_rear_axle(x, y, yaw)                           # [2]
        o.xr, o.yr = xr, yr
        self.to_frenet(xr, yr, yaw, o)                                  # [2]
        t1 = time.perf_counter()
        self.avoidance(xr, yr, yaw, v, dt, o)                           # [3]
        t2 = time.perf_counter()
        self.reference(v, o)                                            # [4]
        t3 = time.perf_counter()
        self.steering(xr, yr, yaw, v, dt, o)                            # [5]
        self.speed_and_brake(v, o)                                      # [6]
        t4 = time.perf_counter()

        # >>> TELEMETRY  t_frenet_ms, t_obstacles_ms, t_plan_ms, t_control_ms
        # >>> METRIC     compute budget per stage (rule: t_total < 0.5 x pose period), jitter
        o.t_frenet_ms = (t1 - t0) * 1e3
        o.t_obstacles_ms = (t2 - t1) * 1e3
        o.t_plan_ms = (t3 - t2) * 1e3
        o.t_control_ms = (t4 - t3) * 1e3
        return o

    # ---------------- [2] TRANSFORMS ----------------

    def to_rear_axle(self, x: float, y: float, yaw: float):
        """Pose reference point -> rear axle centre (pure pursuit geometry lives there)."""
        off = self.p.POSE_TO_REAR_AXLE_M
        if off == 0.0:
            return x, y
        return x - off * math.cos(yaw), y - off * math.sin(yaw)

    def to_frenet(self, x: float, y: float, yaw: float, o: CycleOut):
        """Rear-axle pose -> Frenet (s, d, e_psi, kappa).
        Point search once; afterwards projection onto the current segment + segment walk."""
        L = self.line
        if self.seg < 0:                                   # first cycle: the only point search
            self.seg = L.nearest_full(x, y, yaw)
            self.log(f'Localized on raceline: segment {self.seg}, s={L.s[self.seg]:.1f} m')

        seg, t, ok = self._walk(x, y, self.seg)
        d = self._signed_offset(x, y, seg, t)
        if not ok or abs(d) > self.p.RELOC_DIST_M:          # lost (spin, reset): search again
            # forward along the CSV first (next ~10 m), whole lap only if nothing facing the car is near
            fwd = np.arange(self.seg - 10, self.seg + int(10.0 / L.ds))
            seg = L.nearest_full(x, y, yaw, fwd)
            if (L.x[seg] - x) ** 2 + (L.y[seg] - y) ** 2 > self.p.RELOC_DIST_M ** 2:
                seg = L.nearest_full(x, y, yaw)
            self.log(f'Relocalized: segment {self.seg} -> {seg}')
            seg, t, _ = self._walk(x, y, seg)
            d = self._signed_offset(x, y, seg, t)
        self.seg = seg
        nxt = (seg + 1) % L.n
        psi_ref = L.psi[seg] + t * wrap(L.psi[nxt] - L.psi[seg])
        self._psi_s = psi_ref

        # >>> TELEMETRY  s, d, e_psi, kappa
        # >>> METRIC     lap time + lap count (s wrap-around on pose_stamp timeline),
        #                lateral spread (std of d per s-bin across laps), RMS d / e_psi,
        #                curvature sectors (kappa -> straight / medium / tight)
        o.s = L.s[seg] + t * L.seg_len[seg]
        o.d = d
        o.fx = L.x[seg] + t * L.seg_dx[seg]
        o.fy = L.y[seg] + t * L.seg_dy[seg]
        o.e_psi = wrap(yaw - psi_ref)
        o.kappa = L.kappa[seg] + t * (L.kappa[nxt] - L.kappa[seg])

    def _t_on(self, x: float, y: float, seg: int) -> float:
        """Projection parameter of (x, y) on segment seg: 0 = start point, 1 = end point."""
        L = self.line
        return ((x - L.x[seg]) * L.seg_dx[seg] + (y - L.y[seg]) * L.seg_dy[seg]) * L.seg_inv_len2[seg]

    def _walk(self, x: float, y: float, seg: int):
        """From seg, step forward while the projection is past the segment end (t > 1),
        or backward while it is before the start (t < 0). O(k), k = segments crossed.
        Inside a corner the projection can fall between two segments (t > 1 on one,
        t < 0 on the next): the walk then stops on the shared point (t clamped)."""
        n, lim = self.line.n, self.p.MAX_WALK_SEGS
        t = self._t_on(x, y, seg)
        k = 0
        while t > 1.0 and k < lim:
            seg = (seg + 1) % n
            t = self._t_on(x, y, seg)
            k += 1
        while t < 0.0 and k < lim:
            seg = (seg - 1) % n
            t = self._t_on(x, y, seg)
            k += 1
        return seg, (0.0 if t < 0.0 else (1.0 if t > 1.0 else t)), k < lim

    def _signed_offset(self, x: float, y: float, seg: int, t: float) -> float:
        """Lateral offset d from the projected point, + = left of the raceline."""
        L = self.line
        ex = x - (L.x[seg] + t * L.seg_dx[seg])
        ey = y - (L.y[seg] + t * L.seg_dy[seg])
        return (L.seg_dx[seg] * ey - L.seg_dy[seg] * ex) / L.seg_len[seg]

    # ---------------- [3] AVOIDANCE: BARRIER STATE PATH SHAPING ----------------

    def set_scan(self, ranges, angle_min: float, angle_inc: float):
        """Called by the node's scan callback: store only, processed in [3a] on the next cycle."""
        self.scan = (ranges, angle_min, angle_inc)
        self.scan_new = True

    def avoidance(self, xr: float, yr: float, yaw: float, v: float, dt: float, o: CycleOut):
        """[3a] obstacles -> [3b] barriers -> [3c] d_ref state. Disabled: exact raceline."""
        p = self.p
        if not p.AVOID_ENABLE:
            o.d_ref, o.speed_scale, o.speed_cap = 0.0, 1.0, math.inf
            o.avoid_active, o.n_obstacles = False, 0
            return
        if p.BAS_OBSTACLES and self.scan_new:
            self.detect_obstacles(xr, yr, yaw)                         # [3a]
            self.scan_new = False
        push, bt_ahead, active = self.barriers(o, v)                   # [3b]
        self.update_d_ref(push, bt_ahead, active, dt, o)               # [3c]

    # ---- [3a] LiDAR -> obstacles in Frenet (runs once per NEW scan) ----

    def detect_obstacles(self, xr: float, yr: float, yaw: float):
        """Beams -> map points -> (s, d) against the raceline window ahead -> drop wall
        points using the per-s wall distances -> cluster neighbouring beams -> obstacles.
        O(M*W) numpy, M = kept beams, W = window points (~OBS_RANGE / ds)."""
        p, L = self.p, self.line
        ranges, a0, da = self.scan
        r_all = np.asarray(ranges, dtype=float)
        key = (len(r_all), a0, da)
        if self._beam_cache is None or self._beam_cache[0] != key:   # beam angles: once per scan geometry
            ang = a0 + da * np.arange(len(r_all))
            idx = np.where(np.abs(ang) <= math.radians(p.SCAN_FOV_DEG) / 2)[0][::p.SCAN_STRIDE]
            self._beam_cache = (key, idx, np.cos(ang[idx]), np.sin(ang[idx]))
        _, idx, ca, sa = self._beam_cache
        r = r_all[idx]
        keep = np.isfinite(r) & (r > p.RANGE_MIN) & (r < p.OBS_RANGE)
        if not keep.any():
            self._confirm([])
            return
        r, ca, sa = r[keep], ca[keep], sa[keep]
        bi = idx[keep]                                         # [EDGE] beam index of each kept point

        # laser frame -> rear-axle frame -> map frame
        lx, ly = p.LASER_X + r * ca, p.LASER_Y + r * sa
        c, sn = math.cos(yaw), math.sin(yaw)
        px, py = xr + c * lx - sn * ly, yr + sn * lx + c * ly

        # nearest raceline point inside the window ahead of the car, then local projection
        win = (self.seg + self._win) % L.n
        d2 = (px[:, None] - L.np_x[win][None, :]) ** 2 + (py[:, None] - L.np_y[win][None, :]) ** 2
        k = win[np.argmin(d2, axis=1)]
        ex, ey = px - L.np_x[k], py - L.np_y[k]
        ps = L.np_s[k] + ex * L.np_cos[k] + ey * L.np_sin[k]
        pd = -ex * L.np_sin[k] + ey * L.np_cos[k]

        # inside the track and away from both walls -> not a wall return
        inside = (pd < L.np_wl[k] - p.WALL_GATE) & (pd > -L.np_wr[k] + p.WALL_GATE)
        if inside.sum() < p.MIN_PTS:
            self._confirm([])
            return
        px, py, ps, pd, rr, bi = px[inside], py[inside], ps[inside], pd[inside], r[inside], bi[inside]

        # clusters = runs of neighbouring beams (beams are in angle order)
        jump = np.hypot(np.diff(px), np.diff(py)) > p.CLUSTER_GAP
        cuts = np.concatenate([[0], np.where(jump)[0] + 1, [len(px)]])
        found = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < p.MIN_PTS:
                continue
            cs, cd = ps[a:b], pd[a:b]
            s_c = float(np.mean(cs)) % L.length
            d_c = float(np.mean(cd))
            r_s = max(p.OBS_R_MIN, 0.5 * float(cs.max() - cs.min()))
            r_d = max(p.OBS_R_MIN, 0.5 * float(cd.max() - cd.min()))
            i = L.index_at(s_c, self.seg)
            psi = L.psi[i]
            x_c = L.x[i] - d_c * math.sin(psi)                 # world centre, for RViz only
            y_c = L.y[i] + d_c * math.cos(psi)
            if p.OBS_BIRTH_RANGE > 0.0 and float(rr[a:b].min()) < p.OBS_BIRTH_RANGE \
                    and not self._continues(s_c, d_c):
                continue                                       # [BIRTH] appeared too close: not an obstacle
            if p.OBS_EDGE_JUMP > 0.0 and \
                    self._edge_jump(r_all, int(bi[a]), float(rr[a]), -1) + \
                    self._edge_jump(r_all, int(bi[b - 1]), float(rr[b - 1]), 1) < p.OBS_EDGE_SIDES:
                continue                                       # [EDGE] continuous with its background: wall piece
            found.append((s_c, d_c, r_s, r_d, x_c, y_c, psi, L.w_left[i], L.w_right[i]))
        self._last_found = found
        self._confirm(found)

    def _edge_jump(self, r_all, i: int, r_edge: float, step: int) -> int:
        """[EDGE] 1 if one of the next 3 valid beams (every SCAN_STRIDE) past the cluster edge reads farther than
        r_edge + OBS_EDGE_JUMP (no return / out of the scan counts as farther), else 0. Beams closer than
        RANGE_MIN (the car itself) are skipped. O(1)."""
        p, n = self.p, len(r_all)
        j, seen = i, 0
        while seen < 3:
            j += step * p.SCAN_STRIDE
            if j < 0 or j >= n:
                return 1
            rj = r_all[j]
            if not math.isfinite(rj) or rj - r_edge > p.OBS_EDGE_JUMP:
                return 1
            if rj <= p.RANGE_MIN:
                continue                                       # own body / no data: look one beam further
            seen += 1
        return 0

    def _continues(self, s_c: float, d_c: float) -> bool:
        """[BIRTH] True if (s_c, d_c) is an obstacle already known (confirmed, or seen in the previous scan).
        O(K), K = obstacles (a handful)."""
        p, half = self.p, 0.5 * self.line.length
        for ob in self.obstacles + self._last_found:
            ds = (s_c - ob[0] + half) % self.line.length - half
            if abs(ds) < p.OBS_MATCH_DS and abs(d_c - ob[1]) < p.OBS_MATCH_DD:
                return True
        return False

    def _confirm(self, found):
        """Persistence filter: trust obstacles after OBS_PERSIST scans in a row,
        keep the last ones for OBS_CLEAR empty scans. O(1)."""
        p = self.p
        if found:
            self._hits += 1
            self._misses = 0
            if self._hits >= p.OBS_PERSIST:
                self.obstacles = found
        else:
            self._misses += 1
            if self._misses >= p.OBS_CLEAR:
                self._hits = 0
                self.obstacles = []

    # ---- [3b] barrier functions: h (safety margin), B~ (shifted barrier), z_dot ----
    # The barriers are evaluated on the PLANNED lateral position d_ref (the barrier state),
    # not on the measured d. d_ref is where the car will be when it reaches the obstacle,
    # and it gives d_ref a real equilibrium (push falls as d_ref moves away), so it cannot
    # wind up while the car is still catching up with the shifted target.

    def barriers(self, o: CycleOut, v: float):
        """Returns (push [m/s], sum of B~ for obstacles ahead, any barrier active).
        push = sum over hazards of side * max(0, K_B * B~ + K_DB * z_dot), side = away (+1 left / -1 right)."""
        p, L = self.p, self.line
        q, q_dot = self.d_ref, self.d_ref_rate                      # planned offset and its rate
        s_dot = v * math.cos(o.e_psi) / max(1.0 - o.kappa * o.d, p.DENOM_MIN)   # progress along the line
        push, bt_ahead, active = 0.0, 0.0, False
        self.obs_bt = []

        if p.BAS_WALLS:
            # Two things must stay off the walls, each checked against the walls where it is:
            #   plan:  d_ref, at the target point s + lookahead (last cycle's lookahead)
            #   car:   the front bumper predicted T_LOOK_WALL ahead on its current heading,
            #          d + (CAR_FRONT + T_LOOK_WALL*v)*sin(e_psi), at s + CAR_FRONT + T_LOOK_WALL*v
            # For each wall the tighter of the two sets the barrier.  O(1) lookups.
            m = p.CAR_HALF_WIDTH + p.BAS_MARGIN
            i_t = L.index_at(o.s + o.lookahead_m, self.seg)
            reach = p.CAR_FRONT + p.T_LOOK_WALL * v                   # front bumper, T_LOOK_WALL ahead
            i_f = L.index_at(o.s + reach, self.seg)
            d_f = o.d + reach * math.sin(o.e_psi)
            d_f_dot = v * math.sin(o.e_psi)
            hl_plan, hl_car = L.w_left[i_t] - m - q, L.w_left[i_f] - m - d_f
            hr_plan, hr_car = q + L.w_right[i_t] - m, d_f + L.w_right[i_f] - m
            h_l, hl_dot = (hl_plan, -q_dot) if hl_plan <= hl_car else (hl_car, -d_f_dot)
            h_r, hr_dot = (hr_plan, q_dot) if hr_plan <= hr_car else (hr_car, d_f_dot)
            for h, h_dot, side in ((h_l, hl_dot, -1.0), (h_r, hr_dot, 1.0)):
                bt, zdot = self._shifted_barrier(h, h_dot, p.H_ACT_WALL)
                if bt > 0.0:
                    # never pull toward a hazard: a receding hazard (z_dot < 0) only weakens the push
                    push += side * max(0.0, p.K_B_WALL * bt + p.K_DB_WALL * zdot)
                    active = True

        any_near = False                                              # [BAS-FIX] latch kept while an obstacle is in range
        for (s_j, d_j, r_s, r_d, _, _, _, wl_j, wr_j) in self.obstacles:
            ds = (o.s - s_j + 0.5 * L.length) % L.length - 0.5 * L.length   # < 0: obstacle ahead
            if ds < -p.OBS_RANGE or ds > p.OBS_BEHIND:
                self.obs_bt.append(0.0)
                continue
            any_near = True
            # ellipse around the obstacle, inflated by the car footprint; stretched ahead by T_LOOK*v
            a = r_s + p.BAS_MARGIN + (p.CAR_FRONT + p.T_LOOK * v if ds < 0.0 else p.CAR_REAR)
            b = r_d + p.CAR_HALF_WIDTH + p.BAS_MARGIN
            gs, gd = 2.0 * ds / (a * a), 2.0 * (q - d_j) / (b * b)    # gradient of h in (s, d)
            h = (ds / a) ** 2 + ((q - d_j) / b) ** 2 - 1.0            # < 0 = footprints overlap
            bt, zdot = self._shifted_barrier(h, gs * s_dot + gd * q_dot, p.H_ACT_OBS)
            self.obs_bt.append(bt)
            if bt <= 0.0:
                continue
            side = self._pass_side(h, d_j, r_d, wl_j, wr_j)            # +1 pass left, -1 pass right
            push += side * max(0.0, p.K_B_OBS * bt + p.K_DB_OBS * zdot)
            active = True
            if ds < 0.0:
                bt_ahead += bt
        if not any_near:
            # [BAS-FIX] release the pass-side latch only when no obstacle is in range. Releasing it whenever
            # the barrier dropped for one cycle (obstacle estimate jump beside the car) let the side flip
            # and steer the car back into the obstacle.
            self.side = 0
        return push, bt_ahead, active

    def _shifted_barrier(self, h: float, h_dot: float, h_act: float):
        """B~ = 1/h - 1/h_act (exactly 0 when h >= h_act), capped at B_MAX.
        z_dot = dB/dt = -h_dot / h^2, only on the smooth part (not capped)."""
        p = self.p
        if h >= h_act:
            return 0.0, 0.0
        hc = max(h, p.H_MIN)
        bt = min(1.0 / hc - 1.0 / h_act, p.B_MAX)
        if bt >= p.B_MAX or h <= p.H_MIN:
            return bt, 0.0
        zdot = -h_dot / (hc * hc)
        return bt, max(-p.ZDOT_MAX, min(p.ZDOT_MAX, zdot))

    def _pass_side(self, h: float, d_j: float, r_d: float, wl_j: float, wr_j: float) -> int:
        """Which side to pass on: the bigger gap between the obstacle and the wall, measured
        at the obstacle. Decided when the barrier first switches on and latched; it may only
        switch (by more than GAP_HYST) while the obstacle is still far (h > H_SWITCH), so the
        car never cuts across the obstacle's nose at close range. Returns +1 left / -1 right."""
        p = self.p
        gap_l = wl_j - (d_j + r_d)
        gap_r = (d_j - r_d) + wr_j
        if self.side == 0:
            self.side = 1 if gap_l >= gap_r else -1
        elif h > p.H_SWITCH:
            if self.side == 1 and gap_r > gap_l + p.GAP_HYST:
                self.side = -1
            elif self.side == -1 and gap_l > gap_r + p.GAP_HYST:
                self.side = 1
        return self.side

    # ---- [3c] barrier state -> d_ref (the sideways target shift) ----

    def update_d_ref(self, push: float, bt_ahead: float, active: bool, dt: float, o: CycleOut):
        """d_ref_dot = push - K_RELAX * d_ref, rate-limited, kept inside the walls.
        Barriers push the target away from hazards; K_RELAX brings it back to the raceline."""
        p, L = self.p, self.line
        rate = push - p.K_RELAX * self.d_ref
        rate = max(-p.DREF_RATE_MAX, min(p.DREF_RATE_MAX, rate))
        d_ref = self.d_ref + rate * dt
        self.d_ref_rate = rate                                        # feeds z_dot next cycle
        hi = L.w_left[self.seg] - p.CAR_HALF_WIDTH - p.BAS_MARGIN
        lo = -L.w_right[self.seg] + p.CAR_HALF_WIDTH + p.BAS_MARGIN
        d_ref = min(hi, max(lo, d_ref)) if lo < hi else 0.0
        if not active and abs(d_ref) < p.DREF_EPS:
            d_ref = 0.0                                               # exact raceline again
            self.d_ref_rate = 0.0
        self.d_ref = d_ref

        # >>> TELEMETRY  d_ref, avoid_active, n_obstacles (+ t_obstacles_ms, scan_stamp)
        # >>> METRIC     avoidance events per lap, time off-line (d_ref != 0), max |d_ref|,
        #                min clearance to obstacles (d - d_ref vs obstacle d), wall-barrier
        #                activations per sector (tracking drift toward walls)
        o.d_ref = d_ref
        # slow down for obstacles ahead, and stay slower while the car is still off the raceline
        # (the raceline speed profile is only valid on the raceline)
        o.speed_scale = 1.0 / (1.0 + p.K_BV * bt_ahead + p.K_BV_RETURN * abs(d_ref))
        o.speed_cap = math.inf
        o.avoid_active = active
        o.n_obstacles = len(self.obstacles)

    # ---------------- [4] REFERENCE ----------------

    def reference(self, v: float, o: CycleOut):
        """Lookahead point (PP) or curvature preview point (PD), and the speed reference."""
        p, L = self.p, self.line
        if self.use_pp:
            la = min(p.LA_MAX, max(p.LA_MIN, p.LA_TIME * v))
        else:
            la = max(p.FF_PREVIEW_MIN, p.FF_PREVIEW_T * v)
        s_t = o.s + la
        i = L.index_at(s_t, self.seg)
        t = ((s_t % L.length) - L.s[i]) / L.seg_len[i]
        nxt = (i + 1) % L.n
        psi_t = L.psi[i] + t * wrap(L.psi[nxt] - L.psi[i])
        # target point, shifted sideways by d_ref (0 = raceline), kept inside the walls at s_t
        d_t = o.d_ref
        if d_t != 0.0:
            m = p.CAR_HALF_WIDTH + p.BAS_MARGIN
            d_t = min(L.w_left[i] - m, max(-L.w_right[i] + m, d_t))
        o.tx = L.x[i] + t * L.seg_dx[i] - d_t * math.sin(psi_t)
        o.ty = L.y[i] + t * L.seg_dy[i] + d_t * math.cos(psi_t)
        self._k_preview = L.kappa[i] + t * (L.kappa[nxt] - L.kappa[i])
        # path part for the reaction split: the curvature pure pursuit would command if the car sat
        # exactly on the line (at the foot point, with the line's heading), aiming at the line point
        # at s + la. Any extra curvature is correction toward the line. O(1).
        lx, ly = L.x[i] + t * L.seg_dx[i] - o.fx, L.y[i] + t * L.seg_dy[i] - o.fy
        c, sn = math.cos(self._psi_s), math.sin(self._psi_s)
        xl, yl = c * lx + sn * ly, -sn * lx + c * ly
        self._k_path = 2.0 * yl / max(xl * xl + yl * yl, 1e-6)

        # speed reference: raceline vx a little ahead (drive lag), scaled
        j = L.index_at(o.s + v * p.SPEED_PREVIEW_T, self.seg)
        # >>> TELEMETRY  lookahead_m, v_ref
        # >>> METRIC     speed tracking error (v - v_ref) per sector, lookahead vs speed
        o.lookahead_m = la
        o.v_ref = L.vx[j] * p.SPEED_SCALE

    # ---------------- [5] PHYSICS: STEERING ----------------

    def steering(self, xr: float, yr: float, yaw: float, v: float, dt: float, o: CycleOut):
        """Steering law -> understeer correction -> lateral-accel cap -> rate limit -> clamp."""
        p = self.p
        # reaction scale: 1 = full correction; below 1 = softer correction (path part untouched)
        if p.REACT_V0 > 0.0:
            react = (p.REACT_V0 / max(v, 0.1)) ** p.REACT_EXP
            react = min(1.0, max(p.REACT_MIN, react))
        else:
            react = 1.0
        o.react_scale = react
        if self.use_pp:
            # target in the rear-axle frame; path curvature through it = 2*y_l / Ld^2
            dx, dy = o.tx - xr, o.ty - yr
            c, sn = math.cos(yaw), math.sin(yaw)
            x_l = c * dx + sn * dy
            y_l = -sn * dx + c * dy
            k_cmd = 2.0 * y_l / max(x_l * x_l + y_l * y_l, 1e-6)
            if react < 1.0:
                # split: line's own curvature (kept) + correction toward the line (scaled)
                k_cmd = self._k_path + react * (k_cmd - self._k_path)
            delta = math.atan(p.WHEELBASE * k_cmd) + self.k_us * v * v * k_cmd
        else:
            k_ff = self._k_preview
            v_eff = max(v, p.V_MIN_SCHED)
            kp = p.KP * (p.V_REF / v_eff) ** 2
            kd = p.KD * (p.V_REF / v_eff)
            delta = (math.atan(p.WHEELBASE * k_ff) + self.k_us * v * v * k_ff
                     - react * (kp * (o.d - o.d_ref) + kd * o.e_psi))
        raw = delta

        # lateral-accel cap: never ask for more than A_Y_MAX at this speed
        o.ay_capped = False
        if p.A_Y_MAX > 0.0 and v > 0.5:
            cap = math.atan(p.WHEELBASE * p.A_Y_MAX / (v * v)) + self.k_us * p.A_Y_MAX
            if abs(delta) > cap:
                delta = math.copysign(cap, delta)
                o.ay_capped = True

        # servo slew limit in rad/s, then mechanical limit
        step = p.STEER_STEP_MAX if p.STEER_STEP_MAX > 0.0 else p.STEER_RATE_MAX * dt
        delta = min(self.prev_steer + step, max(self.prev_steer - step, delta))
        delta = min(p.STEER_LIMIT, max(-p.STEER_LIMIT, delta))
        self.prev_steer = delta

        # >>> TELEMETRY  steer_raw, steer_cmd, steer_saturated
        # >>> METRIC     % of cycles saturated (rate / limit / a_y cap), steering smoothness
        #                (std of d(steer)/dt), understeer check (steer_cmd vs L*kappa)
        o.steer_raw = raw
        o.steer_cmd = delta
        o.steer_saturated = abs(delta - raw) > 1e-6

    # ---------------- [6] SPEED + BRAKE ----------------

    def speed_and_brake(self, v: float, o: CycleOut):
        p = self.p
        target = min(o.v_ref * o.speed_scale, o.speed_cap, p.SPEED_MAX, self.guard_speed_cap)
        target = max(p.V_BAS_MIN if o.avoid_active else p.SPEED_MIN, target)
        o.speed_limited = target < o.v_ref - 1e-6 or o.ay_capped
        o.brake_current = 0.0
        if p.BRAKE_ENABLE:
            over = v - target
            if over > p.BRAKE_THRESHOLD:
                frac = min(over / 3.0, 1.0)
                o.brake_current = p.BRAKE_CURRENT_MIN + frac * (p.BRAKE_CURRENT_MAX - p.BRAKE_CURRENT_MIN)
        # >>> TELEMETRY  speed_cmd, FLAG_SPEED_LIMITED
        # >>> METRIC     time at speed cap, braking events per lap
        o.speed_cmd = target


# ============================================================
# SECTION 3: ROS NODE ([1] input -> SECTION 2 -> [7] output -> [8] telemetry)
# ============================================================

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import Point, PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import ColorRGBA, Float64
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:                 # SECTION 0-2 stay importable without ROS (SIM-1, tests)
    rclpy = None

try:
    from race_msgs.msg import Telemetry
except ImportError:
    Telemetry = None


def stamp_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


if rclpy is not None:

    class BaselineNode(Node):
        MAX_VIZ_OBS = 6

        def __init__(self):
            super().__init__('baseline_pp')
            self.p = self._declare_params()
            self.declare_parameter('pose_topic', '/ego_racecar/odom')   # car: /pf/pose/odom
            self.declare_parameter('odom_topic', '')                    # car: /odom ; '' = twist from pose msg
            self.declare_parameter('TELEMETRY_ENABLE', True)
            self.declare_parameter('VIZ_ENABLE', False)                 # sim: true ; race: false
            self.declare_parameter('VIZ_RATE_HZ', 20.0)                 # live markers, throttled
            self.declare_parameter('STALE_POSE_MS', 60.0)
            self.declare_parameter('STALE_ODOM_MS', 60.0)
            self.declare_parameter('STALE_SCAN_MS', 100.0)
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('guard_odom_topic', '/odometry/filtered')   # EKF for the pose guard
            gp = lambda n: self.get_parameter(n).value
            self.telemetry_on = bool(gp('TELEMETRY_ENABLE')) and Telemetry is not None
            self.viz_on = bool(gp('VIZ_ENABLE'))
            self.viz_period = 1.0 / max(1.0, float(gp('VIZ_RATE_HZ')))
            self.next_viz = 0.0
            self.stale_pose_s = float(gp('STALE_POSE_MS')) * 1e-3
            self.stale_odom_s = float(gp('STALE_ODOM_MS')) * 1e-3
            self.stale_scan_s = float(gp('STALE_SCAN_MS')) * 1e-3
            self.scan_stamp = None

            self.ctrl = Controller(self.p, log=self.get_logger().info)

            # state written by [1]
            self.v = 0.0
            self.yaw_rate = 0.0
            self.odom_stamp = None
            self.prev_pose_t = None
            self.cycle = 0
            # pose guard state
            p = self.p
            self.guard = PoseGuard(head_gate=p.GUARD_HEAD_GATE, lat_gate=p.GUARD_LAT_GATE, long_gate=p.GUARD_LONG_GATE,
                                   tau_pos=p.GUARD_TAU_POS, tau_head=p.GUARD_TAU_HEAD, steady_s=p.GUARD_STEADY_S,
                                   steady_pos=p.GUARD_STEADY_POS, steady_head=p.GUARD_STEADY_HEAD,
                                   steady_travel=p.GUARD_STEADY_TRAVEL, max_reject_s=p.GUARD_MAX_REJECT_S,
                                   guard_position=p.GUARD_POSITION, laser_x=p.LASER_X)
            self.guard_buf = deque(maxlen=50)    # [GUARD-SYNC] (stamp, x, y, yaw, vx, wz) EKF poses, last ~1 s
            self.guard_state = 'off'

            # [7] OUTPUT publishers
            self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
            self.brake_pub = self.create_publisher(Float64, '/commands/motor/brake', 1)
            # [8] TELEMETRY publisher (depth 50: recorder may lag briefly, nothing is dropped)
            self.telem_pub = self.create_publisher(Telemetry, '/telemetry', 50) if self.telemetry_on else None
            if bool(gp('TELEMETRY_ENABLE')) and Telemetry is None:
                self.get_logger().warn('race_msgs not built: telemetry OFF')
            # RViz (not part of the control loop; see the RViz section at the end)
            if self.viz_on:
                latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                self.lines_pub = self.create_publisher(MarkerArray, '/baseline/racelines', latched)
                self.lines_pub.publish(self._raceline_markers())      # once; latched for late RViz
                self.targets_pub = self.create_publisher(MarkerArray, '/baseline/targets', 1)
                self.targets = self._make_target_markers()           # built once, updated in place

            # [1] POSE INPUT subscriptions (depth 1: always act on the newest pose)
            # [QOS] sensor streams best effort: a reliable depth-1 reader held scans back ~0.5 s on the car
            # (the node used 205 of 2177 scans in bas_run0). Best effort always delivers the newest message.
            sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                    history=HistoryPolicy.KEEP_LAST)
            odom_topic = gp('odom_topic')
            if odom_topic:
                self.create_subscription(Odometry, odom_topic, self.odom_callback, sensor_qos)
            self.create_subscription(Odometry, gp('pose_topic'), self.pose_callback, sensor_qos)
            if self.p.AVOID_ENABLE and self.p.BAS_OBSTACLES:          # scans only needed for obstacles
                self.create_subscription(LaserScan, gp('scan_topic'), self.scan_callback, sensor_qos)
            self.use_pose_twist = not odom_topic
            if self.p.GUARD_ENABLE:
                ekf_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                     history=HistoryPolicy.KEEP_LAST)       # [QOS] keep every EKF pose for the buffer
                self.create_subscription(Odometry, gp('guard_odom_topic'), self.guard_odom_callback, ekf_qos)
                self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                         lambda m: self.guard.reset(), QoSProfile(depth=1))
                self.get_logger().info(f'Pose guard ON: EKF {gp("guard_odom_topic")}, heading gate '
                                       f'{math.degrees(self.p.GUARD_HEAD_GATE):.0f} deg, speed cap '
                                       f'{self.p.GUARD_SPEED_CAP} m/s while the PF is ignored')

            # params dump -> run registry (steer law, gains, raceline are not in Telemetry)
            self.get_logger().info('PARAMS ' + ', '.join(f'{f.name}={getattr(self.p, f.name)}'
                                                         for f in fields(Params)))

        def _declare_params(self) -> Params:
            p = Params()
            for f in fields(Params):
                default = getattr(p, f.name)
                self.declare_parameter(f.name, default)
                setattr(p, f.name, type(default)(self.get_parameter(f.name).value))
            return p

        # ---------------- [1] POSE INPUT ----------------

        def odom_callback(self, msg):
            """Velocity source when pose comes from the particle filter (car)."""
            self.v = msg.twist.twist.linear.x
            self.yaw_rate = msg.twist.twist.angular.z
            self.odom_stamp = msg.header.stamp

        def guard_odom_callback(self, msg):
            """EKF pose for the pose guard (odom frame), stored with its stamp. O(1)."""
            q = msg.pose.pose.orientation
            self.guard_buf.append((stamp_s(msg.header.stamp), msg.pose.pose.position.x, msg.pose.pose.position.y,
                                   math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
                                   msg.twist.twist.linear.x, msg.twist.twist.angular.z))

        def scan_callback(self, msg):
            """Store only (O(M) copy); obstacles are extracted in stage [3a] of the next cycle."""
            self.ctrl.set_scan(np.asarray(msg.ranges, dtype=np.float32), msg.angle_min, msg.angle_increment)
            self.scan_stamp = msg.header.stamp

        def pose_callback(self, msg):
            """Master timeline: one pose message = one control cycle."""
            t_start = time.perf_counter()
            pp = msg.pose.pose
            x, y = pp.position.x, pp.position.y
            q = pp.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            if self.use_pose_twist:
                self.v = msg.twist.twist.linear.x
                self.yaw_rate = msg.twist.twist.angular.z
                self.odom_stamp = msg.header.stamp

            # dt for the steering slew limit from the node's own clock: particle-filter stamps can
            # repeat or jump, which would freeze (tiny dt) or unlock (big dt) the limit
            t_now = time.monotonic()
            dt = 0.02 if self.prev_pose_t is None else t_now - self.prev_pose_t
            dt = min(0.05, max(0.005, dt))
            self.prev_pose_t = t_now

            # ---- pose guard: PF heading checked against the EKF motion (O(1)) ----
            if self.p.GUARD_ENABLE:
                # [GUARD-SYNC] EKF pose at the PF pose's own stamp (None if the EKF is older than GUARD_STALE_S)
                t_pf = stamp_s(msg.header.stamp)
                odom = ekf_at(self.guard_buf, t_pf, self.p.GUARD_STALE_S)
                (x, y, yaw), state = self.guard.update(t_pf, (x, y, yaw), odom)
                if state != self.guard_state and state in ('reject', 'resync'):
                    if state == 'reject':
                        self.get_logger().warn('Pose guard: PF heading disagrees with the EKF -> ignoring PF, '
                                               f'speed capped at {self.p.GUARD_SPEED_CAP} m/s')
                    else:
                        self.get_logger().warn('Pose guard: PF accepted again')
                self.guard_state = state
                self.ctrl.guard_speed_cap = self.p.GUARD_SPEED_CAP if state == 'reject' else math.inf

            # ---- stages [2]..[6] ----
            o = self.ctrl.step(x, y, yaw, self.v, dt)

            # ---- [7] OUTPUT ----
            now = self.get_clock().now()
            drive = AckermannDriveStamped()
            drive.header.stamp = now.to_msg()
            drive.header.frame_id = 'base_link'
            if not o.stop:
                drive.drive.steering_angle = o.steer_cmd
                drive.drive.speed = o.speed_cmd
            self.drive_pub.publish(drive)
            if o.brake_current > 0.0:
                self.brake_pub.publish(Float64(data=o.brake_current))
            if self.viz_on and not o.stop:
                t_wall = time.perf_counter()
                if t_wall >= self.next_viz:                          # throttle to VIZ_RATE_HZ
                    self.next_viz = t_wall + self.viz_period
                    self._publish_targets(o)

            # ---- [8] TELEMETRY ----
            if self.telemetry_on and not o.stop:
                self._publish_telemetry(msg, now, x, y, yaw, o, t_start)

        # ---------------- [8] TELEMETRY ----------------

        def _publish_telemetry(self, pose_msg, now, x, y, yaw, o: CycleOut, t_start: float):
            """Record only. Every metric is computed post-run from /telemetry (manual trigger)."""
            m = Telemetry()
            # >>> TELEMETRY  header.stamp (publish time), pose_stamp, odom_stamp, cycle
            # >>> METRIC     time sync key (pose_stamp), pose->command latency
            #                (header.stamp - pose_stamp), dropped cycles (gaps in cycle)
            m.header.stamp = now.to_msg()
            m.header.frame_id = 'map'
            m.pose_stamp = pose_msg.header.stamp
            if self.odom_stamp is not None:
                m.odom_stamp = self.odom_stamp
            if self.scan_stamp is not None:
                m.scan_stamp = self.scan_stamp
            m.cycle = self.cycle
            self.cycle += 1
            m.controller_id = Telemetry.CONTROLLER_PP

            # >>> TELEMETRY  x, y, yaw, v, yaw_rate
            # >>> METRIC     driven path overlay, max speed, yaw-rate vs v*kappa (slip / understeer)
            m.x, m.y, m.yaw = float(x), float(y), float(yaw)
            m.v, m.yaw_rate = float(self.v), float(self.yaw_rate)

            # stage [2]..[6] fields (see the METRIC tags at each stage)
            m.s, m.d, m.e_psi, m.kappa = o.s, o.d, o.e_psi, o.kappa
            m.d_ref, m.v_ref, m.lookahead_m = o.d_ref, o.v_ref, o.lookahead_m
            m.n_obstacles, m.avoid_active = o.n_obstacles, o.avoid_active
            m.steer_raw, m.steer_cmd, m.speed_cmd = o.steer_raw, o.steer_cmd, o.speed_cmd
            m.steer_saturated = o.steer_saturated
            m.t_frenet_ms, m.t_obstacles_ms = o.t_frenet_ms, o.t_obstacles_ms
            m.t_plan_ms, m.t_control_ms = o.t_plan_ms, o.t_control_ms
            m.t_total_ms = (time.perf_counter() - t_start) * 1e3

            # >>> TELEMETRY  flags (STALE_POSE, STALE_ODOM, STALE_SCAN, SPEED_LIMITED)
            # >>> METRIC     % of cycles on stale data, % of cycles speed-limited
            now_s = now.nanoseconds * 1e-9
            flags = 0
            if now_s - stamp_s(pose_msg.header.stamp) > self.stale_pose_s:
                flags |= Telemetry.FLAG_STALE_POSE
            if self.odom_stamp is None or now_s - stamp_s(self.odom_stamp) > self.stale_odom_s:
                flags |= Telemetry.FLAG_STALE_ODOM
            if o.speed_limited:
                flags |= Telemetry.FLAG_SPEED_LIMITED
            if self.p.AVOID_ENABLE and self.p.BAS_OBSTACLES and (
                    self.scan_stamp is None or now_s - stamp_s(self.scan_stamp) > self.stale_scan_s):
                flags |= Telemetry.FLAG_STALE_SCAN
            m.flags = flags
            self.telem_pub.publish(m)

        # ---------------- RViz (VIZ_ENABLE; not part of the race loop) ----------------
        #   /baseline/racelines (latched, once): active raceline coloured by speed
        #       (blue = slow, red = fast), the other lines in the same folder thin,
        #       start/finish cube at s = 0
        #   /baseline/targets (VIZ_RATE_HZ): lookahead / preview target, Frenet foot
        #       point, rear axle -> target line, speed text above the car

        def _raceline_markers(self) -> MarkerArray:
            L = self.ctrl.line
            arr = MarkerArray()
            vmin, vmax = min(L.vx), max(L.vx)
            span = max(vmax - vmin, 1e-6)

            active = self._line_marker(0, 'active', L.x, L.y, 0.06, (1.0, 1.0, 1.0, 1.0))
            for i in list(range(L.n)) + [0]:
                c = (L.vx[i] - vmin) / span
                active.colors.append(ColorRGBA(r=c, g=1.0 - abs(2.0 * c - 1.0), b=1.0 - c, a=1.0))
            arr.markers.append(active)

            # every other line the generator wrote next to the active one
            folder = os.path.dirname(os.path.expanduser(self.p.raceline_file))
            me = os.path.abspath(os.path.expanduser(self.p.raceline_file))
            others = {'centerline': (0.6, 0.6, 0.6, 0.6), 'raceline_stable': (0.2, 0.9, 0.2, 0.6),
                      'raceline_optimal': (1.0, 0.85, 0.1, 0.6), 'raceline_fast': (1.0, 0.45, 0.0, 0.6)}
            mid = 1
            for path in sorted(glob.glob(os.path.join(folder, '*.csv'))):
                name = os.path.splitext(os.path.basename(path))[0]
                if os.path.abspath(path) == me or name not in others:
                    continue
                xy = np.loadtxt(path, delimiter=',', comments='#', usecols=(0, 1), ndmin=2)
                arr.markers.append(self._line_marker(mid, name, xy[:, 0].tolist(), xy[:, 1].tolist(),
                                                     0.02, others[name]))
                mid += 1

            start = Marker()
            start.header.frame_id = 'map'
            start.ns, start.id, start.type, start.action = 'start_finish', 0, Marker.CUBE, Marker.ADD
            start.pose.position.x, start.pose.position.y = L.x[0], L.y[0]
            start.pose.orientation.z = math.sin(0.5 * L.psi[0])
            start.pose.orientation.w = math.cos(0.5 * L.psi[0])
            start.scale.x, start.scale.y, start.scale.z = 0.05, 1.0, 0.05
            start.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            arr.markers.append(start)
            return arr

        @staticmethod
        def _line_marker(mid, ns, xs, ys, width, rgba) -> Marker:
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = ns, mid, Marker.LINE_STRIP, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = width
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            n = len(xs)
            m.points = [Point(x=float(xs[i]), y=float(ys[i]), z=0.02) for i in list(range(n)) + [0]]
            return m

        def _make_target_markers(self) -> MarkerArray:
            def mk(mid, mtype, size, rgba):
                m = Marker()
                m.header.frame_id = 'map'
                m.ns, m.id, m.type, m.action = 'baseline_targets', mid, mtype, Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = size
                m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
                return m
            target = mk(0, Marker.SPHERE, 0.20, (1.0, 0.0, 1.0, 1.0))       # magenta: lookahead / preview
            foot = mk(1, Marker.SPHERE, 0.12, (0.0, 1.0, 1.0, 1.0))         # cyan: Frenet foot point
            ray = mk(2, Marker.LINE_LIST, 0.02, (1.0, 1.0, 1.0, 0.8))       # rear axle -> target
            ray.points = [Point(), Point()]
            text = mk(3, Marker.TEXT_VIEW_FACING, 0.22, (1.0, 1.0, 1.0, 1.0))
            obs = [mk(10 + k, Marker.CYLINDER, 0.3, (1.0, 0.0, 0.0, 0.0)) for k in range(self.MAX_VIZ_OBS)]
            return MarkerArray(markers=[target, foot, ray, text] + obs)

        def _publish_targets(self, o: CycleOut):
            target, foot, ray, text = self.targets.markers[:4]
            target.pose.position.x, target.pose.position.y, target.pose.position.z = o.tx, o.ty, 0.1
            foot.pose.position.x, foot.pose.position.y, foot.pose.position.z = o.fx, o.fy, 0.1
            ray.points[0].x, ray.points[0].y = o.xr, o.yr
            ray.points[1].x, ray.points[1].y = o.tx, o.ty
            text.pose.position.x, text.pose.position.y, text.pose.position.z = o.xr, o.yr, 0.6
            text.text = (f'v {self.v:.2f} | cmd {o.speed_cmd:.2f} | ref {o.v_ref:.2f} m/s\n'
                         f'd {o.d:+.2f} m  la {o.lookahead_m:.2f} m  s {o.s:.1f}  react {o.react_scale:.2f}'
                         + (f'  guard {self.guard_state}' if self.p.GUARD_ENABLE else ''))
            if self.p.AVOID_ENABLE:
                text.text += f'\nd_ref {o.d_ref:+.2f} m  obs {o.n_obstacles}  {"AVOID" if o.avoid_active else ""}'
            # obstacles: footprint ellipse, red = barrier active (alpha grows with B~), grey = inactive
            obs_list, bts = self.ctrl.obstacles, self.ctrl.obs_bt
            for k, m in enumerate(self.targets.markers[4:]):
                if k < len(obs_list):
                    _, _, r_s, r_d, x_c, y_c, psi, _, _ = obs_list[k]
                    bt = bts[k] if k < len(bts) else 0.0
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = x_c, y_c, 0.1
                    m.pose.orientation.z, m.pose.orientation.w = math.sin(0.5 * psi), math.cos(0.5 * psi)
                    m.scale.x, m.scale.y, m.scale.z = 2.0 * r_s, 2.0 * r_d, 0.2
                    if bt > 0.0:
                        m.color.r, m.color.g, m.color.b = 1.0, 0.1, 0.1
                        m.color.a = min(1.0, 0.4 + 0.1 * bt)
                    else:
                        m.color.r = m.color.g = m.color.b = 0.6
                        m.color.a = 0.5
                else:
                    m.color.a = 0.0                                  # unused slot: invisible
            self.targets_pub.publish(self.targets)


def main(args=None):
    rclpy.init(args=args)
    node = BaselineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.drive_pub.publish(AckermannDriveStamped())    # zero command on exit
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()