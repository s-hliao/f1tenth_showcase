#!/usr/bin/env python3
"""
Baseline v3 (bv3): Frenet-frame controller with an LQR optimizer. Single file for the current car setup.
Tartan Racing (CMU F1TENTH / Roboracer).

One control cycle per pose message, stages in data-flow order. Each stage is one class with a fixed
input and output, so it can later move into its own package unchanged:

    [0] PARAMETERS     one group per stage, all loaded from the YAML (track block first)
    [1] SCAN PREP      /scan -> crop + downsample (pass-through slot; nothing downstream uses it yet)
    [2] POSE -> STATE  pose message -> State (s, d, e_psi, kappa, v) on the active path
    [3] AVOIDANCE      scan + State -> d_ref, speed scale (barrier states, BaS). avoid.ENABLE: false = bv3 unchanged
    [4] PLANNER        path file (+ [3]) -> Reference (d_ref, curvature preview, speed target)
    [5] OPTIMIZER      State + Reference -> Command; controller chosen by name:
                          LQR           feedforward - K(v)*[d - d_ref, e_psi], K from a single-track model
                          PURE_PURSUIT  bv2 race law (bit-identical to bv2)
                          PD            bv2 Frenet PD law (bit-identical to bv2)
    [6] NODE           ROS inputs/outputs, telemetry, RViz

SINGLE-TRACK (BICYCLE) MODEL: injected in two places, both in [5], both marked >>> SINGLE-TRACK MODEL:
    (a) LqrOptimizer._error_model()  linearized error dynamics -> A, B -> Riccati -> gain table K(v)
    (b) LqrOptimizer._predict()      nonlinear error dynamics rolled forward to cover the actuation delay
    Kinematic model only: no tyre forces, no slip (the tyre model comes later). The one non-kinematic number
    is K_US (understeer gradient), the same value the feedforward already uses; (b) needs it, see _predict.

Run:
    python3 bv3_node.py --ros-args --params-file bv3_car.yaml
    python3 bv3_node.py --ros-args --params-file bv3_sim.yaml
Change track or raceline: edit the `track:` block of the YAML only.

Cost per cycle: O(k) for the path walk (k = segments crossed, 1-2), O(1) for everything else (+ DELAY_S/DT short
    model steps when the delay predictor is on). The LQR gain table is computed once at start-up.
"""
import collections
import math
import os
import time
from dataclasses import dataclass, fields

import numpy as np


# ============================================================
# [0] PARAMETERS (defaults; the YAML overrides every one of them)
# ============================================================

@dataclass
class TrackParams:
    """The ONLY place a map or a raceline is named. A track folder holds both."""
    ROOT: str = '~/f1tenth_ws/src/frenet_baseline/waypoints'   # folder that holds one sub-folder per track
    NAME: str = 'venue_ccw'                                     # track sub-folder
    RACELINE: str = 'raceline_stable.csv'                       # file inside ROOT/NAME (or an absolute path)
    MAP: str = ''                                               # map .yaml inside ROOT/NAME (or absolute); '' = skip check


@dataclass
class ScanParams:
    ENABLE: bool = False                 # [1] subscribe to /scan at all (no stage uses it yet)
    PUBLISH: bool = False                # republish the result on /scan_filtered (for RViz)
    RANGE_MIN: float = 0.30              # drop closer returns (the car itself) [m]
    RANGE_MAX: float = 10.0              # drop farther returns [m]
    STRIDE: int = 1                      # keep every n-th beam


@dataclass
class AvoidParams:
    """[3] Barrier-state (BaS) obstacle avoidance. Defaults = the BaS demo config tested on the car (bas_demo.yaml)."""
    ENABLE: bool = False                 # false: no scan use, d_ref = 0, bit-identical to bv3 without avoidance
    # car footprint, measured from the pose point (car: /pf/pose/odom = laser point)
    CAR_HALF_WIDTH: float = 0.15
    CAR_FRONT: float = 0.414             # pose point -> front bumper, plus margin for the steering delay [m]
    CAR_REAR: float = 0.40               # pose point -> rear bumper [m]
    BAS_MARGIN: float = 0.10             # extra clearance around the car [m]
    # barrier
    H_ACT_OBS: float = 4.0               # barrier switches on when the ellipse level h drops below this (~4 m out)
    T_LOOK: float = 0.45                 # obstacle ellipse stretched ahead by T_LOOK * v [s]
    K_B_OBS: float = 3.0                 # barrier-state gain (how hard d_ref is pushed)
    K_DB_OBS: float = 0.10               # gain on the barrier's rate
    K_RELAX: float = 1.5                 # d_ref decays back to the line when clear [1/s]
    DREF_RATE_MAX: float = 3.0           # max sideways shift rate of the reference [m/s]
    DREF_EPS: float = 0.005              # snap d_ref to exactly 0 when inactive and below this [m]
    PSI_REF_TAU: float = 0.10            # low-pass on the d_ref rate used for the reference heading [s]
    PSI_REF_MAX: float = 0.30            # cap on the reference heading relative to the path [rad]
    H_MIN: float = 1e-3
    B_MAX: float = 200.0
    ZDOT_MAX: float = 500.0
    DENOM_MIN: float = 0.1               # floor for (1 - kappa * d)
    # speed while avoiding
    K_BV: float = 0.8                    # speed scale = 1 / (1 + K_BV * sum B~ ahead + K_BV_RETURN * |d_ref|)
    K_BV_RETURN: float = 0.3
    V_BAS_MIN: float = 1.5               # never slower than this while avoiding [m/s]
    # LiDAR -> obstacles
    LASER_X: float = 0.0                 # laser ahead of the pose point [m] (car: 0, the pose IS the laser point)
    LASER_Y: float = 0.0
    SCAN_STRIDE: int = 2                 # use every n-th beam
    SCAN_FOV_DEG: float = 200.0          # beams within +-FOV/2 of straight ahead
    RANGE_MIN: float = 0.30              # drops the car's own returns (0.13-0.19 m seen in the bags) [m]
    OBS_RANGE: float = 6.0               # detection distance [m]
    OBS_BEHIND: float = 0.6              # keep obstacles until this far behind [m]
    WALL_GATE: float = 0.15              # points closer than this to a wall are wall, not obstacle [m]
    CLUSTER_GAP: float = 0.25            # beams farther apart than this start a new cluster [m]
    MIN_PTS: int = 3                     # points per cluster
    OBS_R_MIN: float = 0.15              # minimum obstacle half-size [m]
    OBS_PERSIST: int = 2                 # scans in a row before obstacles are trusted
    OBS_CLEAR: int = 4                   # empty scans before obstacles are dropped
    OBS_BIRTH_RANGE: float = 1.5         # a NEW obstacle must first be seen farther than this [m]; 0 = off
    OBS_MATCH_DS: float = 1.0            # same obstacle if within this along the path [m] ...
    OBS_MATCH_DD: float = 0.5            # ... and this sideways [m]
    OBS_EDGE_JUMP: float = 0.3           # beams beside a cluster must read this much farther (stands in front) [m]; 0 = off
    OBS_EDGE_SIDES: int = 2              # 2 = both sides, 1 = one side is enough (box touching a wall)


@dataclass
class PoseParams:
    POSE_TO_REAR_AXLE_M: float = 0.0     # pose point -> rear axle along -heading [m]; 0 = track the pose point
    MAX_WALK_SEGS: int = 50              # segments walked per cycle before declaring lost
    RELOC_DIST_M: float = 1.5            # farther than this from the path = search again [m]


@dataclass
class PlannerParams:
    DS: float = 0.1                      # resample spacing if the path file is not uniform [m]
    TRACK_HALF_W: float = 0.8            # wall distance when the file has no width columns [m]
    FF_PREVIEW_T: float = 0.12           # curvature read this far ahead (steering delay) [s] ...
    FF_PREVIEW_MIN: float = 0.3          # ... but at least this far [m]
    SPEED_PREVIEW_T: float = 0.10        # speed read this far ahead (drive lag) [s]
    SPEED_SCALE: float = 1.0             # x path speed
    SPEED_MAX: float = 4.0               # [m/s]
    SPEED_MIN: float = 1.0               # [m/s]
    A_Y: float = 5.0                     # only for files without curvature + speed columns: lateral limit [m/s^2]
    A_BRAKE: float = 5.0                 # [m/s^2]
    A_ACCEL: float = 4.0                 # [m/s^2]
    KAPPA_MEDIAN: int = 5                # median window on derived curvature (points)


@dataclass
class OptimizerParams:
    """Shared by every controller."""
    TYPE: str = 'LQR'                    # LQR | PURE_PURSUIT | PD
    WHEELBASE: float = 0.325             # L [m]
    K_US: float = 0.0215                 # understeer gradient for the feedforward [rad per m/s^2] (car value)
    STEER_LIMIT: float = 0.40            # |delta| max [rad]
    STEER_STEP_MAX: float = 0.18         # slew limit per cycle [rad]; 0 = use STEER_RATE_MAX
    STEER_RATE_MAX: float = 9.0          # slew limit per second [rad/s] (when STEER_STEP_MAX = 0)
    BRAKE_ENABLE: bool = True            # VESC brake current when too fast
    BRAKE_THRESHOLD: float = 0.5         # overspeed that triggers braking [m/s]
    BRAKE_CURRENT_MIN: float = 3000.0    # [mA]
    BRAKE_CURRENT_MAX: float = 8000.0    # [mA]


@dataclass
class LqrParams:
    Q_D: float = 1.0                     # cost on lateral error (d - d_ref)^2
    Q_PSI: float = 0.1                   # cost on heading error e_psi^2
    R_STEER: float = 3.0                 # cost on steering correction^2 (bigger = softer)
    REF_X: float = 0.26                  # tracked point ahead of the rear axle [m] (car: pose = laser 0.26; gym: 0)
    DT: float = 0.025                    # control period the model is discretized at [s] (40 Hz pose)
    DELAY_S: float = 0.125               # pose age + servo lag covered by forward prediction [s]; 0 = off (gym)
    V_MIN: float = 0.5                   # gain table range [m/s]
    V_MAX: float = 10.0
    V_STEP: float = 0.05


@dataclass
class PurePursuitParams:
    LA_TIME: float = 0.25                # lookahead = LA_TIME * v ... [s]
    LA_MIN: float = 0.8                  # ... inside [LA_MIN, LA_MAX] [m]
    LA_MAX: float = 2.5
    REACT_V0: float = 2.2                # correction scale = clamp((REACT_V0 / v)^REACT_EXP, REACT_MIN, 1); 0 = off
    REACT_EXP: float = 2.0
    REACT_MIN: float = 0.2


@dataclass
class PDParams:
    KP: float = 0.45                     # lateral gain at V_REF; scheduled kp = KP * (V_REF / v)^2
    KD: float = 0.55                     # heading gain at V_REF; scheduled kd = KD * (V_REF / v)
    V_REF: float = 3.0
    V_MIN_SCHED: float = 1.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def resolve_track(t: TrackParams):
    """(raceline path, map path or '') from the track block. Absolute entries are used as given."""
    base = os.path.join(os.path.expanduser(t.ROOT), t.NAME)
    line = os.path.expanduser(t.RACELINE)
    line = line if os.path.isabs(line) else os.path.join(base, line)
    m = os.path.expanduser(t.MAP) if t.MAP else ''
    m = m if (not m or os.path.isabs(m)) else os.path.join(base, m)
    return line, m


# ============================================================
# [1] SCAN PREP
# ============================================================

class ScanPrep:
    """Crop + downsample the LiDAR scan. Output kept in self.ranges (float32, NaN = dropped beam)."""

    def __init__(self, p: ScanParams):
        self.p = p
        self.ranges = None
        self.angle_min = 0.0
        self.angle_inc = 0.0

    def update(self, ranges, angle_min: float, angle_inc: float):
        p = self.p
        r = np.asarray(ranges, dtype=np.float32)[::max(1, p.STRIDE)].copy()
        r[(r < p.RANGE_MIN) | (r > p.RANGE_MAX) | ~np.isfinite(r)] = np.nan
        self.ranges, self.angle_min, self.angle_inc = r, angle_min, angle_inc * max(1, p.STRIDE)
        return self.ranges


# ============================================================
# [2] POSE -> STATE
# ============================================================

@dataclass
class State:
    """Vehicle state for one cycle. One instance, overwritten every cycle."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    yaw_rate: float = 0.0
    xr: float = 0.0                      # tracked point (pose point or rear axle, see POSE_TO_REAR_AXLE_M)
    yr: float = 0.0
    seg: int = -1                        # path segment; -1 = not localized yet
    t: float = 0.0                       # position along that segment, 0..1
    s: float = 0.0                       # arc length [m]
    d: float = 0.0                       # lateral offset, + = left of the path [m]
    e_psi: float = 0.0                   # heading error, + = car points left of the path [rad]
    kappa: float = 0.0                   # path curvature at s [1/m]
    psi_path: float = 0.0                # path heading at s [rad]
    fx: float = 0.0                      # foot point on the path (RViz)
    fy: float = 0.0


class FrenetTracker:
    """Pose -> Frenet state on one path. Full search once, then a short walk from the last segment."""

    def __init__(self, path, p: PoseParams, log=print):
        self.line, self.p, self.log = path, p, log
        self.state = State()

    def update(self, x, y, yaw, v, yaw_rate) -> State:
        st, L, p = self.state, self.line, self.p
        st.x, st.y, st.yaw, st.v, st.yaw_rate = x, y, yaw, v, yaw_rate
        off = p.POSE_TO_REAR_AXLE_M
        xr, yr = (x, y) if off == 0.0 else (x - off * math.cos(yaw), y - off * math.sin(yaw))
        st.xr, st.yr = xr, yr

        if st.seg < 0:
            st.seg = self.nearest(xr, yr, yaw)
            self.log(f'Localized on path: segment {st.seg}, s={L.s[st.seg]:.1f} m')
        seg, t, ok = self._walk(xr, yr, st.seg)
        d = self._offset(xr, yr, seg, t)
        if not ok or abs(d) > p.RELOC_DIST_M:                       # lost (spin, reset): search again
            fwd = np.arange(st.seg - 10, st.seg + int(10.0 / L.ds))
            new = self.nearest(xr, yr, yaw, fwd)
            if (L.x[new] - xr) ** 2 + (L.y[new] - yr) ** 2 > p.RELOC_DIST_M ** 2:
                new = self.nearest(xr, yr, yaw)
            self.log(f'Relocalized: segment {st.seg} -> {new}')
            seg, t, _ = self._walk(xr, yr, new)
            d = self._offset(xr, yr, seg, t)

        # >>> TELEMETRY  s, d, e_psi, kappa
        # >>> METRIC     lap time, lateral spread (std of d per s-bin across laps), RMS d / e_psi
        nxt = (seg + 1) % L.n
        st.seg, st.t, st.d = seg, t, d
        st.s = L.s[seg] + t * L.seg_len[seg]
        st.psi_path = L.psi[seg] + t * wrap(L.psi[nxt] - L.psi[seg])
        st.e_psi = wrap(yaw - st.psi_path)
        st.kappa = L.kappa[seg] + t * (L.kappa[nxt] - L.kappa[seg])
        st.fx = L.x[seg] + t * L.seg_dx[seg]
        st.fy = L.y[seg] + t * L.seg_dy[seg]
        return st

    def nearest(self, x, y, yaw, idxs=None) -> int:
        """Nearest path point facing the car (|heading diff| < 90 deg). O(N); start-up and relocalization only."""
        L = self.line
        n = L.n
        idxs = np.arange(n) if idxs is None else np.asarray(idxs) % n
        d2 = np.full(n, np.inf)
        d2[idxs] = (L.np_x[idxs] - x) ** 2 + (L.np_y[idxs] - y) ** 2
        facing = np.cos(yaw - L.np_psi) > 0.0
        if np.any(facing[idxs]):
            d2[~facing] = np.inf
        k = min(5, int(np.isfinite(d2).sum()) - 1)
        cand = np.argpartition(d2, max(k, 0))[:max(k, 0) + 1]
        return int(min(cand, key=lambda i: d2[i] + 2.0 * abs(wrap(yaw - L.psi[i]))))

    def _t_on(self, x, y, seg) -> float:
        L = self.line
        return ((x - L.x[seg]) * L.seg_dx[seg] + (y - L.y[seg]) * L.seg_dy[seg]) * L.seg_inv_len2[seg]

    def _walk(self, x, y, seg):
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

    def _offset(self, x, y, seg, t) -> float:
        L = self.line
        ex = x - (L.x[seg] + t * L.seg_dx[seg])
        ey = y - (L.y[seg] + t * L.seg_dy[seg])
        return (L.seg_dx[seg] * ey - L.seg_dy[seg] * ex) / L.seg_len[seg]


# ============================================================
# [3] AVOIDANCE: BARRIER STATES (BaS)
# ============================================================

class BarrierAvoidance:
    """LiDAR obstacles -> barrier state -> d_ref (sideways shift of the reference) + speed scale.

    The proven BaS layer (bv1 BaS demo on the car, bv2 sim tests), kept to its reliable core:
        [3a] scan -> obstacles in (s, d): beams near the walls dropped, neighbouring beams clustered,
             RANGE_MIN (own body), BIRTH (new obstacles must appear far away), EDGE (must stand in front of
             its background), persistence (OBS_PERSIST in, OBS_CLEAR out)
        [3b] one ellipse barrier per obstacle, stretched ahead by T_LOOK * v, on the PLANNED offset d_ref:
             h = (ds/a)^2 + ((d_ref - d_obs)/b)^2 - 1,  B~ = 1/h - 1/H_ACT_OBS  (0 when far)
        [3c] d_ref_dot = sum side * max(0, K_B * B~ + K_DB * dB/dt) - K_RELAX * d_ref, clamped inside the walls
    Pass side = larger gap to the wall, chosen once per obstacle and locked until it leaves the range.
    No wall barriers (the tracking stays on the line; the d_ref clamp keeps the plan inside the walls).
    Cost: O(K) per cycle (K = obstacles); detection O(M*W) numpy once per NEW scan (M beams, W window points)."""

    def __init__(self, path: 'Path', p: AvoidParams, log=print):
        self.p, self.path = p, path
        x, y = path.np_x, path.np_y
        dx, dy = np.roll(x, -1) - x, np.roll(y, -1) - y
        self.psi = np.arctan2(dy, dx)                               # segment headings (same indices as [2])
        self.cos, self.sin = np.cos(self.psi), np.sin(self.psi)
        self.wl, self.wr = path.np_wl, path.np_wr
        n_ahead = int(math.ceil((p.OBS_RANGE + p.LASER_X) / path.ds)) + 1
        self._win = np.arange(-int(math.ceil(p.OBS_BEHIND / path.ds)) - 2, n_ahead)
        self.scan = None
        self.scan_new = False
        self._beams = None
        self.obstacles = []                                         # (s, d, r_s, r_d, x, y, w_left, w_right)
        self._last_found = []
        self._hits = self._misses = 0
        self.sides = []                                             # [s, side] per obstacle in range: +1 left, -1 right
        self.d_ref = self.d_ref_rate = self.rate_f = 0.0
        self.psi_ref = 0.0                                          # heading of the shifted reference (LQR / PD use it)
        self.speed_scale = 1.0
        self.active = False
        if not path.has_widths:
            log(f'WARN avoidance: path file has no width columns, walls assumed at +-{path.w_left[0]:.2f} m. '
                'Wall points may be taken for obstacles: use a raceline with w_left/w_right.')
        log(f'Avoidance ON: obstacles within {p.OBS_RANGE} m, margin {p.BAS_MARGIN} m, min speed {p.V_BAS_MIN} m/s')

    def set_scan(self, ranges, angle_min: float, angle_inc: float):
        """Store only; processed in the next control cycle."""
        self.scan = (ranges, angle_min, angle_inc)
        self.scan_new = True

    def step(self, st: State, dt: float):
        """One cycle. Uses the pose point (st.x, st.y) and the Frenet state of [2]. Returns (d_ref, speed_scale)."""
        if self.scan_new:
            self.detect(st.x, st.y, st.yaw, st.seg)
            self.scan_new = False
        push, bt_ahead, active = self.barriers(st)
        self.update_d_ref(push, bt_ahead, active, dt, st.seg, st.v)
        return self.d_ref, self.speed_scale

    # ---- [3a] LiDAR -> obstacles in (s, d), once per NEW scan ----

    def detect(self, x, y, yaw, seg):
        p, L = self.p, self.path
        ranges, a0, da = self.scan
        r_all = np.asarray(ranges, dtype=float)
        key = (len(r_all), a0, da)
        if self._beams is None or self._beams[0] != key:            # beam angles cached per scan layout
            ang = a0 + da * np.arange(len(r_all))
            idx = np.where(np.abs(ang) <= math.radians(p.SCAN_FOV_DEG) / 2)[0][::max(1, p.SCAN_STRIDE)]
            self._beams = (key, idx, np.cos(ang[idx]), np.sin(ang[idx]))
        _, idx, ca, sa = self._beams
        r = r_all[idx]
        keep = np.isfinite(r) & (r > p.RANGE_MIN) & (r < p.OBS_RANGE)
        if not keep.any():
            return self._confirm([])
        r, ca, sa, bi = r[keep], ca[keep], sa[keep], idx[keep]

        lx, ly = p.LASER_X + r * ca, p.LASER_Y + r * sa              # laser -> pose-point frame
        c, sn = math.cos(yaw), math.sin(yaw)
        px, py = x + c * lx - sn * ly, y + sn * lx + c * ly            # -> map

        win = (seg + self._win) % L.n                                  # path points near and ahead of the car
        d2 = (px[:, None] - L.np_x[win][None, :]) ** 2 + (py[:, None] - L.np_y[win][None, :]) ** 2
        k = win[np.argmin(d2, axis=1)]
        ex, ey = px - L.np_x[k], py - L.np_y[k]
        ps = L.np_s[k] + ex * self.cos[k] + ey * self.sin[k]
        pd = -ex * self.sin[k] + ey * self.cos[k]

        inside = (pd < self.wl[k] - p.WALL_GATE) & (pd > -self.wr[k] + p.WALL_GATE)
        if inside.sum() < p.MIN_PTS:
            return self._confirm([])
        px, py, ps, pd, rr, bi = px[inside], py[inside], ps[inside], pd[inside], r[inside], bi[inside]

        jump = np.hypot(np.diff(px), np.diff(py)) > p.CLUSTER_GAP
        cuts = np.concatenate([[0], np.where(jump)[0] + 1, [len(px)]])
        found = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < p.MIN_PTS:
                continue
            cs, cd = ps[a:b], pd[a:b]
            s_c, d_c = float(np.mean(cs)) % L.length, float(np.mean(cd))
            r_s = max(p.OBS_R_MIN, 0.5 * float(cs.max() - cs.min()))
            r_d = max(p.OBS_R_MIN, 0.5 * float(cd.max() - cd.min()))
            if p.OBS_BIRTH_RANGE > 0.0 and float(rr[a:b].min()) < p.OBS_BIRTH_RANGE \
                    and not self._continues(s_c, d_c):
                continue                                               # BIRTH: appeared too close
            if p.OBS_EDGE_JUMP > 0.0 and \
                    self._edge_jump(r_all, int(bi[a]), float(rr[a]), -1) + \
                    self._edge_jump(r_all, int(bi[b - 1]), float(rr[b - 1]), 1) < p.OBS_EDGE_SIDES:
                continue                                               # EDGE: continuous with its background
            i = L.index_at(s_c)
            psi = float(self.psi[i])
            found.append((s_c, d_c, r_s, r_d, L.x[i] - d_c * math.sin(psi), L.y[i] + d_c * math.cos(psi),
                          float(self.wl[i]), float(self.wr[i])))
        self._last_found = found
        self._confirm(found)

    def _edge_jump(self, r_all, i, r_edge, step):
        """1 if one of the next 3 valid beams past the cluster edge reads farther than r_edge + OBS_EDGE_JUMP."""
        p, n = self.p, len(r_all)
        j, seen = i, 0
        while seen < 3:
            j += step * max(1, p.SCAN_STRIDE)
            if j < 0 or j >= n:
                return 1
            rj = r_all[j]
            if not math.isfinite(rj) or rj - r_edge > p.OBS_EDGE_JUMP:
                return 1
            if rj <= p.RANGE_MIN:
                continue                                               # own body / no data
            seen += 1
        return 0

    def _continues(self, s_c, d_c):
        p, half, length = self.p, 0.5 * self.path.length, self.path.length
        for ob in self.obstacles + self._last_found:
            if abs((s_c - ob[0] + half) % length - half) < p.OBS_MATCH_DS and abs(d_c - ob[1]) < p.OBS_MATCH_DD:
                return True
        return False

    def _confirm(self, found):
        p = self.p
        if not found:
            self._last_found = []
            self._misses += 1
            if self._misses >= p.OBS_CLEAR:
                self._hits = 0
                self.obstacles = []
            return
        self._hits += 1
        self._misses = 0
        if self._hits >= p.OBS_PERSIST:
            self.obstacles = found

    # ---- [3b] barriers on the PLANNED offset d_ref (the barrier state) ----

    def barriers(self, st: State):
        p, length = self.p, self.path.length
        q, q_dot, v = self.d_ref, self.d_ref_rate, st.v
        s_dot = v * math.cos(st.e_psi) / max(1.0 - st.kappa * st.d, p.DENOM_MIN)
        push, bt_ahead, active = 0.0, 0.0, False
        half, sides_now = 0.5 * length, []
        for (s_j, d_j, r_s, r_d, _, _, wl_j, wr_j) in self.obstacles:
            ds = (st.s - s_j + half) % length - half                    # < 0: obstacle ahead
            if ds < -p.OBS_RANGE or ds > p.OBS_BEHIND:
                continue
            ent = next((e for e in self.sides if abs((s_j - e[0] + half) % length - half) < p.OBS_MATCH_DS), None)
            if ent is None:
                ent = [s_j, 0]
            ent[0] = s_j
            if ent not in sides_now:
                sides_now.append(ent)
            a = r_s + p.BAS_MARGIN + (p.CAR_FRONT + p.T_LOOK * v if ds < 0.0 else p.CAR_REAR)
            b = r_d + p.CAR_HALF_WIDTH + p.BAS_MARGIN
            h = (ds / a) ** 2 + ((q - d_j) / b) ** 2 - 1.0
            h_dot = 2.0 * ds / (a * a) * s_dot + 2.0 * (q - d_j) / (b * b) * q_dot
            bt, zdot = self._shifted_barrier(h, h_dot)
            if bt <= 0.0:
                continue
            if ent[1] == 0:                                            # pass side: larger gap, chosen once
                ent[1] = 1 if wl_j - (d_j + r_d) >= (d_j - r_d) + wr_j else -1
            push += ent[1] * max(0.0, p.K_B_OBS * bt + p.K_DB_OBS * zdot)
            active = True
            if ds < 0.0:
                bt_ahead += bt
        self.sides = sides_now                                         # a side is forgotten when its obstacle leaves the range
        return push, bt_ahead, active

    def _shifted_barrier(self, h, h_dot):
        p = self.p
        if h >= p.H_ACT_OBS:
            return 0.0, 0.0
        hc = max(h, p.H_MIN)
        bt = min(1.0 / hc - 1.0 / p.H_ACT_OBS, p.B_MAX)
        if bt >= p.B_MAX or h <= p.H_MIN:
            return bt, 0.0
        return bt, max(-p.ZDOT_MAX, min(p.ZDOT_MAX, -h_dot / (hc * hc)))

    # ---- [3c] barrier state -> d_ref, speed scale ----

    def update_d_ref(self, push, bt_ahead, active, dt, seg, v):
        p, L = self.p, self.path
        # >>> TELEMETRY  d_ref, avoid_active, n_obstacles
        # >>> METRIC     min clearance to obstacles, avoidance time per lap, lap time with vs without obstacles
        rate = max(-p.DREF_RATE_MAX, min(p.DREF_RATE_MAX, push - p.K_RELAX * self.d_ref))
        d_ref = self.d_ref + rate * dt
        # walls: narrowest corridor over the car's length plus the look-ahead (the track can narrow ahead)
        i0 = seg - int(p.CAR_REAR * L.inv_ds)
        i1 = seg + int((p.CAR_FRONT + p.T_LOOK * max(v, 0.0)) * L.inv_ds) + 1
        idx = [i % L.n for i in range(i0, i1 + 1)]
        m = p.CAR_HALF_WIDTH + p.BAS_MARGIN
        hi = min(L.w_left[i] for i in idx) - m
        lo = -min(L.w_right[i] for i in idx) + m
        d_ref = min(hi, max(lo, d_ref)) if lo < hi else 0.0
        rate = (d_ref - self.d_ref) / dt if dt > 0.0 else 0.0       # actual rate after the clamp (0 while held at a wall)
        if not active and abs(d_ref) < p.DREF_EPS:
            d_ref, rate = 0.0, 0.0
        self.d_ref, self.d_ref_rate, self.active = float(d_ref), float(rate), active
        self.rate_f += (rate - self.rate_f) * dt / (p.PSI_REF_TAU + dt) if d_ref != 0.0 else -self.rate_f
        self.psi_ref = max(-p.PSI_REF_MAX, min(p.PSI_REF_MAX, math.atan2(self.rate_f, max(v, 0.5))))
        self.speed_scale = 1.0 / (1.0 + p.K_BV * bt_ahead + p.K_BV_RETURN * abs(self.d_ref))


# ============================================================
# [4] PLANNER
# ============================================================

@dataclass
class Reference:
    """What the optimizer tracks this cycle. One instance, overwritten every cycle."""
    d_ref: float = 0.0                   # lateral target (0 = on the path) [m]
    psi_ref: float = 0.0                 # heading of the shifted reference relative to the path (from d_ref rate) [rad]
    kappa_ff: float = 0.0                # path curvature at the preview point [1/m]
    preview_m: float = 0.0
    tx: float = 0.0                      # preview point (RViz)
    ty: float = 0.0
    v_ref: float = 0.0                   # path speed x SPEED_SCALE [m/s]
    v_target: float = 0.0                # v_ref inside [SPEED_MIN, SPEED_MAX] [m/s]
    speed_limited: bool = False


class Path:
    """Path arrays as Python lists (per-element access in the loop is ~5x faster than numpy scalars).

    File formats (comma separated, '#' comments):
        generator   x, y, s, psi, kappa, vx, ax, w_left, w_right   (kappa, speed, widths used as given)
        centerline  x, y, w_left, w_right                          (kappa + speed derived once)
        legacy      x, y                                           (kappa + speed derived, fixed widths)
    """

    def __init__(self, path_file: str, p: PlannerParams, log=print):
        data = np.loadtxt(path_file, delimiter=',', comments='#', ndmin=2)
        data = self._uniform(data, p.DS, log)
        x, y = data[:, 0], data[:, 1]
        seg_dx, seg_dy = np.roll(x, -1) - x, np.roll(y, -1) - y
        seg_len = np.hypot(seg_dx, seg_dy)
        s = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        psi = np.arctan2(seg_dy, seg_dx)
        cols = data.shape[1]
        w_left = data[:, 7] if cols >= 9 else (data[:, 2] if cols == 4 else np.full(len(x), p.TRACK_HALF_W))
        w_right = data[:, 8] if cols >= 9 else (data[:, 3] if cols == 4 else np.full(len(x), p.TRACK_HALF_W))
        if cols >= 6:
            psi, kappa, vx = data[:, 3], data[:, 4], data[:, 5]
            src = 'generator (kappa, speed from file)'
        else:
            kappa = self._kappa(psi, seg_len, p.KAPPA_MEDIAN)
            vx = self._speed_profile(kappa, seg_len, p)
            src = 'x,y (kappa, speed derived)'
        self.n = len(x)
        self.length = float(s[-1] + seg_len[-1])
        self.ds = float(np.mean(seg_len))
        self.inv_ds = 1.0 / self.ds
        self.x, self.y, self.s, self.psi = x.tolist(), y.tolist(), s.tolist(), psi.tolist()
        self.kappa, self.vx = kappa.tolist(), vx.tolist()
        self.seg_dx, self.seg_dy, self.seg_len = seg_dx.tolist(), seg_dy.tolist(), seg_len.tolist()
        self.seg_inv_len2 = (1.0 / np.maximum(seg_len ** 2, 1e-12)).tolist()
        self.w_left, self.w_right = np.asarray(w_left, float).tolist(), np.asarray(w_right, float).tolist()
        self.np_x, self.np_y, self.np_psi = x.copy(), y.copy(), np.asarray(psi, float)
        self.np_s, self.np_wl, self.np_wr = s.copy(), np.asarray(w_left, float), np.asarray(w_right, float)
        self.has_widths = cols >= 9 or cols == 4                    # walls known per point (needed by [3])
        log(f'Path {path_file}: {self.n} pts, {self.length:.1f} m, max|kappa| {np.max(np.abs(kappa)):.2f}, '
            f'speed {vx.min():.1f}-{vx.max():.1f} m/s, {src}')

    def index_at(self, s: float) -> int:
        s %= self.length
        i = min(int(s * self.inv_ds), self.n - 1)
        while self.s[i] > s:
            i -= 1
        while i + 1 < self.n and self.s[i + 1] <= s:
            i += 1
        return i

    def at(self, s: float):
        i = self.index_at(s)
        return i, ((s % self.length) - self.s[i]) / self.seg_len[i]

    @staticmethod
    def _uniform(data, ds, log):
        seg = np.hypot(*np.diff(np.vstack([data[:, :2], data[:1, :2]]), axis=0).T)
        if np.max(np.abs(seg - seg.mean())) < 0.01 * seg.mean():
            return data
        s = np.concatenate([[0.0], np.cumsum(seg)])
        closed = np.vstack([data, data[:1]])
        if data.shape[1] >= 6:
            closed[:, 3] = np.unwrap(closed[:, 3])
        s_new = np.arange(0.0, s[-1] - 0.5 * ds, ds)
        out = np.column_stack([np.interp(s_new, s, closed[:, c]) for c in range(data.shape[1])])
        if data.shape[1] >= 6:
            out[:, 3] = (out[:, 3] + np.pi) % (2 * np.pi) - np.pi
        log(f'Path not uniformly spaced: resampled {len(data)} -> {len(out)} points at {ds} m')
        return out

    @staticmethod
    def _kappa(psi, seg_len, window):
        dpsi = np.angle(np.exp(1j * (psi - np.roll(psi, 1))))
        kappa = dpsi / np.maximum(np.roll(seg_len, 1), 1e-6)
        w = int(window)
        if w > 1 and len(kappa) > w:
            pad = w // 2
            ext = np.concatenate([kappa[-pad:], kappa, kappa[:pad]])
            kappa = np.median(np.lib.stride_tricks.sliding_window_view(ext, w), axis=1)
        return kappa

    @staticmethod
    def _speed_profile(kappa, seg_len, p: PlannerParams):
        v = np.clip(np.sqrt(p.A_Y / np.maximum(np.abs(kappa), 1e-4)), p.SPEED_MIN, p.SPEED_MAX)
        n = len(v)
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                j = (i + 1) % n
                v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * p.A_BRAKE * seg_len[i]))
            for i in range(n):
                j = (i - 1) % n
                v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * p.A_ACCEL * seg_len[j]))
        return v


def map_image_size(img: str):
    """(width, height) in pixels from the image header only. PGM (SLAM maps) and PNG (gym maps)."""
    with open(img, 'rb') as f:
        head = f.read(24)
        if head[:8] == b'\x89PNG\r\n\x1a\n':                    # PNG: IHDR width, height (big-endian) at bytes 16-24
            return int.from_bytes(head[16:20], 'big'), int.from_bytes(head[20:24], 'big')
        f.seek(0)                                                   # PGM: P5 <w> <h> <max>, '#' comments allowed
        tok = []
        while len(tok) < 3:
            line = f.readline()
            if not line:
                raise ValueError(f'unsupported map image {img}')
            tok += line.split(b'#')[0].split()
        return int(tok[1]), int(tok[2])


def check_map(map_yaml: str, path: Path, log=print):
    """Warn if the raceline leaves the map's extent (wrong map / raceline pair). Reads only the yaml + image header."""
    if not map_yaml:
        return
    if not os.path.isfile(map_yaml):
        log(f'WARN map not found: {map_yaml}')
        return
    try:
        import yaml
        meta = yaml.safe_load(open(map_yaml))
        img = meta['image'] if os.path.isabs(meta['image']) else os.path.join(os.path.dirname(map_yaml), meta['image'])
        w, h = map_image_size(img)
        res, ox, oy = float(meta['resolution']), meta['origin'][0], meta['origin'][1]
        inside = ((path.np_x >= ox) & (path.np_x <= ox + w * res) & (path.np_y >= oy) & (path.np_y <= oy + h * res))
        if inside.all():
            log(f'Map {map_yaml}: raceline inside the map extent')
        else:
            log(f'WARN raceline leaves the map extent at {int((~inside).sum())} points: wrong map/raceline pair?')
    except Exception as e:                                          # informational only
        log(f'WARN map check skipped ({e})')


class Planner:
    def __init__(self, path: Path, p: PlannerParams):
        self.path, self.p = path, p
        self.ref = Reference()

    def plan(self, st: State, avoid: 'BarrierAvoidance' = None) -> Reference:
        p, L, r = self.p, self.path, self.ref
        r.d_ref = 0.0 if avoid is None else avoid.d_ref             # [3] shifts the target sideways
        r.psi_ref = 0.0 if avoid is None else avoid.psi_ref

        la = max(p.FF_PREVIEW_MIN, p.FF_PREVIEW_T * st.v)
        i, t = L.at(st.s + la)
        nxt = (i + 1) % L.n
        r.preview_m = la
        r.kappa_ff = L.kappa[i] + t * (L.kappa[nxt] - L.kappa[i])
        r.tx = L.x[i] + t * L.seg_dx[i]
        r.ty = L.y[i] + t * L.seg_dy[i]

        # >>> TELEMETRY  v_ref, FLAG_SPEED_LIMITED
        # >>> METRIC     speed tracking error (v - v_ref) per sector, time at the speed cap
        j = L.index_at(st.s + st.v * p.SPEED_PREVIEW_T)
        r.v_ref = L.vx[j] * p.SPEED_SCALE
        r.v_target = max(p.SPEED_MIN, min(r.v_ref, p.SPEED_MAX))
        if avoid is not None and (avoid.active or avoid.speed_scale < 1.0):
            floor = avoid.p.V_BAS_MIN if avoid.active else p.SPEED_MIN
            r.v_target = max(min(floor, r.v_target), r.v_target * avoid.speed_scale)
        r.speed_limited = r.v_target < r.v_ref - 1e-6
        return r


# ============================================================
# [5] OPTIMIZER
# ============================================================

@dataclass
class Command:
    steer_raw: float = 0.0               # controller output before limits [rad]
    steer: float = 0.0                   # published [rad]
    steer_saturated: bool = False
    speed: float = 0.0                   # published [m/s]
    brake_current: float = 0.0           # 0 = no brake command


class Optimizer:
    """Base: a controller implements steer(); slew limit, clamp, speed and brake are shared."""
    NAME = 'base'
    Params = None

    def __init__(self, p: OptimizerParams, path: Path, cp=None, log=print):
        self.p, self.path, self.cp, self.log = p, path, cp, log
        self.prev_steer = 0.0
        self.cmd = Command()

    def step(self, st: State, ref: Reference, dt: float) -> Command:
        c, p = self.cmd, self.p
        # >>> TELEMETRY  steer_raw, steer_cmd, steer_saturated
        # >>> METRIC     % cycles saturated, steering smoothness (std of d(steer)/dt)
        raw = self.steer(st, ref, dt)
        step = p.STEER_STEP_MAX if p.STEER_STEP_MAX > 0.0 else p.STEER_RATE_MAX * dt
        delta = min(self.prev_steer + step, max(self.prev_steer - step, raw))
        delta = min(p.STEER_LIMIT, max(-p.STEER_LIMIT, delta))
        self.prev_steer = delta
        c.steer_raw, c.steer, c.steer_saturated = raw, delta, abs(delta - raw) > 1e-6

        # >>> TELEMETRY  speed_cmd
        # >>> METRIC     braking events per lap
        c.speed = ref.v_target
        c.brake_current = 0.0
        if p.BRAKE_ENABLE:
            over = st.v - c.speed
            if over > p.BRAKE_THRESHOLD:
                frac = min(over / 3.0, 1.0)
                c.brake_current = p.BRAKE_CURRENT_MIN + frac * (p.BRAKE_CURRENT_MAX - p.BRAKE_CURRENT_MIN)
        return c

    def feedforward(self, kappa: float, v: float) -> float:
        """Steering that holds curvature kappa at speed v: kinematic + understeer correction."""
        return math.atan(self.p.WHEELBASE * kappa) + self.p.K_US * v * v * kappa

    def steer(self, st: State, ref: Reference, dt: float) -> float:
        raise NotImplementedError


class LqrOptimizer(Optimizer):
    """delta = feedforward(kappa_ff, v) - K(v) . [d - d_ref, e_psi]

    K(v) comes from the discrete Riccati equation on the single-track error model, solved once per speed
    on a grid at start-up. At run time the gain is one table lookup."""
    NAME = 'lqr'
    Params = LqrParams

    def __init__(self, p, path, cp=None, log=print):
        super().__init__(p, path, cp, log)
        cp = self.cp
        self.v_grid = np.arange(cp.V_MIN, cp.V_MAX + 0.5 * cp.V_STEP, cp.V_STEP)
        Q = np.diag([cp.Q_D, cp.Q_PSI])
        R = np.array([[cp.R_STEER]])
        self.K = np.array([self._dlqr(*self._error_model(v), Q, R) for v in self.v_grid])   # (n_v, 2)
        self.k_d, self.k_psi = self.K[:, 0].tolist(), self.K[:, 1].tolist()
        self.inv_step = 1.0 / cp.V_STEP
        self.v_lo, self.v_hi = float(self.v_grid[0]), float(self.v_grid[-1])   # Python floats: ROS fields reject numpy types
        n = max(0, int(round(cp.DELAY_S / cp.DT)))                 # commands still 'in flight'
        self.sent = collections.deque([0.0] * n, maxlen=n) if n else None
        log('LQR gains [k_d, k_psi]: ' + ', '.join(
            f'{v:.0f} m/s {self.gain(v)[0]:.2f}/{self.gain(v)[1]:.2f}' for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))

    # >>> SINGLE-TRACK MODEL (a): linearized error dynamics for the gain table
    def _error_model(self, v: float):
        """Kinematic single-track model in the path frame, linearized about the path (d = 0, e_psi = 0):
             d_dot     = v * e_psi + REF_X * r           (lateral speed of the tracked point)
             e_psi_dot = r - v * kappa,   r = v * tan(delta) / L  ->  ~ v * delta_fb / L
           The curvature term is carried by the feedforward, so the feedback sees
             x = [d, e_psi],  u = delta_fb
             x_dot = [[0, v], [0, 0]] x + [[REF_X * v / L], [v / L]] u
           Discretized exactly (zero-order hold) over DT."""
        L, h, p = self.p.WHEELBASE, self.cp.DT, self.cp.REF_X
        A = np.array([[1.0, v * h],
                      [0.0, 1.0]])
        B = np.array([[v * v * h * h / (2.0 * L) + p * v * h / L],
                      [v * h / L]])
        return A, B

    @staticmethod
    def _dlqr(A, B, Q, R, iters=2000, tol=1e-10):
        """Discrete LQR gain by Riccati iteration (no scipy needed on the car)."""
        P = Q.copy()
        for _ in range(iters):
            BtP = B.T @ P
            K = np.linalg.solve(R + BtP @ B, BtP @ A)
            P_new = Q + A.T @ P @ (A - B @ K)
            if np.max(np.abs(P_new - P)) < tol:
                P = P_new
                break
            P = P_new
        BtP = B.T @ P
        return np.linalg.solve(R + BtP @ B, BtP @ A).ravel()

    def gain(self, v: float):
        """Linear interpolation in the gain table. O(1)."""
        f = (min(max(v, self.v_lo), self.v_hi) - self.v_lo) * self.inv_step
        i = min(int(f), len(self.k_d) - 2)
        w = f - i
        return (self.k_d[i] + w * (self.k_d[i + 1] - self.k_d[i]),
                self.k_psi[i] + w * (self.k_psi[i + 1] - self.k_psi[i]))

    # >>> SINGLE-TRACK MODEL (b): nonlinear error dynamics rolled forward over the actuation delay
    def _predict(self, d: float, e: float, s: float, v: float):
        """Where the errors will be when this command takes effect. Starts from the measured (old) state and
        replays the steering commands sent during the last DELAY_S, oldest first, one DT step each:
             d_dot = v * sin(e) + REF_X * r * cos(e)
             e_dot = r - kappa(s) * v * cos(e) / (1 - kappa(s) * d)
             s_dot = v * cos(e) / (1 - kappa(s) * d)
             r     = v * tan(delta) / (L + K_US * v^2)
        r is the kinematic yaw rate with the same understeer number the feedforward uses. With plain L the
        prediction over-turns in every corner (the feedforward adds K_US*v^2*kappa of extra steering that the
        car does not turn into yaw) and the car runs wide; the test sim showed this clearly."""
        p, h, path = self.cp.REF_X, self.cp.DT, self.path
        Le = self.p.WHEELBASE + self.p.K_US * v * v
        for delta in self.sent:
            r = v * math.tan(delta) / Le
            i, t = path.at(s)
            k = path.kappa[i]
            c, sn = math.cos(e), math.sin(e)
            g = v * c / max(1.0 - k * d, 0.1)
            d += h * (v * sn + p * r * c)
            e += h * (r - k * g)
            s += h * g
        return d, e

    def steer(self, st: State, ref: Reference, dt: float) -> float:
        d, e = st.d - ref.d_ref, st.e_psi
        if self.sent is not None:
            d, e = self._predict(d, e, st.s, st.v)
        e -= ref.psi_ref                                            # track the shifted reference's heading (0 without [3])
        k_d, k_psi = self.gain(st.v)
        return self.feedforward(ref.kappa_ff, st.v) - (k_d * d + k_psi * e)

    def step(self, st: State, ref: Reference, dt: float) -> Command:
        c = super().step(st, ref, dt)
        if self.sent is not None:
            self.sent.append(c.steer)                               # after slew + clamp: what the servo got
        return c


class PurePursuit(Optimizer):
    """bv2 race law (bit-identical to bv2)."""
    NAME = 'pure_pursuit'
    Params = PurePursuitParams

    def steer(self, st: State, ref: Reference, dt: float) -> float:
        cp, L = self.cp, self.path
        la = min(cp.LA_MAX, max(cp.LA_MIN, cp.LA_TIME * st.v))
        i, t = L.at(st.s + la)
        nxt = (i + 1) % L.n
        px, py = L.x[i] + t * L.seg_dx[i], L.y[i] + t * L.seg_dy[i]
        if ref.d_ref != 0.0:
            psi_t = L.psi[i] + t * wrap(L.psi[nxt] - L.psi[i])
            px, py = px - ref.d_ref * math.sin(psi_t), py + ref.d_ref * math.cos(psi_t)
        c, sn = math.cos(st.yaw), math.sin(st.yaw)
        dx, dy = px - st.xr, py - st.yr
        xl, yl = c * dx + sn * dy, -sn * dx + c * dy
        k_cmd = 2.0 * yl / max(xl * xl + yl * yl, 1e-6)
        if cp.REACT_V0 > 0.0:
            react = min(1.0, max(cp.REACT_MIN, (cp.REACT_V0 / max(st.v, 0.1)) ** cp.REACT_EXP))
            if react < 1.0:
                lx, ly = L.x[i] + t * L.seg_dx[i] - st.fx, L.y[i] + t * L.seg_dy[i] - st.fy
                c, sn = math.cos(st.psi_path), math.sin(st.psi_path)
                xl, yl = c * lx + sn * ly, -sn * lx + c * ly
                k_path = 2.0 * yl / max(xl * xl + yl * yl, 1e-6)
                k_cmd = k_path + react * (k_cmd - k_path)
        return self.feedforward(k_cmd, st.v)


class FrenetPD(Optimizer):
    """bv2 Frenet PD law (bit-identical to bv2)."""
    NAME = 'pd'
    Params = PDParams

    def steer(self, st: State, ref: Reference, dt: float) -> float:
        cp = self.cp
        v = max(st.v, cp.V_MIN_SCHED)
        kp = cp.KP * (cp.V_REF / v) ** 2
        kd = cp.KD * (cp.V_REF / v)
        return self.feedforward(ref.kappa_ff, st.v) - (kp * (st.d - ref.d_ref) + kd * (st.e_psi - ref.psi_ref))


OPTIMIZERS = {'LQR': LqrOptimizer, 'PURE_PURSUIT': PurePursuit, 'PD': FrenetPD}


# ============================================================
# Pipeline without ROS (same code in the node, the offline sim and tests)
# ============================================================

class Bv3:
    def __init__(self, track: TrackParams, pose_p: PoseParams, plan_p: PlannerParams, opt_p: OptimizerParams,
                 ctrl_p=None, warmup_cycles: int = 10, log=print, avoid_p: AvoidParams = None):
        line_file, map_file = resolve_track(track)
        cls = OPTIMIZERS[opt_p.TYPE.upper()]
        self.path = Path(line_file, plan_p, log)                    # [4] path, loaded once
        check_map(map_file, self.path, log)
        self.pose = FrenetTracker(self.path, pose_p, log)           # [2]
        self.avoid = (BarrierAvoidance(self.path, avoid_p, log)
                      if avoid_p is not None and avoid_p.ENABLE else None)   # [3] None = off
        self.planner = Planner(self.path, plan_p)                   # [4]
        self.opt = cls(opt_p, self.path, ctrl_p if ctrl_p is not None else (cls.Params() if cls.Params else None),
                       log)                                         # [5]
        self.warmup, self.warmup_cycles = 0, warmup_cycles
        self.t_ms = [0.0, 0.0, 0.0, 0.0]                           # frenet, avoidance, plan, control
        log(f'Optimizer {cls.__name__} ({opt_p.TYPE}): {self.opt.cp}')

    def step(self, x, y, yaw, v, yaw_rate, dt):
        """One cycle -> (state, reference, command), or None while warming up (publish a stop)."""
        if self.warmup < self.warmup_cycles:
            self.warmup += 1
            return None
        t0 = time.perf_counter()
        st = self.pose.update(x, y, yaw, v, yaw_rate)              # [2]
        t1 = time.perf_counter()
        if self.avoid is not None:
            self.avoid.step(st, dt)                                # [3]
        t2 = time.perf_counter()
        ref = self.planner.plan(st, self.avoid)                    # [4]
        t3 = time.perf_counter()
        cmd = self.opt.step(st, ref, dt)                           # [5]
        t4 = time.perf_counter()
        # >>> TELEMETRY  t_frenet_ms, t_obstacles_ms, t_plan_ms, t_control_ms
        # >>> METRIC     compute budget per stage (rule: total < 0.5 x pose period)
        self.t_ms[0], self.t_ms[1] = (t1 - t0) * 1e3, (t2 - t1) * 1e3
        self.t_ms[2], self.t_ms[3] = (t3 - t2) * 1e3, (t4 - t3) * 1e3
        return st, ref, cmd

    def set_scan(self, ranges, angle_min: float, angle_inc: float):
        if self.avoid is not None:
            self.avoid.set_scan(ranges, angle_min, angle_inc)


# ============================================================
# [6] NODE
# ============================================================
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import Point
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import ColorRGBA, Float64
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    rclpy = None
try:
    from race_msgs.msg import Telemetry
except ImportError:
    Telemetry = None


if rclpy is not None:
    class Bv3Node(Node):
        def __init__(self):
            super().__init__('bv3')
            gp = self._param
            track = self._group('track', TrackParams())             # [0] first: which map + raceline
            pose_topic = gp('pose_topic', '/pf/pose/odom')          # sim: /ego_racecar/odom
            odom_topic = gp('odom_topic', '')                       # '' = speed from the pose message
            scan_topic = gp('scan_topic', '/scan')
            self.telemetry_on = gp('TELEMETRY_ENABLE', True) and Telemetry is not None
            self.viz_on = gp('VIZ_ENABLE', False)
            self.viz_period = 1.0 / gp('VIZ_RATE_HZ', 10.0)

            scan_p = self._group('scan', ScanParams())
            pose_p = self._group('pose', PoseParams())
            plan_p = self._group('planner', PlannerParams())
            opt_p = self._group('optimizer', OptimizerParams())
            cls = OPTIMIZERS[opt_p.TYPE.upper()]
            ctrl_p = self._group(cls.NAME, cls.Params()) if cls.Params else None
            avoid_p = self._group('avoid', AvoidParams())
            self.core = Bv3(track, pose_p, plan_p, opt_p, ctrl_p, gp('WARMUP_CYCLES', 10), self.get_logger().info,
                            avoid_p)
            self.scan = ScanPrep(scan_p)

            self.v = 0.0
            self.yaw_rate = 0.0
            self.prev_t = None
            self.next_viz = 0.0
            self.cycle = 0
            self.avoid_prev = False

            # ---- outputs ----
            self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
            self.brake_pub = self.create_publisher(Float64, '/commands/motor/brake', 1)
            self.telem_pub = self.create_publisher(Telemetry, '/telemetry', 50) if self.telemetry_on else None
            self.scan_pub = self.create_publisher(LaserScan, '/scan_filtered', 1) if scan_p.ENABLE and scan_p.PUBLISH else None
            if self.viz_on:
                latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                self.create_publisher(MarkerArray, '/bv3/path', latched).publish(self._path_marker())
                self.viz_pub = self.create_publisher(MarkerArray, '/bv3/targets', 1)

            # ---- inputs (best effort, depth 1: always the newest message) ----
            sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
            self.use_pose_twist = not odom_topic
            if odom_topic:
                self.create_subscription(Odometry, odom_topic, self.odom_callback, sensor_qos)
            if scan_p.ENABLE or avoid_p.ENABLE:
                self.create_subscription(LaserScan, scan_topic, self.scan_callback, sensor_qos)
            self.create_subscription(Odometry, pose_topic, self.pose_callback, sensor_qos)
            self.get_logger().info(f'bv3 ready: track {track.NAME}, pose {pose_topic}, optimizer {opt_p.TYPE}, '
                                   f'avoidance {"ON" if avoid_p.ENABLE else "off"}')

        # ---- [0] parameters: top-level values and one group per stage ----

        def _param(self, name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        def _group(self, group, obj):
            for f in fields(obj):
                setattr(obj, f.name, self._param(f'{group}.{f.name}', getattr(obj, f.name)))
            self.get_logger().info(f'{group}: ' + ', '.join(f'{f.name}={getattr(obj, f.name)}' for f in fields(obj)))
            return obj

        # ---- inputs ----

        def odom_callback(self, msg):
            self.v = msg.twist.twist.linear.x
            self.yaw_rate = msg.twist.twist.angular.z

        def scan_callback(self, msg):                               # [1] and [3]
            self.core.set_scan(msg.ranges, msg.angle_min, msg.angle_increment)   # [3] raw scan, stored only
            if not self.scan.p.ENABLE:
                return
            r = self.scan.update(msg.ranges, msg.angle_min, msg.angle_increment)
            if self.scan_pub is not None:
                out = LaserScan(header=msg.header, angle_min=msg.angle_min, angle_max=msg.angle_max,
                                angle_increment=self.scan.angle_inc, time_increment=msg.time_increment,
                                scan_time=msg.scan_time, range_min=self.scan.p.RANGE_MIN,
                                range_max=self.scan.p.RANGE_MAX)
                out.ranges = np.nan_to_num(r, nan=float('inf')).tolist()
                self.scan_pub.publish(out)

        def pose_callback(self, msg):
            """One pose message = one control cycle."""
            t_start = time.perf_counter()
            pp = msg.pose.pose
            x, y, yaw = pp.position.x, pp.position.y, quat_to_yaw(pp.orientation)
            if self.use_pose_twist:
                self.v = msg.twist.twist.linear.x
                self.yaw_rate = msg.twist.twist.angular.z
            t_now = time.monotonic()                                # node clock: pose stamps can repeat or jump
            dt = 0.025 if self.prev_t is None else min(0.05, max(0.005, t_now - self.prev_t))
            self.prev_t = t_now

            out = self.core.step(x, y, yaw, self.v, self.yaw_rate, dt)

            now = self.get_clock().now()
            drive = AckermannDriveStamped()
            drive.header.stamp = now.to_msg()
            drive.header.frame_id = 'base_link'
            if out is not None:
                st, ref, cmd = out
                drive.drive.steering_angle = cmd.steer
                drive.drive.speed = cmd.speed
            self.drive_pub.publish(drive)
            if out is None:
                return
            av = self.core.avoid
            if av is not None and av.active != self.avoid_prev:     # one log line per avoidance event
                self.avoid_prev = av.active
                self.get_logger().info(f'AVOID {"ON" if av.active else "off"}: d_ref {av.d_ref:+.2f} m, '
                                       f'obstacles {len(av.obstacles)}, s {st.s:.1f} m')
            if cmd.brake_current > 0.0:
                self.brake_pub.publish(Float64(data=cmd.brake_current))
            if self.viz_on and t_start >= self.next_viz:
                self.next_viz = t_start + self.viz_period
                self._publish_targets(st, ref, cmd)
            if self.telemetry_on:
                self._publish_telemetry(msg, now, st, ref, cmd, t_start)

        # ---- telemetry (record only; metrics are computed post-run) ----

        def _publish_telemetry(self, pose_msg, now, st, ref, cmd, t_start):
            m = Telemetry()
            m.header.stamp = now.to_msg()
            m.header.frame_id = 'map'
            m.pose_stamp = pose_msg.header.stamp
            m.cycle = self.cycle
            self.cycle += 1
            m.controller_id = Telemetry.CONTROLLER_PP                 # every bv3 law (LQR / PP / PD) = baseline, not MPC
            m.x, m.y, m.yaw, m.v, m.yaw_rate = st.x, st.y, st.yaw, st.v, st.yaw_rate
            m.s, m.d, m.e_psi, m.kappa = st.s, st.d, st.e_psi, st.kappa
            m.d_ref, m.v_ref, m.lookahead_m = ref.d_ref, ref.v_ref, ref.preview_m
            av = self.core.avoid
            m.n_obstacles = 0 if av is None else min(255, len(av.obstacles))
            m.avoid_active = False if av is None else bool(av.active)
            m.steer_raw, m.steer_cmd, m.speed_cmd = cmd.steer_raw, cmd.steer, cmd.speed
            m.steer_saturated = cmd.steer_saturated
            m.t_frenet_ms, m.t_obstacles_ms, m.t_plan_ms, m.t_control_ms = self.core.t_ms
            m.t_total_ms = (time.perf_counter() - t_start) * 1e3
            m.flags = Telemetry.FLAG_SPEED_LIMITED if ref.speed_limited else 0
            self.telem_pub.publish(m)

        # ---- RViz (VIZ_ENABLE; off on the car) ----

        @staticmethod
        def _marker(mid, mtype, size, rgba):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = 'bv3', mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = size
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            return m

        def _path_marker(self):
            L = self.core.path
            m = self._marker(0, Marker.LINE_STRIP, 0.05, (1.0, 1.0, 1.0, 1.0))
            vmin, span = min(L.vx), max(max(L.vx) - min(L.vx), 1e-6)
            for i in list(range(L.n)) + [0]:                        # coloured by speed: blue slow, red fast
                c = (L.vx[i] - vmin) / span
                m.points.append(Point(x=L.x[i], y=L.y[i], z=0.02))
                m.colors.append(ColorRGBA(r=c, g=1.0 - abs(2.0 * c - 1.0), b=1.0 - c, a=1.0))
            return MarkerArray(markers=[m])

        def _publish_targets(self, st, ref, cmd):
            preview = self._marker(1, Marker.SPHERE, 0.20, (1.0, 0.0, 1.0, 1.0))
            foot = self._marker(2, Marker.SPHERE, 0.12, (0.0, 1.0, 1.0, 1.0))
            text = self._marker(3, Marker.TEXT_VIEW_FACING, 0.22, (1.0, 1.0, 1.0, 1.0))
            preview.pose.position.x, preview.pose.position.y = ref.tx, ref.ty
            foot.pose.position.x, foot.pose.position.y = st.fx, st.fy
            text.pose.position.x, text.pose.position.y, text.pose.position.z = st.xr, st.yr, 0.6
            text.text = (f'v {st.v:.2f} cmd {cmd.speed:.2f} m/s | d {st.d:+.2f} m '
                         f'e_psi {math.degrees(st.e_psi):+.0f} deg | steer {cmd.steer:+.2f}')
            markers = [preview, foot, text]
            av = self.core.avoid
            if av is not None:                                      # [3] obstacles (red) + shifted target (orange)
                text.text += f'\nd_ref {ref.d_ref:+.2f} m | obstacles {len(av.obstacles)}{" | AVOID" if av.active else ""}'
                c, sn = math.cos(st.psi_path), math.sin(st.psi_path)
                target = self._marker(4, Marker.SPHERE, 0.16, (1.0, 0.5, 0.0, 1.0))
                target.pose.position.x, target.pose.position.y = st.fx - ref.d_ref * sn, st.fy + ref.d_ref * c
                markers.append(target)
                for k in range(8):                                  # fixed ids: unused ones are hidden
                    m = self._marker(10 + k, Marker.CYLINDER, 0.1, (1.0, 0.1, 0.1, 0.7))
                    if k < len(av.obstacles):
                        ob = av.obstacles[k]
                        m.pose.position.x, m.pose.position.y = ob[4], ob[5]
                        m.scale.x = m.scale.y = 2.0 * max(ob[2], ob[3])
                        m.scale.z = 0.3
                    else:
                        m.action = Marker.DELETE
                    markers.append(m)
            self.viz_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    if rclpy is None:
        raise SystemExit('ROS 2 (rclpy) not found: source /opt/ros/humble/setup.bash and the workspace first')
    rclpy.init(args=args)
    node = Bv3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.drive_pub.publish(AckermannDriveStamped())            # stop the car on exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()