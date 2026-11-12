#!/usr/bin/env python3
"""
Barrier-state (BaS) obstacle avoidance layer for the bv1 Frenet pursuit node.
ROS-free: the node feeds it the car's Frenet state, the pose and the latest scan; it returns d_ref
(sideways shift of the tracking target, 0 = the line itself) and a speed scale.

Same maths and fixes as the bv2 layer that was tested in sim and on the venue bags (2026-09-30):
    [3a] LiDAR -> obstacles in Frenet (s, d) against the line ahead, wall points dropped with the
         per-point wall distances, clusters of neighbouring beams, persistence filter,
         RANGE_MIN (car's own returns), BIRTH (new obstacle must first be seen far away),
         EDGE (obstacle must stand in front of its background)
    [3b] barrier functions: walls (planned target + predicted front bumper) and obstacles (ellipse
         stretched ahead by T_LOOK * v); shifted barrier B~ = 1/h - 1/h_act, zero when far
    [3c] barrier state -> d_ref:  d_ref_dot = sum side * max(0, K_B * B~ + K_DB * z_dot) - K_RELAX * d_ref
         pass side chosen per obstacle (larger gap to the wall), kept while that obstacle is in range
Frenet convention (bv1): s along the line, d positive to the LEFT, e_psi = yaw - line heading.
Cost: barriers O(K) per cycle (K = obstacles); detection O(M*W) numpy once per NEW scan.
"""
import math
from dataclasses import dataclass, fields

import numpy as np


@dataclass
class BasParams:
    # car footprint, measured from the pose point (bv1 pose = /pf/pose/odom = laser point)
    CAR_HALF_WIDTH: float = 0.15
    CAR_FRONT: float = 0.414                 # [m] kept from the tested config: extra look-ahead for the steering delay
    CAR_REAR: float = 0.40                   # [m] laser -> rear bumper
    BAS_MARGIN: float = 0.10                 # [m] extra clearance
    TRACK_HALF_W: float = 0.8                # [m] walls when the CSV has no width columns
    # what the barriers act on
    BAS_WALLS: bool = False                  # wall barriers on the plan / front bumper. Off: in sim they caused wall
                                             # contacts under heavy pose noise; bv1 tracking keeps the car off the walls
    BAS_OBSTACLES: bool = True               # LiDAR obstacle barriers
    # barrier shape
    H_ACT_WALL: float = 0.10                 # [m] wall barrier switches on below this clearance
    H_ACT_OBS: float = 4.0                   # obstacle barrier switches on below this ellipse level (~4 m out)
    T_LOOK: float = 0.45                     # [s] obstacle ellipse grows ahead by T_LOOK * v (0.8 doubled false
                                             # activations on the real bags, same sim result)
    T_LOOK_WALL: float = 0.30                # [s] front bumper predicted this far ahead for the wall check
    WALL_PLAN_LA: float = 0.6                # [m] the plan is checked against the walls this far ahead (bv1 ff preview)
    H_MIN: float = 1e-3
    B_MAX: float = 200.0
    ZDOT_MAX: float = 500.0
    DENOM_MIN: float = 0.1                   # floor for (1 - kappa*d)
    # barrier state -> d_ref
    K_B_WALL: float = 0.30
    K_DB_WALL: float = 0.0005
    K_B_OBS: float = 3.0
    K_DB_OBS: float = 0.10
    K_RELAX: float = 1.5                     # [1/s] pull back to the line
    DREF_RATE_MAX: float = 3.0               # [m/s]
    DREF_EPS: float = 0.005                  # [m] snap to exactly 0 when inactive
    GAP_HYST: float = 0.05
    H_SWITCH: float = 1000.0                 # pass side locked once chosen (1000 = never switches)
    # speed
    K_BV: float = 0.8                        # scale 1/(1 + K_BV * sum B~ ahead + K_BV_RETURN * |d_ref|)
    K_BV_RETURN: float = 0.3
    V_BAS_MIN: float = 1.5                   # [m/s] never slower than this while avoiding
    # LiDAR obstacle detection
    LASER_X: float = 0.0                     # [m] laser ahead of the pose point (0: bv1 pose IS the laser point)
    LASER_Y: float = 0.0
    SCAN_STRIDE: int = 2
    SCAN_FOV_DEG: float = 200.0
    RANGE_MIN: float = 0.30                  # [m] drops the car's own returns (0.13-0.19 m seen in the bags)
    OBS_RANGE: float = 6.0
    OBS_BEHIND: float = 0.6
    WALL_GATE: float = 0.15
    CLUSTER_GAP: float = 0.25
    MIN_PTS: int = 3
    OBS_R_MIN: float = 0.15
    OBS_PERSIST: int = 2
    OBS_CLEAR: int = 4
    OBS_BIRTH_RANGE: float = 1.5             # [BIRTH] 0 = off
    OBS_MATCH_DS: float = 1.0
    OBS_MATCH_DD: float = 0.5
    OBS_EDGE_JUMP: float = 0.3               # [EDGE] 0 = off
    OBS_EDGE_SIDES: int = 2


BAS_PARAM_NAMES = [f.name for f in fields(BasParams)]


class BasLine:
    """The line the car tracks, as numpy arrays (built once from the bv1 waypoints)."""

    def __init__(self, xy: np.ndarray, w_left: np.ndarray, w_right: np.ndarray):
        self.n = len(xy)
        self.x, self.y = xy[:, 0].astype(float), xy[:, 1].astype(float)
        d = np.roll(xy, -1, axis=0) - xy
        seg = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        self.length = float(self.s[-1] + seg[-1])
        self.psi = np.arctan2(d[:, 1], d[:, 0])
        self.cos, self.sin = np.cos(self.psi), np.sin(self.psi)
        self.wl, self.wr = np.asarray(w_left, float), np.asarray(w_right, float)
        self.ds = float(np.mean(seg))

    def index_at(self, s: float) -> int:
        """Segment index containing arc length s. O(log n)."""
        return int(np.searchsorted(self.s, s % self.length, side='right')) - 1


class BarrierLayer:
    def __init__(self, p: BasParams, line: BasLine):
        self.p, self.L = p, line
        self.d_ref = 0.0
        self.d_ref_rate = 0.0
        self.sides = []                      # [s, side] per obstacle in range: +1 pass left, -1 pass right
        self.scan = None
        self.scan_new = False
        self._beam_cache = None
        self.obstacles = []                  # confirmed: (s, d, r_s, r_d, x, y, psi, w_left, w_right)
        self._last_found = []
        self._hits = 0
        self._misses = 0
        self.obs_bt = []
        n_win = int(math.ceil((p.OBS_RANGE + p.LASER_X) / line.ds)) + 1
        self._win = np.arange(-int(math.ceil(p.OBS_BEHIND / line.ds)) - 2, n_win)
        # last cycle, for debug / telemetry
        self.speed_scale = 1.0
        self.active = False
        self.bt_ahead = 0.0
        self.h_obs_min = math.inf

    def set_scan(self, ranges, angle_min: float, angle_inc: float):
        self.scan = (ranges, angle_min, angle_inc)
        self.scan_new = True

    def step(self, x, y, yaw, s, d, e_psi, kappa, v, seg, dt):
        """One control cycle. (x, y, yaw) = pose point in map; seg = current line index. Returns (d_ref, speed_scale)."""
        p = self.p
        if p.BAS_OBSTACLES and self.scan_new and self.scan is not None:
            self.detect_obstacles(x, y, yaw, seg)
            self.scan_new = False
        push, bt_ahead, active = self.barriers(s, d, e_psi, kappa, v, seg)
        self.update_d_ref(push, bt_ahead, active, dt, seg)
        return self.d_ref, self.speed_scale

    # ---- [3a] LiDAR -> obstacles in Frenet (once per NEW scan) ----

    def detect_obstacles(self, x, y, yaw, seg):
        p, L = self.p, self.L
        ranges, a0, da = self.scan
        r_all = np.asarray(ranges, dtype=float)
        key = (len(r_all), a0, da)
        if self._beam_cache is None or self._beam_cache[0] != key:
            ang = a0 + da * np.arange(len(r_all))
            idx = np.where(np.abs(ang) <= math.radians(p.SCAN_FOV_DEG) / 2)[0][::p.SCAN_STRIDE]
            self._beam_cache = (key, idx, np.cos(ang[idx]), np.sin(ang[idx]))
        _, idx, ca, sa = self._beam_cache
        r = r_all[idx]
        keep = np.isfinite(r) & (r > p.RANGE_MIN) & (r < p.OBS_RANGE)
        if not keep.any():
            self._last_found = []
            self._confirm([])
            return
        r, ca, sa, bi = r[keep], ca[keep], sa[keep], idx[keep]

        lx, ly = p.LASER_X + r * ca, p.LASER_Y + r * sa              # laser -> pose-point frame
        c, sn = math.cos(yaw), math.sin(yaw)
        px, py = x + c * lx - sn * ly, y + sn * lx + c * ly            # -> map

        win = (seg + self._win) % L.n                                  # line points near and ahead of the car
        d2 = (px[:, None] - L.x[win][None, :]) ** 2 + (py[:, None] - L.y[win][None, :]) ** 2
        k = win[np.argmin(d2, axis=1)]
        ex, ey = px - L.x[k], py - L.y[k]
        ps = L.s[k] + ex * L.cos[k] + ey * L.sin[k]
        pd = -ex * L.sin[k] + ey * L.cos[k]

        inside = (pd < L.wl[k] - p.WALL_GATE) & (pd > -L.wr[k] + p.WALL_GATE)
        if inside.sum() < p.MIN_PTS:
            self._last_found = []
            self._confirm([])
            return
        px, py, ps, pd, rr, bi = px[inside], py[inside], ps[inside], pd[inside], r[inside], bi[inside]

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
            i = L.index_at(s_c)
            if p.OBS_BIRTH_RANGE > 0.0 and float(rr[a:b].min()) < p.OBS_BIRTH_RANGE \
                    and not self._continues(s_c, d_c):
                continue                                               # [BIRTH] appeared too close
            if p.OBS_EDGE_JUMP > 0.0 and \
                    self._edge_jump(r_all, int(bi[a]), float(rr[a]), -1) + \
                    self._edge_jump(r_all, int(bi[b - 1]), float(rr[b - 1]), 1) < p.OBS_EDGE_SIDES:
                continue                                               # [EDGE] continuous with its background
            psi = float(L.psi[i])
            x_c = L.x[i] - d_c * math.sin(psi)
            y_c = L.y[i] + d_c * math.cos(psi)
            found.append((s_c, d_c, r_s, r_d, x_c, y_c, psi, float(L.wl[i]), float(L.wr[i])))
        self._last_found = found
        self._confirm(found)

    def _edge_jump(self, r_all, i, r_edge, step):
        """1 if one of the next 3 valid beams past the cluster edge reads farther than r_edge + OBS_EDGE_JUMP."""
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
                continue                                               # own body / no data
            seen += 1
        return 0

    def _continues(self, s_c, d_c):
        p, L = self.p, self.L
        half = 0.5 * L.length
        for ob in self.obstacles + self._last_found:
            ds = (s_c - ob[0] + half) % L.length - half
            if abs(ds) < p.OBS_MATCH_DS and abs(d_c - ob[1]) < p.OBS_MATCH_DD:
                return True
        return False

    def _confirm(self, found):
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

    # ---- [3b] barriers, evaluated on the PLANNED offset d_ref (the barrier state) ----

    def barriers(self, s, d, e_psi, kappa, v, seg):
        p, L = self.p, self.L
        q, q_dot = self.d_ref, self.d_ref_rate
        s_dot = v * math.cos(e_psi) / max(1.0 - kappa * d, p.DENOM_MIN)
        push, bt_ahead, active = 0.0, 0.0, False
        self.obs_bt = []
        self.h_obs_min = math.inf

        if p.BAS_WALLS:
            m = p.CAR_HALF_WIDTH + p.BAS_MARGIN
            i_t = L.index_at(s + p.WALL_PLAN_LA)
            reach = p.CAR_FRONT + p.T_LOOK_WALL * v
            i_f = L.index_at(s + reach)
            d_f = d + reach * math.sin(e_psi)
            d_f_dot = v * math.sin(e_psi)
            hl_plan, hl_car = L.wl[i_t] - m - q, L.wl[i_f] - m - d_f
            hr_plan, hr_car = q + L.wr[i_t] - m, d_f + L.wr[i_f] - m
            h_l, hl_dot = (hl_plan, -q_dot) if hl_plan <= hl_car else (hl_car, -d_f_dot)
            h_r, hr_dot = (hr_plan, q_dot) if hr_plan <= hr_car else (hr_car, d_f_dot)
            for h, h_dot, side in ((h_l, hl_dot, -1.0), (h_r, hr_dot, 1.0)):
                bt, zdot = self._shifted_barrier(h, h_dot, p.H_ACT_WALL)
                if bt > 0.0:
                    push += side * max(0.0, p.K_B_WALL * bt + p.K_DB_WALL * zdot)
                    active = True

        half, sides_now = 0.5 * L.length, []
        for (s_j, d_j, r_s, r_d, _, _, _, wl_j, wr_j) in self.obstacles:
            ds = (s - s_j + half) % L.length - half                   # < 0: obstacle ahead
            if ds < -p.OBS_RANGE or ds > p.OBS_BEHIND:
                self.obs_bt.append(0.0)
                continue
            ent = None
            for e in self.sides:
                if abs((s_j - e[0] + half) % L.length - half) < p.OBS_MATCH_DS:
                    ent = e
                    break
            if ent is None:
                ent = [s_j, 0]
            ent[0] = s_j
            if ent not in sides_now:
                sides_now.append(ent)
            a = r_s + p.BAS_MARGIN + (p.CAR_FRONT + p.T_LOOK * v if ds < 0.0 else p.CAR_REAR)
            b = r_d + p.CAR_HALF_WIDTH + p.BAS_MARGIN
            gs, gd = 2.0 * ds / (a * a), 2.0 * (q - d_j) / (b * b)
            h = (ds / a) ** 2 + ((q - d_j) / b) ** 2 - 1.0
            self.h_obs_min = min(self.h_obs_min, h)
            bt, zdot = self._shifted_barrier(h, gs * s_dot + gd * q_dot, p.H_ACT_OBS)
            self.obs_bt.append(bt)
            if bt <= 0.0:
                continue
            side = self._pass_side(ent, h, d_j, r_d, wl_j, wr_j)
            push += side * max(0.0, p.K_B_OBS * bt + p.K_DB_OBS * zdot)
            active = True
            if ds < 0.0:
                bt_ahead += bt
        self.sides = sides_now
        return push, bt_ahead, active

    def _shifted_barrier(self, h, h_dot, h_act):
        p = self.p
        if h >= h_act:
            return 0.0, 0.0
        hc = max(h, p.H_MIN)
        bt = min(1.0 / hc - 1.0 / h_act, p.B_MAX)
        if bt >= p.B_MAX or h <= p.H_MIN:
            return bt, 0.0
        zdot = -h_dot / (hc * hc)
        return bt, max(-p.ZDOT_MAX, min(p.ZDOT_MAX, zdot))

    def _pass_side(self, ent, h, d_j, r_d, wl_j, wr_j):
        """Larger gap between the obstacle and the wall; chosen when this obstacle's barrier first switches on."""
        p = self.p
        gap_l = wl_j - (d_j + r_d)
        gap_r = (d_j - r_d) + wr_j
        if ent[1] == 0:
            ent[1] = 1 if gap_l >= gap_r else -1
        elif h > p.H_SWITCH:
            if ent[1] == 1 and gap_r > gap_l + p.GAP_HYST:
                ent[1] = -1
            elif ent[1] == -1 and gap_l > gap_r + p.GAP_HYST:
                ent[1] = 1
        return ent[1]

    # ---- [3c] barrier state -> d_ref ----

    def update_d_ref(self, push, bt_ahead, active, dt, seg):
        p, L = self.p, self.L
        rate = push - p.K_RELAX * self.d_ref
        rate = max(-p.DREF_RATE_MAX, min(p.DREF_RATE_MAX, rate))
        d_ref = self.d_ref + rate * dt
        self.d_ref_rate = rate
        hi = L.wl[seg] - p.CAR_HALF_WIDTH - p.BAS_MARGIN
        lo = -L.wr[seg] + p.CAR_HALF_WIDTH + p.BAS_MARGIN
        d_ref = min(hi, max(lo, d_ref)) if lo < hi else 0.0
        if not active and abs(d_ref) < p.DREF_EPS:
            d_ref = 0.0
            self.d_ref_rate = 0.0
        self.d_ref = d_ref
        self.speed_scale = 1.0 / (1.0 + p.K_BV * bt_ahead + p.K_BV_RETURN * abs(d_ref))
        self.active = active
        self.bt_ahead = bt_ahead
