#!/usr/bin/env python3
"""make_track.py: turn a closed centreline into frenet_baseline track folders the LQR (frenet_bv3) can follow.

    python3 make_track.py <centreline.csv: x,y per row> --name ethzmobil --out <frenet_baseline>/waypoints
        [--scale 1.0] [--half-width 0.8] [--dr-ay 1.5 --dr-vmax 2.0] [--ay 3.1 --vmax 4.0]

Frame: the origin (0, 0) is the MIDDLE OF THE LONGEST STRAIGHT and +x is the driving direction there, so a
dead-reckoning run that starts at the origin facing +x starts on the line. Writes two folders, one per direction:
    <name>_cw/  and  <name>_ccw/   each with
        centerline.csv       x_m, y_m, w_left_m, w_right_m
        raceline_dr.csv      9-column generator format, gentle speed profile for the first (dead-reckoning) laps
        raceline_stable.csv  9-column, normal speed profile (centreline geometry; no racing-line optimisation)
        report.md
9-column format (frenet_baseline): x_m, y_m, s_m, psi_rad, kappa_radpm, vx_mps, ax_mps2, w_left_m, w_right_m, 0.1 m step.
"""
import argparse
import os

import numpy as np
from scipy.interpolate import splev, splprep

STEP = 0.1
STRAIGHT_KAPPA = 0.12      # |kappa| below this counts as straight [1/m] (radius > 8.3 m)


def resample_closed(x, y, smooth):
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-6:
        x, y = x[:-1], y[:-1]
    tck, _ = splprep([x, y], s=smooth * len(x), per=True)
    dense = np.linspace(0, 1, 20000, endpoint=False)
    xd, yd = splev(dense, tck)
    seg = np.hypot(np.diff(np.r_[xd, xd[0]]), np.diff(np.r_[yd, yd[0]]))
    s = np.r_[0.0, np.cumsum(seg)]
    length = s[-1]
    n = int(round(length / STEP))
    u = np.interp(np.linspace(0, length, n, endpoint=False), s, np.r_[dense, 1.0])
    xs, ys = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    w = 5                                                # curvature averaged over 0.5 m (circular)
    kappa = np.convolve(np.r_[kappa[-w:], kappa, kappa[:w]], np.ones(2 * w + 1) / (2 * w + 1), 'valid')
    return np.asarray(xs), np.asarray(ys), np.asarray(kappa), length


def recentre(x, y, kappa):
    """Rotate the arrays so index 0 is the middle of the longest straight, then move it to (0,0) heading +x."""
    k_s = np.convolve(np.r_[kappa[-10:], kappa, kappa[:10]], np.ones(21) / 21, 'valid')
    straight = np.abs(k_s) < STRAIGHT_KAPPA
    if straight.all() or not straight.any():
        mid = 0
        run = 0
    else:
        start = int(np.argmin(straight))          # rotate so the array starts on a curve: runs do not wrap
        st = np.roll(straight, -start)
        best, cur, best_end = 0, 0, 0
        for i, v in enumerate(st):
            cur = cur + 1 if v else 0
            if cur > best:
                best, best_end = cur, i
        mid = (start + best_end - best // 2) % len(x)
        run = best
    x, y, kappa = (np.roll(a, -mid) for a in (x, y, kappa))
    psi0 = np.arctan2(y[1] - y[-1], x[1] - x[-1])
    c, s = np.cos(-psi0), np.sin(-psi0)
    x0, y0 = x[0], y[0]
    xr = c * (x - x0) - s * (y - y0)
    yr = s * (x - x0) + c * (y - y0)
    return xr, yr, kappa, run * STEP


def speed_profile(kappa, ay, vmax, a_acc=2.0, a_dec=3.0):
    v = np.minimum(vmax, np.sqrt(ay / np.maximum(np.abs(kappa), 1e-4)))
    n = len(v)
    for _ in range(2):                                   # closed loop: two sweeps settle the wrap
        for i in range(1, 2 * n):                        # forward: acceleration limit
            a, b = (i - 1) % n, i % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * a_acc * STEP))
        for i in range(2 * n - 2, -1, -1):               # backward: braking limit
            a, b = (i + 1) % n, i % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * a_dec * STEP))
    ax = (np.roll(v, -1) ** 2 - v ** 2) / (2 * STEP)
    return v, ax


def write(folder, x, y, kappa, hw, ay_dr, vmax_dr, ay, vmax, src, scale, straight_len):
    os.makedirs(folder, exist_ok=True)
    n = len(x)
    s = np.arange(n) * STEP
    psi = np.unwrap(np.arctan2(np.roll(y, -1) - np.roll(y, 1), np.roll(x, -1) - np.roll(x, 1)))
    w = np.full(n, hw)
    np.savetxt(os.path.join(folder, 'centerline.csv'), np.column_stack([x, y, w, w]), delimiter=',', fmt='%.4f',
               header='x_m,y_m,w_left_m,w_right_m')
    lines = {}
    for fname, a_y, v_max in (('raceline_dr.csv', ay_dr, vmax_dr), ('raceline_stable.csv', ay, vmax)):
        v, ax = speed_profile(kappa, a_y, v_max)
        np.savetxt(os.path.join(folder, fname), np.column_stack([x, y, s, psi, kappa, v, ax, w, w]),
                   delimiter=',', fmt='%.4f',
                   header='x_m,y_m,s_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,w_left_m,w_right_m')
        lines[fname] = (v, a_y, v_max)
    turn = 'clockwise' if np.sum(kappa) * STEP < 0 else 'counter-clockwise'
    rmin = 1.0 / np.abs(kappa).max()
    with open(os.path.join(folder, 'report.md'), 'w') as f:
        f.write(f'# {os.path.basename(folder)}\n\n')
        f.write(f'- Source: `{src}` x {scale:g}\n')
        f.write(f'- Size {np.ptp(x):.2f} x {np.ptp(y):.2f} m, lap {n * STEP:.2f} m, driven {turn}\n')
        f.write(f'- Tightest centreline radius {rmin:.2f} m (F1TENTH minimum turn ~0.93 m at STEER_LIMIT 0.40)\n')
        f.write(f'- Origin (0, 0) = middle of the longest straight ({straight_len:.1f} m of |kappa| < '
                f'{STRAIGHT_KAPPA} 1/m), +x = driving direction. Dead reckoning starts here.\n')
        f.write(f'- Half width {hw:g} m each side (placeholder until the physical track is built)\n')
        for fname, (v, a_y, v_max) in lines.items():
            f.write(f'- {fname}: a_y {a_y:g} m/s^2, v_max {v_max:g} m/s -> {v.min():.2f}..{v.max():.2f} m/s, '
                    f'lap ~{np.sum(STEP / v):.1f} s\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('centreline')
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--smooth', type=float, default=0.0,
                    help='spline smoothing per point [m^2]; 0 = interpolate the given points (smoothing cuts corners '
                         'and makes them TIGHTER)')
    ap.add_argument('--half-width', type=float, default=0.8)
    ap.add_argument('--dr-ay', type=float, default=1.0)
    ap.add_argument('--dr-vmax', type=float, default=0.7)
    ap.add_argument('--ay', type=float, default=3.1)
    ap.add_argument('--vmax', type=float, default=4.0)
    a = ap.parse_args()
    xy = np.loadtxt(a.centreline, delimiter=',', comments='#', ndmin=2)
    x, y = xy[:, 0] * a.scale, xy[:, 1] * a.scale
    for direction in ('cw', 'ccw'):
        xs, ys, k, _ = resample_closed(x, y, a.smooth)
        if (np.sum(k) < 0) != (direction == 'cw'):        # reverse to get the requested direction
            xs, ys, k, _ = resample_closed(x[::-1], y[::-1], a.smooth)
        xr, yr, kr, straight_len = recentre(xs, ys, k)
        folder = os.path.join(a.out, f'{a.name}_{direction}')
        write(folder, xr, yr, kr, a.half_width, a.dr_ay, a.dr_vmax, a.ay, a.vmax,
              os.path.basename(a.centreline), a.scale, straight_len)
        print(open(os.path.join(folder, 'report.md')).read())


if __name__ == '__main__':
    main()
