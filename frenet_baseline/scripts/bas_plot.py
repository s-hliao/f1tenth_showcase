#!/usr/bin/env python3
"""
Post-run plot for the bv1 + BaS demo (one bag per speed).

    python3 bas_plot.py bags/bas_v3.0 --waypoints aims_centerline.csv
    python3 bas_plot.py bags/bas_v2.0 bags/bas_v3.0 bags/bas_v4.0 --waypoints aims_centerline.csv

Reads /bas/debug (Float32MultiArray, one per control cycle) and /pf/pose/odom from the bag.
Writes <bag>_bas.png next to each bag and prints one line per obstacle pass.
With several bags it also writes bas_speeds.png (all runs side by side).
Needs: pip install rosbags matplotlib numpy  (no ROS needed, runs on the laptop too).
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

FIELDS = ['s', 'd', 'd_ref', 'e_psi', 'v', 'speed_cmd', 'steer', 'n_obstacles', 'active',
          'bt_ahead', 'h_obs_min', 'speed_scale', 'obs_ds', 'obs_d', 'obs_side', 'guard_reject']
F = {k: i for i, k in enumerate(FIELDS)}
CAR_HALF_WIDTH = 0.15


def read_bag(path, debug_topic, odom_topic):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    dbg, odo = [], []
    with AnyReader([Path(path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        conns = [c for c in r.connections if c.topic in (debug_topic, odom_topic)]
        if not any(c.topic == debug_topic for c in conns):
            sys.exit(f'{path}: no {debug_topic} in the bag (was bas_enable true?)')
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            if c.topic == debug_topic:
                row = list(m.data)[:len(FIELDS)]
                dbg.append([t * 1e-9] + row + [0.0] * (len(FIELDS) - len(row)))
            else:
                p, q = m.pose.pose.position, m.pose.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                odo.append([t * 1e-9, p.x, p.y, yaw, m.twist.twist.linear.x])
    dbg, odo = np.array(dbg), np.array(odo) if odo else np.zeros((0, 5))
    t0 = dbg[0, 0]
    dbg[:, 0] -= t0
    if len(odo):
        odo[:, 0] -= t0
    return dbg, odo


def load_line(csv):
    a = np.loadtxt(os.path.expanduser(csv), delimiter=',', comments='#')
    xy = a[:, :2]
    seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    L = s[-1] + seg[-1]
    nxt = np.roll(xy, -1, axis=0)
    psi = np.arctan2(nxt[:, 1] - xy[:, 1], nxt[:, 0] - xy[:, 0])
    wl = a[:, 2] if a.shape[1] >= 4 else None
    wr = a[:, 3] if a.shape[1] >= 4 else None
    return xy, s, L, psi, wl, wr


def frenet_to_xy(line, s, d):
    xy, ss, L, psi, _, _ = line
    s = np.mod(s, L)
    i = np.clip(np.searchsorted(ss, s, side='right') - 1, 0, len(ss) - 1)
    ds = s - ss[i]
    x = xy[i, 0] + ds * np.cos(psi[i]) - d * np.sin(psi[i])
    y = xy[i, 1] + ds * np.sin(psi[i]) + d * np.cos(psi[i])
    return x, y


def passes(dbg, gap=0.5):
    """Contiguous BaS-active stretches (gaps shorter than `gap` s are merged)."""
    t = dbg[:, 0]
    idx = np.nonzero(dbg[:, 1 + F['active']] > 0.5)[0]
    out = []
    for k in idx:
        if out and t[k] - t[out[-1][1]] < gap:
            out[-1][1] = k
        else:
            out.append([k, k])
    return [tuple(p) for p in out]


def col(dbg, k):
    return dbg[:, 1 + F[k]]


def summarize(name, dbg, L):
    print(f'\n{name}: {dbg[-1, 0]:.1f} s, {len(dbg)} cycles, '
          f'speed cmd max {col(dbg, "speed_cmd").max():.2f} m/s, actual max {col(dbg, "v").max():.2f} m/s')
    rows = []
    for k, (i, j) in enumerate(passes(dbg)):
        sl = slice(i, j + 1)
        dd = col(dbg, 'd')[sl] - col(dbg, 'obs_d')[sl]
        ods = col(dbg, 'obs_ds')[sl]
        near = np.abs(ods) < 0.5                        # car alongside the obstacle
        lat = np.abs(dd[near]).min() if near.any() else float('nan')
        side = col(dbg, 'obs_side')[sl]
        side = side[side != 0]
        side_s = ('left' if np.median(side) > 0 else 'right') if len(side) else '-'
        r = dict(n=k + 1, t=dbg[i, 0], s=col(dbg, 's')[i] % L, dur=dbg[j, 0] - dbg[i, 0],
                 v_in=col(dbg, 'v')[max(i - 1, 0)], v_min=col(dbg, 'v')[sl].min(),
                 dref=np.abs(col(dbg, 'd_ref')[sl]).max(), side=side_s, lat=lat,
                 gap=lat - CAR_HALF_WIDTH if near.any() else float('nan'),
                 hmin=col(dbg, 'h_obs_min')[sl].min())
        rows.append(r)
        print(f'  pass {r["n"]}: t {r["t"]:5.1f} s  s {r["s"]:5.1f} m  {r["dur"]:.1f} s  '
              f'v in {r["v_in"]:.2f} -> min {r["v_min"]:.2f} m/s  |d_ref| max {r["dref"]:.2f} m  side {r["side"]}  '
              f'centre offset alongside {r["lat"]:.2f} m  h_min {r["hmin"]:.2f}')
    if not rows:
        print('  no BaS activation in this run')
    return rows


def plot_run(name, dbg, odo, line, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xy, ss, L, psi, wl, wr = line
    t = dbg[:, 0]
    act = col(dbg, 'active') > 0.5
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1])

    # 1. map view
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(xy[:, 0], xy[:, 1], color='0.7', lw=1, label='centerline')
    if wl is not None:
        lx, ly = frenet_to_xy(line, ss, wl)
        rx, ry = frenet_to_xy(line, ss, -wr)
        ax.plot(lx, ly, 'k', lw=0.8)
        ax.plot(rx, ry, 'k', lw=0.8)
    if len(odo):
        ax.plot(odo[:, 1], odo[:, 2], color='tab:blue', lw=1.2, label='driven path (/pf/pose/odom)')
    px, py = frenet_to_xy(line, col(dbg, 's'), col(dbg, 'd'))
    ax.scatter(px[act], py[act], s=4, c='tab:red', zorder=3, label='BaS active')
    tx, ty = frenet_to_xy(line, col(dbg, 's'), col(dbg, 'd_ref'))
    ax.plot(tx, ty, color='tab:orange', lw=0.8, alpha=0.8, label='shifted reference d_ref')
    has = col(dbg, 'obs_ds') < 90
    if has.any():
        ox, oy = frenet_to_xy(line, col(dbg, 's')[has] + col(dbg, 'obs_ds')[has], col(dbg, 'obs_d')[has])
        ax.scatter(ox, oy, s=6, c='tab:purple', alpha=0.25, zorder=2, label='detected obstacle (each cycle)')
    ax.plot(xy[0, 0], xy[0, 1], 'go')
    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=8)
    ax.set_title(f'{name}: path and obstacles')

    def shade(a):
        on = False
        for k in range(len(t)):
            if act[k] and not on:
                t0, on = t[k], True
            elif not act[k] and on:
                a.axvspan(t0, t[k], color='tab:red', alpha=0.12, lw=0)
                on = False
        if on:
            a.axvspan(t0, t[-1], color='tab:red', alpha=0.12, lw=0)

    # 2. lateral: d and d_ref
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(t, col(dbg, 'd'), label='d (car offset)')
    a1.plot(t, col(dbg, 'd_ref'), label='d_ref (barrier state)')
    od = np.where(has & (np.abs(col(dbg, 'obs_ds')) < 3.0), col(dbg, 'obs_d'), np.nan)
    a1.plot(t, od, '.', ms=2, color='tab:purple', label='obstacle d (within 3 m)')
    shade(a1)
    a1.set_ylabel('lateral [m]')
    a1.legend(fontsize=8, loc='upper right')
    a1.grid(alpha=0.3)

    # 3. speed
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(t, col(dbg, 'speed_cmd'), label='speed command')
    a2.plot(t, col(dbg, 'v'), label='speed (odom)')
    a2b = a2.twinx()
    a2b.plot(t, col(dbg, 'speed_scale'), color='tab:green', lw=0.8, label='BaS speed scale')
    a2b.set_ylim(0, 1.05)
    a2b.set_ylabel('scale')
    shade(a2)
    a2.set_ylabel('speed [m/s]')
    a2.legend(fontsize=8, loc='lower left')
    a2b.legend(fontsize=8, loc='lower right')
    a2.grid(alpha=0.3)

    # 4. barrier
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    a3.plot(t, col(dbg, 'bt_ahead'), color='tab:red', label='shifted barrier B~ (obstacles ahead)')
    a3.set_ylabel('B~')
    a3b = a3.twinx()
    a3b.plot(t, np.minimum(col(dbg, 'h_obs_min'), 10), color='tab:gray', lw=0.8, label='h min (safety distance)')
    a3b.set_ylabel('h')
    shade(a3)
    a3.set_xlabel('time [s]  (red band = BaS active)')
    a3.legend(fontsize=8, loc='upper left')
    a3b.legend(fontsize=8, loc='upper right')
    a3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  plot: {out}')


def plot_compare(runs, line, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    L = line[2]
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, dbg in runs:
        s = col(dbg, 's') % L
        brk = np.where(np.abs(np.diff(s)) > L / 2)[0] + 1      # split at lap wrap
        for k, part in enumerate(np.split(np.arange(len(s)), brk)):
            lab = name if k == 0 else None
            l0, = axs[0].plot(s[part], col(dbg, 'd')[part], lw=1, label=lab) if k == 0 else \
                axs[0].plot(s[part], col(dbg, 'd')[part], lw=1, color=l0.get_color())
            axs[1].plot(s[part], col(dbg, 'v')[part], lw=1, color=l0.get_color(), label=lab)
        has = (np.abs(col(dbg, 'obs_ds')) < 0.3)
        axs[0].scatter((s + col(dbg, 'obs_ds'))[has] % L, col(dbg, 'obs_d')[has], s=30, marker='s',
                       c='k', alpha=0.05)
    axs[0].set_ylabel('car offset d [m]')
    axs[0].set_title('all speeds: lateral offset (black squares = obstacles)')
    axs[1].set_ylabel('speed [m/s]')
    axs[1].set_xlabel('distance along the track s [m]')
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'\ncomparison plot: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--waypoints', required=True, help='same centerline CSV the node used')
    ap.add_argument('--debug_topic', default='/bas/debug')
    ap.add_argument('--odom_topic', default='/pf/pose/odom')
    a = ap.parse_args()
    line = load_line(a.waypoints)
    runs = []
    for b in a.bags:
        b = b.rstrip('/')
        name = os.path.basename(b)
        dbg, odo = read_bag(b, a.debug_topic, a.odom_topic)
        summarize(name, dbg, line[2])
        plot_run(name, dbg, odo, line, b + '_bas.png')
        runs.append((name, dbg))
    if len(runs) > 1:
        plot_compare(runs, line, os.path.join(os.path.dirname(a.bags[0].rstrip('/')) or '.', 'bas_speeds.png'))


if __name__ == '__main__':
    main()
