#!/usr/bin/env python3
"""
Post-run speed report from a /telemetry bag (manual trigger, never on the car).

    python3 telemetry_report.py <bag_folder> [--plot]

Answers "is the car reaching its top speed?":
    - per lap: lap time, max measured v, max commanded speed, top-speed reach %
    - overall: where the max speed happened, speed tracking error on straights
      and in corners, % of cycles speed-limited
    - --plot: v, speed_cmd, v_ref and d along s for every lap -> <bag>/speed_report.png

Needs: ROS 2 sourced (rosbag2_py) and the workspace sourced (race_msgs).
"""
import argparse
import os
import sys

import numpy as np
import yaml

FIELDS = ('t', 's', 'd', 'kappa', 'v', 'v_ref', 'speed_cmd', 'flags', 'd_ref', 'avoid', 't_total_ms')
FLAG_SPEED_LIMITED = 16


# ---------------- load (rosbag2 -> numpy columns) ----------------

def load(bag: str) -> dict:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from race_msgs.msg import Telemetry

    meta = yaml.safe_load(open(os.path.join(bag, 'metadata.yaml')))
    storage_id = meta['rosbag2_bagfile_information'].get('storage_identifier', 'sqlite3')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage_id),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/telemetry']))

    rows = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        m = deserialize_message(data, Telemetry)
        t = m.pose_stamp.sec + m.pose_stamp.nanosec * 1e-9       # pose is the master timeline
        rows.append((t, m.s, m.d, m.kappa, m.v, m.v_ref, m.speed_cmd, m.flags,
                     m.d_ref, float(m.avoid_active), m.t_total_ms))
    if not rows:
        sys.exit(f'No /telemetry messages in {bag}. Was TELEMETRY_ENABLE true and race_msgs built?')
    a = np.array(rows, dtype=float)
    return {k: a[:, i] for i, k in enumerate(FIELDS)}


# ---------------- analyse (pure numpy; also used by offline tests) ----------------

def split_laps(tel: dict):
    """Lap boundaries where s wraps (drops by more than half the track). Only full laps."""
    s = tel['s']
    track_len = float(np.max(s))
    cross = np.where(np.diff(s) < -0.5 * track_len)[0] + 1
    return [(cross[k], cross[k + 1]) for k in range(len(cross) - 1)], track_len


def analyse(tel: dict) -> str:
    t, v, cmd, vref, k = tel['t'], tel['v'], tel['speed_cmd'], tel['v_ref'], tel['kappa']
    laps, track_len = split_laps(tel)
    rate = 1.0 / np.median(np.diff(t)) if len(t) > 1 else float('nan')
    out = [f'Samples {len(t)}  duration {t[-1] - t[0]:.1f} s  telemetry {rate:.0f} Hz  '
           f'track {track_len:.1f} m  full laps {len(laps)}', '']

    out.append(' lap |  time s | max v | max cmd | reach % | mean v')
    out.append('-----+---------+-------+---------+---------+-------')
    for n, (i0, i1) in enumerate(laps, 1):
        lv, lc = v[i0:i1], cmd[i0:i1]
        out.append(f'{n:4d} | {t[i1] - t[i0]:7.2f} | {lv.max():5.2f} | {lc.max():7.2f} | '
                   f'{100 * lv.max() / max(lc.max(), 1e-6):7.1f} | {lv.mean():5.2f}')
    if not laps:
        out.append('   (no full lap recorded: drive past the start line twice)')

    i_max = int(np.argmax(v))
    top_cmd = cmd.max()
    at_top = cmd >= 0.98 * top_cmd                         # cycles where the top speed is commanded
    straight, corner = np.abs(k) < 0.1, np.abs(k) >= 0.1
    err = v - cmd
    out += ['',
            f'Max measured v   {v.max():.2f} m/s at s = {tel["s"][i_max]:.1f} m',
            f'Max commanded    {top_cmd:.2f} m/s   (max raceline v_ref x scale: {vref.max():.2f} m/s)',
            f'Top-speed reach  {100 * v.max() / max(top_cmd, 1e-6):.1f} %  '
            f'| time with v >= 95% of cmd while top speed is commanded: '
            f'{100 * np.mean(v[at_top] >= 0.95 * cmd[at_top]):.0f} %',
            f'Speed error v - cmd, median [p05, p95]:  straights {np.median(err[straight]):+.2f} '
            f'[{np.percentile(err[straight], 5):+.2f}, {np.percentile(err[straight], 95):+.2f}]  '
            f'corners {np.median(err[corner]):+.2f} '
            f'[{np.percentile(err[corner], 5):+.2f}, {np.percentile(err[corner], 95):+.2f}] m/s'
            if straight.any() and corner.any() else '',
            f'Speed-limited cycles {100 * np.mean((tel["flags"].astype(int) & FLAG_SPEED_LIMITED) > 0):.1f} %']
    out += ['', lateral_spread(tel, laps, track_len)]
    if 'avoid' in tel and tel['avoid'].any():
        out.append(f'Avoidance: active {100 * tel["avoid"].mean():.1f} % of cycles, '
                   f'max |d_ref| {np.abs(tel["d_ref"]).max():.2f} m')
    if 't_total_ms' in tel:
        out.append(f'Compute per cycle: median {np.median(tel["t_total_ms"]):.2f} ms, '
                   f'p99 {np.percentile(tel["t_total_ms"], 99):.2f} ms')
    return '\n'.join(x for x in out if x is not None)


def lateral_spread(tel: dict, laps, track_len: float, bin_m: float = 0.5) -> str:
    """Gate metric: std of d across laps in each s-bin, averaged over the track (needs >= 2 laps).
    Also RMS of d (tracking error to the raceline, avoidance cycles excluded)."""
    d = tel['d']
    keep = tel['avoid'] < 0.5 if 'avoid' in tel else np.ones(len(d), bool)
    rms = float(np.sqrt(np.mean(d[keep] ** 2))) if keep.any() else float('nan')
    if len(laps) < 2:
        return f'RMS lateral error {rms:.3f} m | lateral spread needs >= 2 full laps'
    nb = int(np.ceil(track_len / bin_m))
    per_lap = np.full((len(laps), nb), np.nan)
    for k, (i0, i1) in enumerate(laps):
        idx = np.minimum((tel['s'][i0:i1] / bin_m).astype(int), nb - 1)
        m = keep[i0:i1]
        sums = np.bincount(idx[m], weights=d[i0:i1][m], minlength=nb)
        cnt = np.bincount(idx[m], minlength=nb)
        with np.errstate(invalid='ignore', divide='ignore'):
            per_lap[k] = sums / cnt
    spread = np.nanmean(np.nanstd(per_lap, axis=0))
    return (f'Lateral spread (std of d across {len(laps)} laps per {bin_m} m bin, averaged) {spread:.3f} m'
            f' | RMS lateral error {rms:.3f} m')


def plot(tel: dict, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    laps, _ = split_laps(tel)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    spans = laps if laps else [(0, len(tel['s']))]
    for n, (i0, i1) in enumerate(spans, 1):
        ax1.plot(tel['s'][i0:i1], tel['v'][i0:i1], lw=1, label=f'v lap {n}')
        ax2.plot(tel['s'][i0:i1], tel['d'][i0:i1], lw=1)
    i0, i1 = spans[0]
    ax1.plot(tel['s'][i0:i1], tel['speed_cmd'][i0:i1], 'k--', lw=1.2, label='speed_cmd')
    ax1.plot(tel['s'][i0:i1], tel['v_ref'][i0:i1], color='grey', lw=0.8, alpha=0.7, label='v_ref')
    ax1.set_ylabel('speed [m/s]')
    ax1.legend(ncol=4, fontsize=8)
    ax2.set_ylabel('lateral error d [m]')
    ax2.set_xlabel('s along raceline [m]')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    tel = load(a.bag)
    print(analyse(tel))
    if a.plot:
        print('Plot:', plot(tel, os.path.join(a.bag, 'speed_report.png')))
