#!/usr/bin/env python3
"""
Lateral grip from a constant-steer circle test (manual trigger, laptop or Jetson).

    python3 grip_from_bag.py <bag_folder>

Needs topics: /drive (the fixed command), /odom (wheel speed), and
/sensors/imu/raw (yaw rate; falls back to /odom yaw rate if missing).
For every commanded speed step it uses the steady second half of the step and prints:
    measured speed v, yaw rate r, circle radius R = v/r,
    lateral accel a_y = v*r, grip mu = a_y / g, and steer vs L/R (understeer).
The last step that still holds the circle (R close to the slow-speed R) gives mu_lat.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

G, L = 9.81, 0.325


def read(bag):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    meta = yaml.safe_load(open(os.path.join(bag, 'metadata.yaml')))
    sid = meta['rosbag2_bagfile_information'].get('storage_identifier', 'sqlite3')
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=sid), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
    out = defaultdict(list)
    while r.has_next():
        topic, data, t = r.read_next()
        if topic not in ('/drive', '/odom', '/sensors/imu/raw'):
            continue
        m = deserialize_message(data, types[topic])
        ts = t * 1e-9
        if topic == '/drive':
            out['drive'].append((ts, m.drive.speed, m.drive.steering_angle))
        elif topic == '/odom':
            out['odom'].append((ts, m.twist.twist.linear.x, m.twist.twist.angular.z))
        else:
            out['imu'].append((ts, m.angular_velocity.z))
    return {k: np.array(v) for k, v in out.items()}


def analyse(d):
    if 'drive' not in d or 'odom' not in d:
        sys.exit('Need /drive and /odom in the bag.')
    drv, odo = d['drive'], d['odom']
    use_imu = 'imu' in d and len(d['imu']) > 10
    rate_t, rate = (d['imu'][:, 0], d['imu'][:, 1]) if use_imu else (odo[:, 0], odo[:, 2])
    print(f'yaw rate from {"IMU" if use_imu else "odom (IMU missing)"}\n')
    print(' cmd v | steer  |  v meas | yaw r  |  R [m] | a_y m/s2 |  mu  | steer - L/R')
    print('-------+--------+---------+--------+--------+----------+------+------------')
    # steps = runs of identical (speed, steer) commands
    key = np.round(drv[:, 1], 2) * 1000 + np.round(drv[:, 2], 3)
    cut = np.where(np.diff(key) != 0)[0] + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(drv)]):
        t0, t1 = drv[a, 0], drv[b - 1, 0]
        if t1 - t0 < 3.0 or abs(drv[a, 1]) < 0.3:
            continue                                   # too short or standing still
        tm = t0 + 0.5 * (t1 - t0)                      # steady part: second half of the step
        v = np.median(odo[(odo[:, 0] > tm) & (odo[:, 0] < t1), 1])
        r = np.median(np.abs(rate[(rate_t > tm) & (rate_t < t1)]))
        if not np.isfinite(v) or r < 1e-3:
            continue
        steer = drv[a, 2]
        R, ay = v / r, v * r
        print(f'{drv[a, 1]:6.2f} | {steer:+.3f} | {v:7.2f} | {r:6.2f} | {R:6.2f} | {ay:8.2f} | {ay / G:4.2f} | '
              f'{abs(steer) - np.arctan(L / R):+.3f}')


if __name__ == '__main__':
    analyse(read(sys.argv[1]))
