#!/usr/bin/env python3
"""imu_check: confirm the gyro sign / units before dead reckoning.

    ros2 run showcase_state_estimation imu_check            # with the car's bringup (VESC driver) running

Turn the car LEFT (counter-clockwise seen from above) by hand, about a quarter turn in ~1 s. The integrated yaw of
/sensors/imu/raw (scaled like imu_zupt_prep, deg/s -> rad/s) should read about +90 deg. If it reads about -90, use
yaw_sign:=-1. If it reads about +1.6 (i.e. ~90 x pi/180), the driver already reports rad/s: use gyro_scale:=1.0.
"""
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


def main(args=None):
    rclpy.init(args=args)
    n = Node('imu_check')
    st = {'yaw': 0.0, 't': None, 'n': 0, 'peak': 0.0}

    def cb(m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        wz = m.angular_velocity.z * math.pi / 180.0
        if st['t'] is not None and 0 < t - st['t'] < 0.2:
            st['yaw'] += wz * (t - st['t'])
        st['t'] = t
        st['n'] += 1
        st['peak'] = max(st['peak'], abs(m.angular_velocity.z))

    n.create_subscription(Imu, '/sensors/imu/raw', cb, 10)
    print('Turn the car LEFT (CCW from above) ~90 deg now; reading for 8 s...', flush=True)
    t0 = time.time()
    while time.time() - t0 < 8.0:
        rclpy.spin_once(n, timeout_sec=0.05)
    yaw = math.degrees(st['yaw'])
    print(f'{st["n"]} samples, raw |gyro z| peak {st["peak"]:.2f}, integrated yaw {yaw:+.1f} deg (as deg/s input)')
    if st['n'] == 0:
        print('NO /sensors/imu/raw: is the VESC driver running (and this shell on the car domain)?')
    elif abs(yaw) < 10.0:
        print('Tiny: either the car was not turned, or the driver already reports rad/s -> gyro_scale:=1.0')
    else:
        print('OK: yaw_sign 1 (default)' if yaw > 0 else 'NEGATIVE: launch with yaw_sign:=-1')
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
