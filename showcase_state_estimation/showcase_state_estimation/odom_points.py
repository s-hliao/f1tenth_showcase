#!/usr/bin/env python3
"""odom_points: re-express a rigid-body odometry at other points of the car.

Dead reckoning estimates the rear axle (base_link). Downstream expects other points:
    /dr/pose/odom      laser point (+0.26 m): what the LQR tracks (its REF_X), same point the particle filter gives
    /showcase/odometry cg (+0.15 m): what the relay, ground station and LLA estimator use in PF mode
For a point at +d along the heading: p = p_axle + d (cos psi, sin psi), v_body = (vx, vy + d * r).
"""
import copy
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class OdomPoints(Node):
    def __init__(self):
        super().__init__('odom_points')
        self.declare_parameter('input', '/odometry/dr_base')
        self.declare_parameter('outputs', ['/dr/pose/odom', '/showcase/odometry'])
        self.declare_parameter('offsets', [0.26, 0.15])
        self.declare_parameter('frames', ['laser', 'cg'])
        outs = list(self.get_parameter('outputs').value)
        offs = [float(v) for v in self.get_parameter('offsets').value]
        frames = list(self.get_parameter('frames').value)
        self.outs = [(self.create_publisher(Odometry, t, 5), d, f) for t, d, f in zip(outs, offs, frames)]
        self.create_subscription(Odometry, self.get_parameter('input').value, self.cb, 5)

    def cb(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)
        r = m.twist.twist.angular.z
        for pub, d, frame in self.outs:
            o = Odometry()
            o.header = m.header
            o.child_frame_id = frame
            o.pose = copy.deepcopy(m.pose)          # copies: m is shared by every output
            o.twist = copy.deepcopy(m.twist)
            o.pose.pose.position.x = m.pose.pose.position.x + d * c
            o.pose.pose.position.y = m.pose.pose.position.y + d * s
            o.twist.twist.linear.y = m.twist.twist.linear.y + d * r
            pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    n = OdomPoints()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
