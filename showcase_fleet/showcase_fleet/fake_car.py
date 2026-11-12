"""fake_car (desk test only): stands in for VESC + PF + EKF + the LQR on a car's LOCAL domain.

Publishes /showcase/drive_raw (a pure-pursuit stand-in for the controller), and simulates a kinematic bicycle that
moves ONLY according to the gated /drive, so start/stop through drive_gate is exercised end to end. Also publishes
/odometry/filtered, /pf/pose/odom and (if vesc_msgs is installed) /sensors/core with a plausible motor current.

    source dds_env.sh local car1 && ros2 run showcase_fleet fake_car --ros-args -p start_s:=0.0
"""
import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan

from showcase_fleet import common as C

try:
    from vesc_msgs.msg import VescStateStamped
except ImportError:  # laptop images may not have vesc_msgs
    VescStateStamped = None

WHEELBASE = 0.33


class FakeCar(Node):
    def __init__(self):
        super().__init__('fake_car')
        self.declare_parameter('track', 'venue_ccw')
        self.declare_parameter('line', 'raceline_stable.csv')
        self.declare_parameter('raceline_file', '')
        self.declare_parameter('start_s', 0.0)          # start this far along the raceline [fraction 0..1]
        self.declare_parameter('speed_scale', 0.6)
        self.declare_parameter('rate_hz', 40.0)
        self.declare_parameter('controller', True)     # False: a real controller (lqr_launch.py) drives drive_raw
        self.declare_parameter('ekf', True)            # False: the real EKF publishes /odometry/filtered
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        rl = C.load_raceline(C.resolve_raceline(g('track'), g('line'), g('raceline_file')))
        self.px, self.py = rl['x'], rl['y']
        self.pv = rl['v'] if rl['v'] is not None else np.full(len(self.px), 3.0)
        i0 = int(float(g('start_s')) * (len(self.px) - 1))
        i1 = (i0 + 1) % len(self.px)
        self.x, self.y = float(self.px[i0]), float(self.py[i0])
        self.yaw = math.atan2(self.py[i1] - self.py[i0], self.px[i1] - self.px[i0])
        self.v = 0.0
        self.steer_applied = 0.0
        self.v_cmd = 0.0
        self.speed_scale = float(g('speed_scale'))
        self.dt = 1.0 / float(g('rate_hz'))
        self.own_controller = bool(g('controller'))
        self.own_ekf = bool(g('ekf'))

        self.raw_pub = self.create_publisher(AckermannDriveStamped, C.DRIVE_RAW_TOPIC, 1)
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 1)
        self.pf_pub = self.create_publisher(Odometry, '/pf/pose/odom', 1)
        self.wheel_pub = self.create_publisher(Odometry, '/odom', 1)            # VESC wheel odometry (rear axle)
        self.imu_pub = self.create_publisher(Imu, '/sensors/imu/raw', 1)        # VESC IMU: gyro deg/s, accel g
        self.declare_parameter('gyro_bias_dps', 0.4)                            # realistic MEMS bias [deg/s]
        self.gyro_bias = float(self.get_parameter('gyro_bias_dps').value)
        # synthetic LiDAR at the laser point: flat list x1, y1, r1, x2, ... of round obstacles in the track frame
        self.declare_parameter('obstacles', [0.0])
        ob = [float(v) for v in self.get_parameter('obstacles').value]
        self.obstacles = [ob[i:i + 3] for i in range(0, len(ob) - 2, 3)]
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 1)
        self.scan_ang = np.linspace(-2.356, 2.356, 1081)
        self.create_timer(1.0 / 20.0, self.scan)
        self.vesc_pub = self.create_publisher(VescStateStamped, '/sensors/core', 1) if VescStateStamped else None
        self.create_subscription(AckermannDriveStamped, C.DRIVE_OUT_TOPIC, self.on_drive, 1)
        self.create_timer(self.dt, self.step)
        self.t_drive = 0.0

    def scan(self):
        lx, ly = self.x + 0.26 * math.cos(self.yaw), self.y + 0.26 * math.sin(self.yaw)
        a = self.yaw + self.scan_ang
        dx, dy = np.cos(a), np.sin(a)
        r = np.full(len(a), 30.0)
        for ox, oy, rad in self.obstacles:          # ray / circle intersection
            fx, fy = lx - ox, ly - oy
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - rad * rad
            disc = b * b - c
            t = -b - np.sqrt(np.maximum(disc, 0.0))
            hit = (disc >= 0) & (t > 0)
            r = np.where(hit, np.minimum(r, t), r)
        m = LaserScan()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'laser'
        m.angle_min, m.angle_max = float(self.scan_ang[0]), float(self.scan_ang[-1])
        m.angle_increment = float(self.scan_ang[1] - self.scan_ang[0])
        m.range_min, m.range_max = 0.06, 30.0
        m.ranges = (r + np.random.normal(0, 0.01, len(r))).astype(np.float32).tolist()
        self.scan_pub.publish(m)

    def on_drive(self, msg):            # what the gate lets through = what the "VESC" executes
        self.v_cmd = float(msg.drive.speed)
        self.steer_applied = float(msg.drive.steering_angle)
        self.t_drive = time.monotonic()

    def step(self):
        dt = self.dt
        if time.monotonic() - self.t_drive > 0.5:        # mux/VESC timeout: coast to a stop
            self.v_cmd = 0.0
        a = float(np.clip((self.v_cmd - self.v) / 0.3, -6.0, 4.0))
        self.v = max(0.0, self.v + a * dt)
        r = self.v * math.tan(self.steer_applied) / WHEELBASE
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += r * dt

        # controller stand-in: pure pursuit on the raceline
        d = np.hypot(self.px - self.x, self.py - self.y)
        i = int(np.argmin(d))
        look = max(0.8, 0.35 * self.v)
        j = i
        while np.hypot(self.px[j] - self.x, self.py[j] - self.y) < look:
            j = (j + 1) % len(self.px)
            if j == i:
                break
        dx, dy = self.px[j] - self.x, self.py[j] - self.y
        alpha = math.atan2(dy, dx) - self.yaw
        steer = float(np.clip(math.atan2(2 * WHEELBASE * math.sin(alpha), max(look, 1e-3)), -0.4, 0.4))
        stamp = self.get_clock().now().to_msg()
        raw = AckermannDriveStamped()
        raw.header.stamp = stamp
        raw.drive.speed = float(self.pv[i] * self.speed_scale)
        raw.drive.steering_angle = steer
        if self.own_controller:
            self.raw_pub.publish(raw)

        def odom_at(d, frame, child):      # (x, y) is the rear axle; d = point ahead of it
            od = Odometry()
            od.header.stamp = stamp
            od.header.frame_id = frame
            od.child_frame_id = child
            od.pose.pose.position.x = self.x + d * math.cos(self.yaw)
            od.pose.pose.position.y = self.y + d * math.sin(self.yaw)
            od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
            od.twist.twist.linear.x = self.v
            od.twist.twist.linear.y = d * r
            od.twist.twist.angular.z = r
            return od
        if self.own_ekf:
            self.odom_pub.publish(odom_at(0.15, 'map', 'cg'))
        self.pf_pub.publish(odom_at(0.26, 'map', 'laser'))                    # PF localizes the laser
        w = odom_at(0.0, 'odom', 'base_link')
        w.twist.twist.linear.y = 0.0
        self.wheel_pub.publish(w)
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'imu_link'
        imu.angular_velocity.z = math.degrees(r) + self.gyro_bias + float(np.random.normal(0.0, 0.3))
        imu.linear_acceleration.z = 1.0
        self.imu_pub.publish(imu)
        if self.vesc_pub is not None:
            m = VescStateStamped()
            m.header.stamp = stamp
            m.state.speed = self.v / 0.05 * 11.82 / (2 * math.pi) * 60.0 * 2.0   # rough erpm
            iq = 2.0 * a + 0.8 * self.v
            if hasattr(m.state, 'avg_iq'):
                m.state.avg_iq = iq
            m.state.current_motor = iq
            self.vesc_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = FakeCar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
