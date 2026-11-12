#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
import math

class ImuZuptPrepNode(Node):
    def __init__(self):
        super().__init__('imu_zupt_prep')
        
        # --- Subscribers ---
        self.raw_imu_sub = self.create_subscription(Imu, '/sensors/imu/raw', self.imu_cb, 10)
        self.cmd_odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        # --- Publishers ---
        self.clean_imu_pub = self.create_publisher(Imu, '/sensors/imu/data', 10)
        self.zupt_pub = self.create_publisher(TwistWithCovarianceStamped, '/zupt', 10)
        
        # --- State Variables & Constants ---
        self.cmd_v = 0.0
        self.G_TO_MS2 = 9.80665
        self.DEG_TO_RAD = math.pi / 180.0

        # --- showcase additions (defaults = llampc behaviour) ---
        # gyro_scale: raw gyro -> rad/s (VESC IMU reports deg/s; use 1.0 if a driver already gives rad/s)
        # yaw_sign:   -1 if the IMU is mounted so that a left (CCW) turn reads negative (check: imu_check)
        # bias_calib_s: average the gyro for this long at standstill after start and subtract it (dead reckoning
        #               is only as good as the yaw-rate bias; 0 = off)
        self.declare_parameter('gyro_scale', self.DEG_TO_RAD)
        self.declare_parameter('yaw_sign', 1.0)
        self.declare_parameter('bias_calib_s', 0.0)
        self.gyro_scale = float(self.get_parameter('gyro_scale').value)
        self.yaw_sign = float(self.get_parameter('yaw_sign').value)
        self.calib_s = float(self.get_parameter('bias_calib_s').value)
        self.bias = [0.0, 0.0, 0.0]
        self._calib = []
        self._calib_t0 = None
        self.calibrated = self.calib_s <= 0.0

    def odom_cb(self, msg):
        """ Tracks commanded velocity to know if we are TRYING to move. """
        self.cmd_v = msg.twist.twist.linear.x

    def imu_cb(self, raw_msg):
        """ Fixes the IMU data and checks if we are parked. """
        
        # ==========================================
        # PART A: FIX AND PUBLISH THE IMU
        # ==========================================
        clean_imu = Imu()
        clean_imu.header.stamp = raw_msg.header.stamp
        clean_imu.header.frame_id = 'imu_link'
        
        # Convert units (g's -> m/s^2, deg/s -> rad/s)
        clean_imu.linear_acceleration.x = raw_msg.linear_acceleration.x * self.G_TO_MS2
        clean_imu.linear_acceleration.y = raw_msg.linear_acceleration.y * self.G_TO_MS2
        clean_imu.linear_acceleration.z = raw_msg.linear_acceleration.z * self.G_TO_MS2
        
        g = [raw_msg.angular_velocity.x * self.gyro_scale, raw_msg.angular_velocity.y * self.gyro_scale,
             raw_msg.angular_velocity.z * self.gyro_scale * self.yaw_sign]
        if not self.calibrated:
            # collect while the car is standing still; publish nothing until the bias is known
            t = raw_msg.header.stamp.sec + raw_msg.header.stamp.nanosec * 1e-9
            if abs(self.cmd_v) > 0.01:
                self._calib, self._calib_t0 = [], None           # moved: start over
                return
            self._calib_t0 = t if self._calib_t0 is None else self._calib_t0
            self._calib.append(g)
            if t - self._calib_t0 < self.calib_s:
                return
            n = len(self._calib)
            self.bias = [sum(v[i] for v in self._calib) / n for i in range(3)]
            self.calibrated = True
            self.get_logger().info(f'gyro bias {self.bias[2]:+.5f} rad/s ({math.degrees(self.bias[2]):+.3f} deg/s) '
                                   f'from {n} samples; yaw sign {self.yaw_sign:+.0f}')
        clean_imu.angular_velocity.x = g[0] - self.bias[0]
        clean_imu.angular_velocity.y = g[1] - self.bias[1]
        clean_imu.angular_velocity.z = g[2] - self.bias[2]
        
        # Pass orientation exactly as is
        clean_imu.orientation = raw_msg.orientation
        
        # Inject "Good Guess" Covariances
        clean_imu.linear_acceleration_covariance = [
            0.05, 0.0,  0.0,   # 0.05 is a safe, moderate variance for MEMS accelerometers
            0.0,  0.05, 0.0,
            0.0,  0.0,  0.05
        ]
        clean_imu.angular_velocity_covariance = [
            0.005, 0.0,   0.0, # Gyros are usually cleaner than accelerometers
            0.0,   0.005, 0.0,
            0.0,   0.0,   0.005
        ]
        clean_imu.orientation_covariance[0] = -1.0 # Tell EKF to ignore absolute orientation
        
        self.clean_imu_pub.publish(clean_imu)

        # ==========================================
        # PART B: THE ZUPT (STOP DETECTOR) LOGIC
        # ==========================================
        # Are we commanded to stop, AND is the rotational gyro physically quiet?
        if abs(self.cmd_v) < 0.01 and abs(clean_imu.linear_acceleration.x) < 0.2 and abs(clean_imu.angular_velocity.z) < 0.05:
            zupt_msg = TwistWithCovarianceStamped()
            zupt_msg.header.stamp = raw_msg.header.stamp
            zupt_msg.header.frame_id = 'base_link'
            
            # Pin velocities to exactly 0
            zupt_msg.twist.twist.linear.x = 0.0
            zupt_msg.twist.twist.linear.y = 0.0
            zupt_msg.twist.twist.angular.z = 0.0
            
            # Inject extreme confidence (1e-6) that we are stopped
            cov = [0.0] * 36
            cov[0] = 1e-6   # Trust Vx = 0
            cov[7] = 1e-6   # Trust Vy = 0
            cov[35] = 1e-6  # Trust Vyaw = 0
            zupt_msg.twist.covariance = cov
            
            self.zupt_pub.publish(zupt_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuZuptPrepNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()