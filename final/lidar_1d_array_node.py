#!/usr/bin/env python3

"""1D LiDAR Array Node
terminal 1: launch bring up
terminal 2: source /opt/ros/humble/setup.bash
/usr/bin/python3 lidar_1d_array_node.py
terminal 3: source /opt/ros/humble/setup.bash
ros2 topic echo /lidar_1d
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


class Lidar1DArrayNode(Node):
    def __init__(self):
        super().__init__("lidar_1d_array_node")

        # ---------------- ROS parameters ----------------
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("output_topic", "/lidar_1d")

        # 60-degree fan centered around forward direction.
        self.declare_parameter("fan_angle_deg", 60.0)

        # Number of output lidar values.
        self.declare_parameter("num_points", 8)

        # Local angular window around each anchor direction.
        # Larger = smoother, smaller = more directional.
        self.declare_parameter("window_deg", 2.0)

        # If your lidar's 0 degree is not exactly forward, change this.
        # Example: if forward is +90 deg in your scan frame, set angle_offset_deg = 90.
        self.declare_parameter("angle_offset_deg", 0.0)

        # Fill value when no valid range exists.
        self.declare_parameter("invalid_fill_value", 10.0)

        # Choose "median" or "min".
        # median = smoother ML input
        # min = more conservative obstacle distance
        self.declare_parameter("reduction_method", "median")

        self.scan_topic = self.get_parameter("scan_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.fan_angle_deg = float(self.get_parameter("fan_angle_deg").value)
        self.num_points = int(self.get_parameter("num_points").value)
        self.window_deg = float(self.get_parameter("window_deg").value)
        self.angle_offset_deg = float(self.get_parameter("angle_offset_deg").value)
        self.invalid_fill_value = float(self.get_parameter("invalid_fill_value").value)
        self.reduction_method = str(self.get_parameter("reduction_method").value)

        # LaserScan often uses sensor-data QoS.
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            scan_qos,
        )

        self.lidar_pub = self.create_publisher(
            Float32MultiArray,
            self.output_topic,
            10,
        )

        self.get_logger().info("lidar_1d_array_node started")
        self.get_logger().info(f"Subscribing to: {self.scan_topic}")
        self.get_logger().info(f"Publishing to: {self.output_topic}")
        self.get_logger().info(f"Fan angle: {self.fan_angle_deg} deg")
        self.get_logger().info(f"Number of points: {self.num_points}")
        self.get_logger().info(f"Window: +/- {self.window_deg} deg")
        self.get_logger().info(f"Angle offset: {self.angle_offset_deg} deg")
        self.get_logger().info(f"Reduction method: {self.reduction_method}")

    def scan_callback(self, scan_msg: LaserScan):
        distances = self.extract_1d_lidar_array(scan_msg)

        output_msg = Float32MultiArray()
        output_msg.data = distances

        self.lidar_pub.publish(output_msg)

        rounded = [round(x, 3) for x in distances]
        self.get_logger().info(f"lidar_1d: {rounded}", throttle_duration_sec=0.5)

    def extract_1d_lidar_array(self, scan_msg: LaserScan):
        half_fan_rad = math.radians(self.fan_angle_deg / 2.0)
        angle_offset_rad = math.radians(self.angle_offset_deg)

        anchor_angles = np.linspace(
            -half_fan_rad,
            half_fan_rad,
            self.num_points,
        )

        distances = []

        for anchor_angle in anchor_angles:
            target_angle = anchor_angle + angle_offset_rad
            distance = self.get_range_near_angle(scan_msg, target_angle)
            distances.append(float(distance))

        return distances

    def get_range_near_angle(self, scan_msg: LaserScan, target_angle_rad: float):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)

        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        angle_increment = scan_msg.angle_increment

        if target_angle_rad < angle_min or target_angle_rad > angle_max:
            return self.invalid_fill_value

        target_index = int(round((target_angle_rad - angle_min) / angle_increment))

        if target_index < 0 or target_index >= len(ranges):
            return self.invalid_fill_value

        window_rad = math.radians(self.window_deg)
        window_size = int(round(window_rad / abs(angle_increment)))

        start_index = max(0, target_index - window_size)
        end_index = min(len(ranges), target_index + window_size + 1)

        local_ranges = ranges[start_index:end_index]

        valid_mask = np.isfinite(local_ranges)
        valid_mask &= local_ranges >= scan_msg.range_min
        valid_mask &= local_ranges <= scan_msg.range_max

        valid_ranges = local_ranges[valid_mask]

        if len(valid_ranges) == 0:
            return self.invalid_fill_value

        if self.reduction_method == "min":
            return float(np.min(valid_ranges))

        return float(np.median(valid_ranges))


def main(args=None):
    rclpy.init(args=args)

    node = Lidar1DArrayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()