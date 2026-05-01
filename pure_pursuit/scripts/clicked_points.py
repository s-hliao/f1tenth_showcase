#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import csv
import os


class WaypointLogger(Node):
    def __init__(self):
        super().__init__("waypoint_logger")
        self.log_sub = self.create_subscription(
            PointStamped, "/clicked_point", self.waypoint_callback, 10
        )

        file_path = "./path_racePenn.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.csv_file = open(file_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.get_logger().info("logger started")

    def close_csv(self):
        self.csv_file.close()

    def waypoint_callback(self, msg):
        x = msg.point.x
        y = msg.point.y
        self.csv_writer.writerow([x, y])
        self.csv_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
