#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import csv
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy import interpolate

import os
class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")
        print(os.getcwd())
        self.waypoints = self.load_waypoints("/home/team10/f1tenth_ws/src/pure_pursuit/waypoints/path_racePenn_2.csv")

        self.subscription = self.create_subscription(
            Odometry, "/pf/pose/odom", self.pose_callback, 10
        )
        self.marker_pub = self.create_publisher(Marker, "/visualization_marker", 10)
        self.publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.lookahead_dist = 1.0
        self.wheelbase = 0.324 

        self.prev_steer = 0.0
        self.steer_alpha = 0.25 

        self.timer = self.create_timer(10, self.publish_all_waypoints)

    def load_waypoints(self, path):
        data = np.genfromtxt(path, delimiter=",", skip_header=0)
        self.raw_waypoints = data.tolist()
        x = data[:, 0]
        y = data[:, 1]

        x = np.append(x,data[0,0])
        y = np.append(y,data[0,1])

        tck, _ = interpolate.splprep([x, y], s=0.0, k=2)

        u_fine = np.linspace(0, 1, 10000)
        x_fine, y_fine = interpolate.splev(u_fine, tck)

        return np.vstack((x_fine, y_fine)).T.tolist()

    def pose_callback(self, pose_msg):
        current_pose = pose_msg.pose.pose

        # TODO: find the current waypoint to track using methods mentioned in lecture
        goal_point = self.get_lookahead_point(current_pose)

        self.publish_marker(goal_point)

        # TODO: transform goal point to vehicle frame of reference
        transformed_goal = self.transform_to_vehicle_frame(current_pose, goal_point)

        # TODO: calculate curvature/steering angle
        # gamma = 2 * y / L^2
        dist_sq = transformed_goal[0] ** 2 + transformed_goal[1] ** 2
        curvature = 2 * transformed_goal[1] / dist_sq

        # TODO: publish drive message, don't forget to limit the steering angle.
        drive_msg = AckermannDriveStamped()
        # drive_msg.drive.steering_angle = np.clip(
        #     curvature * self.wheelbase, -0.4189, 0.4189
        # )

        # Ackermann form δ = atan(L · κ)
        # steering_angle = np.clip(
        #     curvature * self.wheelbase, -0.4, 0.4
        # )
        steering_angle = np.clip(
            np.arctan(curvature * self.wheelbase), -0.4, 0.4
        )

        # Low-pass filter on steering to absorb particle-filter jitter
        steering_angle = (self.steer_alpha * self.prev_steer
                          + (1.0 - self.steer_alpha) * steering_angle)
        self.prev_steer = float(steering_angle)

        abs_angle = np.abs(steering_angle)
        print(f'{abs_angle=}')

        # Linear ramp instead of 1.0↔3.0 cliff
        # speed = 1.
        # if abs_angle < 0.39:
        #     speed = 3.
        # elif abs_angle < 10:
        #      speed = 1.0
        # elif abs_angle < 20:
        #      speed = 1.2
        speed_max = 4
        speed_min = 1.5
        speed = speed_max - (speed_max - speed_min) * (abs_angle / 0.4)
        speed = float(np.clip(speed, speed_min, speed_max))

        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.publisher.publish(drive_msg)

    def publish_marker(self, goal_point):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_point[0])
        marker.pose.position.y = float(goal_point[1])
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        self.marker_pub.publish(marker)

    def transform_to_vehicle_frame(self, current_pose, goal_point):
        rx, ry = current_pose.position.x, current_pose.position.y

        q = current_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        dx = goal_point[0] - rx
        dy = goal_point[1] - ry

        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        transformed_x = dx * cos_yaw - dy * sin_yaw
        transformed_y = dx * sin_yaw + dy * cos_yaw

        return [transformed_x, transformed_y]

    def get_lookahead_point(self, current_pose):
        rx, ry = current_pose.position.x, current_pose.position.y
        q = current_pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))

        wps = np.array(self.waypoints)  #(N, 2)

        dists = np.sqrt((wps[:, 0] - rx) ** 2 + (wps[:, 1] - ry) ** 2)

        dx = wps[:, 0] - rx
        dy = wps[:, 1] - ry

        x_local = dx * np.cos(yaw) + dy * np.sin(yaw)

        mask = (dists >= self.lookahead_dist) & (x_local > 0)

        valid_indices = np.where(mask)[0]

        if valid_indices.size > 0:
            valid_dists = dists[valid_indices]
            best_idx = valid_indices[np.argmin(valid_dists)]
            return wps[best_idx].tolist()

        return wps[np.argmin(dists)].tolist()

    def publish_all_waypoints(self):
        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.ns = "smooth_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.scale.x = 0.05
        path_marker.color.g = 1.0
        path_marker.color.a = 1.0
        for p in self.waypoints:
            path_marker.points.append(Point(x=p[0], y=p[1], z=0.0))
        self.marker_pub.publish(path_marker)

        raw_marker = Marker()
        raw_marker.header.frame_id = "map"
        raw_marker.ns = "raw_points"
        raw_marker.id = 1
        raw_marker.type = Marker.SPHERE_LIST
        raw_marker.scale.x = 0.15
        raw_marker.scale.y = 0.15
        raw_marker.scale.z = 0.15
        raw_marker.color.r = 1.0
        raw_marker.color.a = 1.0
        for p in self.raw_waypoints:
            raw_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
        self.marker_pub.publish(raw_marker)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()






# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# import numpy as np
# import csv
# import os
# from nav_msgs.msg import Odometry
# from ackermann_msgs.msg import AckermannDriveStamped
# from visualization_msgs.msg import Marker, MarkerArray
# from math import atan2, sqrt, cos, sin

# class PurePursuit(Node):
#     def __init__(self):
#         super().__init__('pure_pursuit_node')

#         self.lookahead_distance = 1.0
#         self.speed = 1.5

#         waypoint_file = os.path.expanduser(
#             '~/f1tenth_ws/src/pure_pursuit/waypoints/path_race2.csv')

#         self.waypoints = []
#         with open(waypoint_file, 'r') as f:
#             reader = csv.reader(f)
#             next(reader)
#             for row in reader:
#                 self.waypoints.append([float(row[0]), float(row[1])])
#         self.waypoints = np.array(self.waypoints)
#         self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')

#         self.pose_sub = self.create_subscription(
#             Odometry, '/ego_racecar/odom', self.pose_callback, 10)
#         self.drive_pub = self.create_publisher(
#             AckermannDriveStamped, '/drive', 10)
#         self.waypoint_pub = self.create_publisher(
#             MarkerArray, '/waypoints_viz', 10)
#         self.target_pub = self.create_publisher(
#             Marker, '/target_waypoint', 10)

#         self.get_logger().info('Pure Pursuit Node Ready!')

#     def pose_callback(self, pose_msg):
#         cx = pose_msg.pose.pose.position.x
#         cy = pose_msg.pose.pose.position.y

#         qx = pose_msg.pose.pose.orientation.x
#         qy = pose_msg.pose.pose.orientation.y
#         qz = pose_msg.pose.pose.orientation.z
#         qw = pose_msg.pose.pose.orientation.w
#         theta = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

#         target = self.find_lookahead_point(cx, cy, theta)

#         if target is None:
#             self.get_logger().warn('No lookahead point found!')
#             return

#         dx = target[0] - cx
#         dy = target[1] - cy
#         local_x = dx * cos(theta) + dy * sin(theta)
#         local_y = -dx * sin(theta) + dy * cos(theta)

#         L = sqrt(local_x**2 + local_y**2)
#         curvature = 2 * local_y / (L ** 2)
#         steering_angle = atan2(curvature, 1.0)
#         steering_angle = max(-0.4, min(0.4, steering_angle))

#         drive_msg = AckermannDriveStamped()
#         drive_msg.drive.speed = self.speed
#         drive_msg.drive.steering_angle = steering_angle
#         self.drive_pub.publish(drive_msg)

#         self.visualize_waypoints()
#         self.visualize_target(target)

#     def find_lookahead_point(self, cx, cy, theta):
#         best = None
#         best_diff = float('inf')

#         for wp in self.waypoints:
#             dx = wp[0] - cx
#             dy = wp[1] - cy

#             local_x = dx * cos(theta) + dy * sin(theta)

#             if local_x < 0:
#                 continue

#             dist = sqrt(dx**2 + dy**2)
#             diff = abs(dist - self.lookahead_distance)

#             if diff < best_diff:
#                 best_diff = diff
#                 best = wp

#         return best

#     # Visualizing Waypoints in RViz - Youtube and other resources.
#     def visualize_waypoints(self):
#         marker_array = MarkerArray()
#         for i, wp in enumerate(self.waypoints):
#             marker = Marker()
#             marker.header.frame_id = 'map'
#             marker.header.stamp = self.get_clock().now().to_msg()
#             marker.id = i
#             marker.type = Marker.SPHERE
#             marker.action = Marker.ADD
#             marker.pose.position.x = wp[0]
#             marker.pose.position.y = wp[1]
#             marker.pose.position.z = 0.0
#             marker.scale.x = 0.1
#             marker.scale.y = 0.1
#             marker.scale.z = 0.1
#             marker.color.r = 0.0
#             marker.color.g = 1.0
#             marker.color.b = 0.0
#             marker.color.a = 1.0
#             marker_array.markers.append(marker)
#         self.waypoint_pub.publish(marker_array)

#     def visualize_target(self, target):
#         marker = Marker()
#         marker.header.frame_id = 'map'
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.id = 0
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD
#         marker.pose.position.x = target[0]
#         marker.pose.position.y = target[1]
#         marker.pose.position.z = 0.0
#         marker.scale.x = 0.3
#         marker.scale.y = 0.3
#         marker.scale.z = 0.3
#         marker.color.r = 1.0
#         marker.color.g = 0.0
#         marker.color.b = 0.0
#         marker.color.a = 1.0
#         self.target_pub.publish(marker)

# def main(args=None):
#     rclpy.init(args=args)
#     print("PurePursuit Initialized")
#     pure_pursuit_node = PurePursuit()
#     rclpy.spin(pure_pursuit_node)
#     pure_pursuit_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
