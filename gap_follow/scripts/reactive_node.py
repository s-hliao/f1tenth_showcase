#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import time
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class ReactiveFollowGap(Node):
    """
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        self.scan_subscriber = self.create_subscription(LaserScan,lidarscan_topic,self.lidar_callback,10)
        # TODO: Publish to drive
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.marker_pub = self.create_publisher(Marker, "/target_point_marker", 10)
        self.target_scan_pub = self.create_publisher(LaserScan, "/target_scan", 10)
        self.target_zone_pub = self.create_publisher(Marker, "/target_zone_marker", 10)

        self.processed_lines_pub = self.create_publisher(
            Marker, "/processed_scan_lines", 10
        )
        self.scan_subscriber
        self.publisher

        self.mean_windows_size = 3
        self.max_range = 5
        self.half_car_size = .4
        self.diff_value = .28

        self.considered_angle = np.deg2rad(90) ## only consider -self.considered_angle ~ self.considered_angle 

        self.angle_increment = None
        self.laser_count = None
        self.last_time = time.time()

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            2.Rejecting high values (eg. > 3m) ## should reject first?
            1.Setting each value to the mean over some window
        """
        tmp_ranges = ranges.copy()
        tmp_ranges[tmp_ranges > self.max_range] = self.max_range
        kernel = np.ones(self.mean_windows_size) / self.mean_windows_size
        proc_ranges = np.convolve(tmp_ranges, kernel, mode="same")
        return proc_ranges

    def disparity_extend(self,ranges):
        ## find the big jump in reading
        diff = np.diff(ranges)
        # print('diff: ' , diff)
        indices = np.where(np.abs(diff) > self.diff_value)[0]
        # next_indices = np.clip(indices+1,len(ranges))
        # diffs = np.vstack((indices,next_indices))
        # left_larger = diff[diff[0] > diff[1]]
        # right_larger = diff[diff[1] > diff[0]]
        # print(indices)
        free_space_ranges = ranges.copy()

        left_closest = np.min(ranges[self.laser_count // 2 :])
        right_closest = np.min(ranges[: self.laser_count // 2])

        tunnel_threshold = 0.4
        is_in_tunnel = (left_closest < tunnel_threshold) and (
            right_closest < tunnel_threshold
        )

        if is_in_tunnel:
            print('tunnel detected!')
            active_half_width = self.half_car_size * 0.7
            max_angle_limit = np.deg2rad(90)
        else:
            active_half_width = self.half_car_size
            max_angle_limit = float("inf")

        for i in indices:
            left = ranges[i]
            # right = ranges[i + 1] if i + 1 < self.laser_count else i  
            right = ranges[i + 1] if i + 1 < self.laser_count else ranges[i]  
            closest = min(left,right)
            ratio = np.clip(active_half_width / closest, -1.0, 1.0)
            mask_angle = np.arcsin(ratio) #* .6
            mask_angle = np.clip(mask_angle, 0, max_angle_limit)
            mask_count = int(np.ceil(mask_angle / self.angle_increment))
            # mask_count = np.ceil(np.arctan2(self.half_car_size,closest) / self.angle_increment).astype(int)
            # mask_count = 40
            # print(mask_count)
            if left < right: ## masking (i) i+1 ... i+mask_count
                start = i
                end = min(i + 1 + mask_count, self.laser_count)
                free_space_ranges[start:end] = 0
                # free_space_ranges[start:end] = closest
                # print(free_space_ranges[start:end])
            else:
                # start = max(i - 1 + mask_count, 0)
                start = max(i + 1 - mask_count, 0)
                end = i + 1
                free_space_ranges[start:end] = 0
                # free_space_ranges[start:end] = closest
        if self.cur_time - self.last_time > 5:
            print(
                ranges[135 - 10 : 135 + 10],
                np.diff(ranges[135 - 10 : 135 + 10]),
                indices,
                np.where(free_space_ranges == 0),
                # free_space_ranges,
                # np.rad2deg(mask_angle),
                # mask_count
            )
            print("\n\n")
            self.last_time = self.cur_time
        return free_space_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        return 0,self.laser_count
        mask = np.where(free_space_ranges > 0)[0]
        diff = np.diff(mask)

        if len(mask) == 0:
            print('NO GAP FOUND')
            return -1,-1
        # If mask is [10, 11, 12, 50, 51], np.diff(mask) is [1, 1, 38, 1]
        # The '38' tells us there is a wall between index 12 and 50.
        gaps = np.split(mask,np.where(diff>1)[0]+1) ## split at specific index (which are the indices that the diff is too big.)
        self.gaps = gaps
        ## find the widest one
        lens = [len(g) for g in gaps]
        start, end = gaps[np.argmax(lens)][0], gaps[np.argmax(lens)][-1]
        # print(f"lens: ", gaps[0][0], gaps[1][0], mask)
        return start,end

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        deepest = start_i + np.argmax(ranges[start_i:end_i+1])
        return deepest

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """

        self.cur_time = time.time()
        drive_msg = AckermannDriveStamped()

        ranges = np.array(data.ranges)
        if self.angle_increment is None:
            self.angle_increment = data.angle_increment
            self.angle_min = data.angle_min
            end_angle_index = round((self.considered_angle - self.angle_min) / self.angle_increment) 
            start_angle_index = (
                round(  ## lets define start as the right hand side (the negative side)
                    (-self.considered_angle - self.angle_min) / self.angle_increment
                )
            )
            self.start_angle_index = max(0,start_angle_index)
            self.end_angle_index = min(end_angle_index,len(ranges))

        ranges = ranges[self.start_angle_index:self.end_angle_index]
        if self.laser_count is None:
            self.laser_count = len(ranges)

        proc_ranges = self.preprocess_lidar(ranges)

        # TODO:
        # Find closest point to LiDAR
        # Eliminate all points inside 'bubble' (set them to zero)
        free_space_ranges = self.disparity_extend(proc_ranges)

        # Find max length gap
        start_index, end_index = self.find_max_gap(free_space_ranges)
        if start_index == -1 or end_index == -1:
            drive_msg.drive.speed = .0

        # Find the best point in the gap
        target_index = self.find_best_point(start_index,end_index,free_space_ranges)

        ## Convert index to angle here?
        angle = self.angle_min + self.angle_increment * (target_index + self.start_angle_index)
        abs_angle = np.rad2deg(np.abs(angle))
        if abs_angle < 10:
            speed = 3.0
        elif abs_angle < 20:
            speed = 2.0
        else:
            speed = 1.0
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed
        # Publish Drive message
        drive_msg.header.stamp=self.get_clock().now().to_msg()
        self.publisher.publish(drive_msg)

        target_angle = self.angle_min + self.angle_increment * (target_index + self.start_angle_index)
        target_dist = proc_ranges[target_index] 

        global_target_idx = target_index 

        self.publish_processed_lines(free_space_ranges)
        self.publish_target_zone(global_target_idx, free_space_ranges)
        self.publish_marker(target_angle, target_dist)

        # print(global_target_idx)

        if int(self.cur_time) % 5 == -1:
            print(
                self.considered_angle,
                -self.considered_angle,
                data.angle_min,
                data.angle_max,
                self.start_angle_index,
                self.end_angle_index,
                np.where(
                    proc_ranges[global_target_idx - 10 : global_target_idx + 10] == 0
                ),
                proc_ranges[global_target_idx - 10 : global_target_idx + 10],
                np.diff(proc_ranges[global_target_idx - 10 : global_target_idx + 10]),
                np.where(
                    np.abs(
                        np.diff(ranges[global_target_idx - 10 : global_target_idx + 10])
                    )
                    > self.diff_value
                )[0],
                global_target_idx
            )
            print("\n\n")
            self.last_time = self.cur_time

    def publish_target_zone(self, target_index, ranges):
        if target_index < 0:
            return

        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser_model"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_zone"
        marker.id = 2
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.02

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        start_idx = max(0, target_index - 10)
        end_idx = min(len(ranges), target_index + 11)
        # print(start_idx,end_idx)

        for i in range(start_idx, end_idx):
            dist = ranges[i]
            if dist <= 0:
                continue

            angle = self.angle_min + (self.start_angle_index + i) * self.angle_increment

            p1 = Point()
            p1.x, p1.y, p1.z = 0.0, 0.0, 0.05

            p2 = Point()
            p2.x = float(dist * np.cos(angle))
            p2.y = float(dist * np.sin(angle))
            p2.z = 0.05

            marker.points.append(p1)
            marker.points.append(p2)

        self.target_zone_pub.publish(marker)

    def publish_processed_lines(self, free_space_ranges):
        lines = Marker()
        lines.header.frame_id = "ego_racecar/laser_model"
        lines.header.stamp = self.get_clock().now().to_msg()
        lines.ns = "processed_lines"
        lines.id = 1
        lines.type = Marker.LINE_LIST
        lines.action = Marker.ADD

        lines.scale.x = 0.01

        lines.color.a = 0.5
        lines.color.r = 0.0
        lines.color.g = 0.8
        lines.color.b = 1.0

        step = 5
        for i in range(0, len(free_space_ranges), step):
            dist = free_space_ranges[i]

            angle = self.angle_min + (i + self.start_angle_index) * self.angle_increment

            start_pt = Point()
            start_pt.x, start_pt.y, start_pt.z = 0.0, 0.0, 0.0

            end_pt = Point()
            end_pt.x = float(dist * np.cos(angle))
            end_pt.y = float(dist * np.sin(angle))
            end_pt.z = 0.0

            lines.points.append(start_pt)
            lines.points.append(end_pt)

        self.processed_lines_pub.publish(lines)

    def publish_marker(self, angle, dist):
        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser_model"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = dist * np.cos(angle)
        marker.pose.position.y = dist * np.sin(angle)
        marker.pose.position.z = 0.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("Gap follow Initialized")
    np.set_printoptions(threshold=np.inf,suppress=True)
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
