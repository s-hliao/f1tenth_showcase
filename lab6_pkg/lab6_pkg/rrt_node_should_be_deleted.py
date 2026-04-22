#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point, Pose, PointStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from scipy import interpolate
from scipy import linalg as LA
import math
import csv

# TODO: import as you need
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from scipy import interpolate

# TF transformations for euler angles
def euler_from_quaternion(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counter-clockwise)
    pitch is rotation around y in radians (counter-clockwise)
    yaw is rotation around z in radians (counter-clockwise)
    """
    w, x, y, z = q.w, q.x, q.y, q.z
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

# class def for tree nodes
# It's up to you if you want to use this
class RRTNode(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# class def for RRT
class RRT(Node):
    def __init__(self):
        super().__init__("rrt_node")
        
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        pose_topic = "/ego_racecar/odom"
        scan_topic = "/scan"

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.
        
        # Parameters
        self.wheelbase = 0.33
        self.lookahead_dist = 2.0
        self.interpolate_interval = 10000
        
        # Occupancy grid parameters
        self.grid_res = 0.1
        self.grid_width = int(8.0 / self.grid_res)
        self.grid_height = int(7.0 / self.grid_res)
        self.grid_y_offset = int(4.0 / self.grid_res)
        self.grid_x_offset = int(1.0 / self.grid_res)
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)

        # RRT parameters
        self.max_iter = 400
        self.step_size = 0.5
        self.goal_tolerance = 0.5

        # Waypoints
        self.waypoints = self.load_waypoints("/home/team10/f1tenth_ws/src/pure_pursuit/waypoints/path_race3.csv")

        # RRT attributes
        self.raw_waypoints = self.waypoints # Lab 5 uses raw_waypoints for MarkerArray

        # TODO: create subscribers
        self.pose_pub = self.create_subscription(
            PoseStamped, "/pf/viz/inferred_pose", self.pose_callback, 1
        )
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 1)

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need
        self.publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.tree_pub = self.create_publisher(MarkerArray, "/rrt_tree", 10)
        self.path_pub = self.create_publisher(Marker, "/rrt_path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker", 10)
        self.current_mark_pub = self.create_publisher(Marker, "/cur_marker", 10)
        self.grid_viz_pub = self.create_publisher(Marker, "/occupancy_grid_viz", 10)
        self.sample_viz_pub = self.create_publisher(Marker, "/sample_bounds_viz", 10)

        # Timer
        self.timer = self.create_timer(0.5, self.publish_all_waypoints)

        # class attributes
        # TODO: maybe create your occupancy grid here

    def load_waypoints(self, path):
        data = np.genfromtxt(path, delimiter=",", skip_header=0)
        x, y = data[:, 0], data[:, 1]
        tick, _ = interpolate.splprep([x, y], s=0.0, k=2)
        points = np.linspace(0, 1, self.interpolate_interval)
        x_aug, y_aug = interpolate.splev(points, tick)
        out = np.zeros((self.interpolate_interval, 2))
        out[:, 0], out[:, 1] = x_aug, y_aug
        return out.tolist()

        # class attributes
        # TODO: maybe create your occupancy grid here

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        ranges = np.array(scan_msg.ranges)
        valid_idx = np.isfinite(ranges) & (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        
        x = ranges[valid_idx] * np.cos(angles[valid_idx])
        y = ranges[valid_idx] * np.sin(angles[valid_idx])

        new_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        car_radius_cells = int(0.3 / self.grid_res)
        ix = (x / self.grid_res).astype(int) + self.grid_x_offset
        iy = (y / self.grid_res).astype(int) + self.grid_y_offset

        valid_ixy = (ix >= 0) & (ix < self.grid_height) & (iy >= 0) & (iy < self.grid_width)
        ix, iy = ix[valid_ixy], iy[valid_ixy]

        for dx in range(-car_radius_cells, car_radius_cells + 1):
            for dy in range(-car_radius_cells, car_radius_cells + 1):
                if dx*dx + dy*dy <= car_radius_cells*car_radius_cells:
                    n_ix, n_iy = ix + dx, iy + dy
                    v = (n_ix >= 0) & (n_ix < self.grid_height) & (n_iy >= 0) & (n_iy < self.grid_width)
                    new_grid[n_ix[v], n_iy[v]] = True

        self.occupancy_grid = new_grid
        self.publish_grid_markers(new_grid)

    def publish_grid_markers(self, grid):
        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "occupancy_grid"
        marker.id = 100
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y = 0.05, 0.05
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
        
        indices = np.where(grid)
        for ix, iy in zip(indices[0], indices[1]):
            p = Point()
            p.x = float(ix - self.grid_x_offset) * self.grid_res
            p.y = float(iy - self.grid_y_offset) * self.grid_res
            marker.points.append(p)
        self.grid_viz_pub.publish(marker)

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        ## This is like lab5 for localization
        current_pose = pose_msg.pose.pose
        goal_point = self.get_lookahead_point(current_pose)
        
        self.publish_cur_waypoint(goal_point)
        
        transformed_goal = self.transform_to_vehicle_frame(current_pose, goal_point)
        goal_x, goal_y = transformed_goal[0], transformed_goal[1]

        # Initial tree
        root = RRTNode()
        root.x = 0.0
        root.y = 0.0
        root.is_root = True
        tree = [root]

        path_found = False
        goal_node = None
        
        for _ in range(self.max_iter):
            sampled_pt = self.sample()
            
            # 20% bias logic: override sampled point 
            if np.random.rand() < 0.2:
                sampled_pt = (goal_x, goal_y)

            nearest_idx = self.nearest(tree, sampled_pt)
            nearest_node = tree[nearest_idx]
            
            new_node = self.steer(nearest_node, sampled_pt)
            
            if not self.check_collision(nearest_node, new_node):
                new_node.parent = nearest_node
                tree.append(new_node)
                
                if self.is_goal(new_node, goal_x, goal_y):
                    path_found = True
                    goal_node = new_node
                    break
                    
        if (not path_found) and len(tree) > 1:
            dists = [np.hypot(n.x - goal_x, n.y - goal_y) for n in tree]
            goal_node = tree[np.argmin(dists)]
            
        if goal_node is not None:
            path = self.find_path(tree, goal_node)
            
            # pure pursuit on path
            if len(path) > 1:
                target_idx = min(2, len(path) - 1)
                target_node = path[target_idx]
                
                L = target_node.x**2 + target_node.y**2
                curvature = 2 * target_node.y / L
                
                drive_msg = AckermannDriveStamped()
                steering_angle = np.clip(
                    curvature * self.wheelbase, -0.4, 0.4
                )
                abs_angle = np.abs(steering_angle)
                speed = .5
                if abs_angle < 10:
                    speed = .75
                elif abs_angle < 20:
                    speed = 1.0
                drive_msg.drive.steering_angle = steering_angle
                drive_msg.drive.speed = speed
                self.publisher.publish(drive_msg)
                
            # Visualization
            self._visualize(goal_x, goal_y, tree, path)

    def transform_to_vehicle_frame(self, current_pose, goal_point):
        cur_x, cur_y = current_pose.position.x, current_pose.position.y

        q = current_pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))

        dx = goal_point[0] - cur_x
        dy = goal_point[1] - cur_y


        """
        x_local    cos(th) sin(th) | dx
                =
        y_local   -sin(th) cos(th) | dy

        (th) should be -yaw here because we are reversing from global to vehicle.
        """
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        transformed_x = dx * cos_yaw - dy * sin_yaw
        transformed_y = dx * sin_yaw + dy * cos_yaw

        return [transformed_x, transformed_y]

    def get_lookahead_point(self, current_pose):
        cur_x, cur_y = current_pose.position.x, current_pose.position.y
        ## We need heading to know if a point is ahead of us. So get orientation
        q = current_pose.orientation
        ## Quarternion, we want to avoid Gimbal lock: https://support.apdm.com/hc/en-us/articles/214504206-Converting-quaternions-to-Euler-angles
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))

        waypoints = np.array(self.waypoints)  #(N, 2)

        dists = np.sqrt((waypoints[:, 0] - cur_x) ** 2 + (waypoints[:, 1] - cur_y) ** 2)

        dx = waypoints[:, 0] - cur_x
        dy = waypoints[:, 1] - cur_y

        ## make sure that the point is ahead of us.
        x_local = dx * np.cos(yaw) + dy * np.sin(yaw)

        mask = (dists >= self.lookahead_dist) & (x_local > 0)

        valid_indices = np.where(mask)[0]

        if len(valid_indices) > 0:
            valid_dists = dists[valid_indices]
            best_idx = valid_indices[np.argmin(valid_dists)]
            return waypoints[best_idx].tolist()

        return waypoints[np.argmin(dists)].tolist()

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point
        """
        x = np.random.uniform(-5.0, 5.0)
        y = np.random.uniform(-5.0, 5.0)
        return (x, y)

    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        dists = [np.hypot(n.x - sampled_point[0], n.y - sampled_point[1]) for n in tree]
        nearest_node = int(np.argmin(dists))
        return nearest_node

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (RRTNode): new node created from steering
        """
        dx = sampled_point[0] - nearest_node.x
        dy = sampled_point[1] - nearest_node.y
        d = np.hypot(dx, dy)
        
        new_node = RRTNode()
        if d < self.step_size:
            new_node.x = sampled_point[0]
            new_node.y = sampled_point[1]
        else:
            theta = math.atan2(dy, dx)
            new_node.x = nearest_node.x + self.step_size * math.cos(theta)
            new_node.y = nearest_node.y + self.step_size * math.sin(theta)
        return new_node

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (RRTNode): nearest node on the tree
            new_node (RRTNode): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        d = np.hypot(new_node.x - nearest_node.x, new_node.y - nearest_node.y)
        steps = int(d / (self.grid_res / 2.0))
        steps = max(1, steps)
        
        for i in range(steps + 1):
            x = nearest_node.x + (new_node.x - nearest_node.x) * i / steps
            y = nearest_node.y + (new_node.y - nearest_node.y) * i / steps
            
            ix = int(x / self.grid_res) + self.grid_x_offset
            iy = int(y / self.grid_res) + self.grid_y_offset
            
            if 0 <= ix < self.grid_height and 0 <= iy < self.grid_width:
                if self.occupancy_grid[ix, iy]:
                    return True
            else:
                return True
        return False

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (RRTNode): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        return np.hypot(latest_added_node.x - goal_x, latest_added_node.y - goal_y) < self.goal_tolerance

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of RRTNodes connecting the starting point to
        the goal once the latest added RRTNode is close enough to the goal

        Args:
            tree ([]): current tree as a list of RRTNodes
            latest_added_node (RRTNode): latest added RRTNode in the tree
        Returns:
            path ([]): valid path as a list of RRTNodes
        """
        path = []
        curr = latest_added_node
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path

    def _visualize(self, gx, gy, tree, path):
        # Tree
        tree_marker = MarkerArray()
        t_marker = Marker()
        t_marker.header.frame_id = "ego_racecar/laser"
        t_marker.header.stamp = self.get_clock().now().to_msg()
        t_marker.ns = "tree"
        t_marker.id = 0
        t_marker.type = Marker.LINE_LIST
        t_marker.action = Marker.ADD
        t_marker.scale.x = 0.02
        t_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)
        for node in tree:
            if node.parent is not None:
                p1, p2 = Point(), Point()
                p1.x, p1.y = float(node.x), float(node.y)
                p2.x, p2.y = float(node.parent.x), float(node.parent.y)
                t_marker.points.extend([p1, p2])
        tree_marker.markers.append(t_marker)
        self.tree_pub.publish(tree_marker)

        # Path
        path_marker = Marker()
        path_marker.header.frame_id = "ego_racecar/laser"
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = "path"
        path_marker.id = 1
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05
        path_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        for node in path:
            p = Point()
            p.x, p.y = float(node.x), float(node.y)
            path_marker.points.append(p)
        self.path_pub.publish(path_marker)
        
        # Sampling bounds
        sample_marker = Marker()
        sample_marker.header = Header(frame_id="map")
        print(f'{sample_marker=}')
        sample_marker.header.stamp = self.get_clock().now().to_msg()
        sample_marker.ns = "sample_bounds"
        sample_marker.type = Marker.LINE_STRIP
        sample_marker.action = Marker.ADD
        sample_marker.scale.x = 0.05
        sample_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        bounds = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (-5.0, -5.0)]
        for bx, by in bounds:
            p = Point()
            p.x, p.y = float(bx), float(by)
            sample_marker.points.append(p)
        self.sample_viz_pub.publish(sample_marker)

    def publish_cur_waypoint(self, goal_point):
        marker = Marker()
        marker.header = Header(frame_id="map")
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_point[0])
        marker.pose.position.y = float(goal_point[1])
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        self.current_mark_pub.publish(marker)

    def publish_all_waypoints(self):
        marker_array = MarkerArray()
        for idx, (x, y) in enumerate(self.raw_waypoints):
            if idx % 50 != 0: continue
            marker = Marker()
            marker.header = Header(frame_id="map")
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.scale.x = marker.scale.y = marker.scale.z = 0.15
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (RRTNode): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (RRTNode): node at one end of the straight line
            n2 (RRTNode): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of RRTNodes
            node (RRTNode): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of RRTNodes
        """
        neighborhood = []
        return neighborhood

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
