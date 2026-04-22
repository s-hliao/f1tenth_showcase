// RRT assignment
// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <tf2_ros/transform_broadcaster.h>

using namespace std;

typedef struct RRT_Node {
    double x, y;
    double cost;   // only used for RRT*
    int parent;    // index of parent node in tree vector, -1 if root
    bool is_root = false;
} RRT_Node;


class RRT : public rclcpp::Node {
public:
    RRT();
    virtual ~RRT();
private:
    // Subscribers
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    // Publishers
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cur_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr grid_viz_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr sample_viz_pub_;

    rclcpp::TimerBase::SharedPtr waypoint_timer_;

    // Random generator
    std::mt19937 gen;
    std::uniform_real_distribution<> x_dist;
    std::uniform_real_distribution<> y_dist;

    // RRT parameters
    double wheelbase_;
    double lookahead_dist_;
    int max_iter_;
    double step_size_;
    double goal_tolerance_;

    // Occupancy grid (flat row-major: index = ix * grid_width_ + iy)
    double grid_res_;
    int grid_width_;
    int grid_height_;
    int grid_x_offset_;
    int grid_y_offset_;
    std::vector<bool> occupancy_grid_;

    // Waypoints
    std::vector<std::array<double, 2>> waypoints_;
    std::vector<std::array<double, 2>> raw_waypoints_;

    // Callbacks
    void pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg);
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);
    void publish_all_waypoints();

    // Helper methods
    void load_waypoints(const std::string &path);
    std::array<double, 2> get_lookahead_point(const geometry_msgs::msg::Pose &current_pose);
    std::array<double, 2> transform_to_vehicle_frame(
        const geometry_msgs::msg::Pose &current_pose,
        const std::array<double, 2> &goal_point);
    void publish_grid_markers();
    void publish_cur_waypoint(const std::array<double, 2> &goal_point);
    void visualize(double gx, double gy,
                   const std::vector<RRT_Node> &tree,
                   const std::vector<RRT_Node> &path);

    // RRT methods
    std::vector<double> sample();
    int nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point);
    RRT_Node steer(RRT_Node &nearest_node, std::vector<double> &sampled_point);
    bool check_collision(RRT_Node &nearest_node, RRT_Node &new_node);
    bool is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y);
    std::vector<RRT_Node> find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node);

    // RRT* methods
    double cost(std::vector<RRT_Node> &tree, RRT_Node &node);
    double line_cost(RRT_Node &n1, RRT_Node &n2);
    std::vector<int> near(std::vector<RRT_Node> &tree, RRT_Node &node);
};
