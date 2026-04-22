// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well

#include "rrt/rrt.h"

RRT::~RRT() {
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "RRT shutting down");
}

RRT::RRT() : rclcpp::Node("rrt_node"), gen((std::random_device())()),
             x_dist(-5.0, 5.0), y_dist(-5.0, 5.0) {

    // Parameters (mirrored from Python)
    wheelbase_      = 0.3;
    lookahead_dist_ = 1.5;
    max_iter_       = 5;
    step_size_      = 0.5;
    goal_tolerance_ = 0.5;

    // Occupancy grid
    grid_res_      = 0.05;
    grid_width_    = static_cast<int>(24.0 / grid_res_);    // 480
    grid_height_   = static_cast<int>(24.0 / grid_res_);    // 480
    grid_x_offset_ = static_cast<int>(12.0 / grid_res_);    // 240
    grid_y_offset_ = static_cast<int>(12.0 / grid_res_);    // 240
    occupancy_grid_.assign(grid_height_ * grid_width_, false);

    load_waypoints("/home/team10/f1tenth_ws/src/pure_pursuit/waypoints/path_race3.csv");

    // Subscribers
    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/pf/viz/inferred_pose", 100,
        std::bind(&RRT::pose_callback, this, std::placeholders::_1));
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 100,
        std::bind(&RRT::scan_callback, this, std::placeholders::_1));

    // Publishers
    drive_pub_      = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
    tree_pub_       = this->create_publisher<visualization_msgs::msg::MarkerArray>("/rrt_tree", 10);
    path_pub_       = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_path", 10);
    marker_pub_     = this->create_publisher<visualization_msgs::msg::Marker>("/visualization_marker", 10);
    cur_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/cur_marker", 10);
    grid_viz_pub_   = this->create_publisher<visualization_msgs::msg::Marker>("/occupancy_grid_viz", 10);
    sample_viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/sample_bounds_viz", 10);

    waypoint_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&RRT::publish_all_waypoints, this));

    RCLCPP_INFO(rclcpp::get_logger("RRT"), "Created new RRT Object.");
}

void RRT::load_waypoints(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Cannot open waypoints file: %s", path.c_str());
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<double> vals;
        while (std::getline(ss, token, ',')) {
            try { vals.push_back(std::stod(token)); } catch (...) {}
        }
        if (vals.size() >= 2) {
            raw_waypoints_.push_back({vals[0], vals[1]});
            waypoints_.push_back({vals[0], vals[1]});
        }
    }
    RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints", waypoints_.size());
}

void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    std::vector<bool> new_grid(grid_height_ * grid_width_, false);
    int car_radius_cells = static_cast<int>(0.1 / grid_res_);
    double min_safe_dist = 0.3;

    int n = static_cast<int>(scan_msg->ranges.size());
    for (int i = 0; i < n; i++) {
        float r = scan_msg->ranges[i];
        if (!std::isfinite(r) || r <= min_safe_dist || r >= scan_msg->range_max) continue;

        double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
        double x = r * std::cos(angle);
        double y = r * std::sin(angle);

        int ix = static_cast<int>(x / grid_res_) + grid_x_offset_;
        int iy = static_cast<int>(y / grid_res_) + grid_y_offset_;

        for (int dx = -car_radius_cells; dx <= car_radius_cells; dx++) {
            for (int dy = -car_radius_cells; dy <= car_radius_cells; dy++) {
                if (dx * dx + dy * dy <= car_radius_cells * car_radius_cells) {
                    int nx = ix + dx, ny = iy + dy;
                    if (nx >= 0 && nx < grid_height_ && ny >= 0 && ny < grid_width_) {
                        new_grid[nx * grid_width_ + ny] = true;
                    }
                }
            }
        }
    }

    // Deadzone behind the car
    int deadzone_offset = static_cast<int>(-0.5 / grid_res_);
    int rear_boundary = grid_x_offset_ + deadzone_offset;
    for (int ix = 0; ix < rear_boundary; ix++) {
        for (int iy = 0; iy < grid_width_; iy++) {
            new_grid[ix * grid_width_ + iy] = true;
        }
    }

    occupancy_grid_ = new_grid;
    publish_grid_markers();
}

void RRT::publish_grid_markers() {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "laser";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "occupancy_grid";
    marker.id = 100;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.color.r = 1.0f;
    marker.color.a = 0.5f;

    for (int ix = 0; ix < grid_height_; ix++) {
        for (int iy = 0; iy < grid_width_; iy++) {
            if (occupancy_grid_[ix * grid_width_ + iy]) {
                geometry_msgs::msg::Point p;
                p.x = (ix - grid_x_offset_) * grid_res_;
                p.y = (iy - grid_y_offset_) * grid_res_;
                marker.points.push_back(p);
            }
        }
    }
    grid_viz_pub_->publish(marker);
}

std::array<double, 2> RRT::get_lookahead_point(const geometry_msgs::msg::Pose &current_pose) {
    double cur_x = current_pose.position.x;
    double cur_y = current_pose.position.y;
    const auto &q = current_pose.orientation;
    double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z));

    double best_dist = 1e9;
    int best_idx = 0;
    bool found_valid = false;

    for (int i = 0; i < static_cast<int>(waypoints_.size()); i++) {
        double dx = waypoints_[i][0] - cur_x;
        double dy = waypoints_[i][1] - cur_y;
        double d = std::hypot(dx, dy);
        double x_local = dx * std::cos(yaw) + dy * std::sin(yaw);
        if (d >= lookahead_dist_ && x_local > 0.0) {
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
                found_valid = true;
            }
        }
    }

    if (found_valid) return waypoints_[best_idx];

    // Fallback: closest waypoint
    double min_d = 1e9;
    for (int i = 0; i < static_cast<int>(waypoints_.size()); i++) {
        double d = std::hypot(waypoints_[i][0] - cur_x, waypoints_[i][1] - cur_y);
        if (d < min_d) { min_d = d; best_idx = i; }
    }
    return waypoints_[best_idx];
}

std::array<double, 2> RRT::transform_to_vehicle_frame(
    const geometry_msgs::msg::Pose &current_pose,
    const std::array<double, 2> &goal_point) {

    double cur_x = current_pose.position.x;
    double cur_y = current_pose.position.y;
    const auto &q = current_pose.orientation;
    double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z));

    double dx = goal_point[0] - cur_x;
    double dy = goal_point[1] - cur_y;

    double cos_yaw = std::cos(-yaw);
    double sin_yaw = std::sin(-yaw);
    return {dx * cos_yaw - dy * sin_yaw,
            dx * sin_yaw + dy * cos_yaw};
}

void RRT::pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg) {
    const auto &current_pose = pose_msg->pose;
    auto goal_point = get_lookahead_point(current_pose);
    publish_cur_waypoint(goal_point);

    auto transformed_goal = transform_to_vehicle_frame(current_pose, goal_point);
    double goal_x = transformed_goal[0];
    double goal_y = transformed_goal[1];

    // Initialize tree with root at vehicle origin
    RRT_Node root;
    root.x = 0.0; root.y = 0.0;
    root.is_root = true;
    root.parent = -1;
    root.cost = 0.0;
    std::vector<RRT_Node> tree = {root};

    bool path_found = false;
    RRT_Node goal_node = root;

    for (int i = 0; i < max_iter_; i++) {
        auto sampled_pt = sample();
        int nearest_idx = nearest(tree, sampled_pt);
        auto new_node = steer(tree[nearest_idx], sampled_pt);

        if (!check_collision(tree[nearest_idx], new_node)) {
            new_node.parent = nearest_idx;
            tree.push_back(new_node);
            goal_node = tree.back();

            if (is_goal(new_node, goal_x, goal_y)) {
                path_found = true;
                break;
            }
        }
    }

    if (!path_found && tree.size() > 1) {
        double min_d = 1e9;
        for (auto &n : tree) {
            double d = std::hypot(n.x - goal_x, n.y - goal_y);
            if (d < min_d) { min_d = d; goal_node = n; }
        }
    }

    auto path = find_path(tree, goal_node);
    if (path.size() > 1) {
        int target_idx = std::min(2, static_cast<int>(path.size()) - 1);
        const auto &target_node = path[target_idx];

        double L = target_node.x * target_node.x + target_node.y * target_node.y;
        double curvature = (L > 1e-6) ? (2.0 * target_node.y / L) : 0.0;

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        double steering_angle = std::max(-0.4, std::min(0.4, curvature * wheelbase_));
        double abs_angle = std::abs(steering_angle);
        double speed = 0.5;
        if (abs_angle < 10.0) speed = 0.75;
        else if (abs_angle < 20.0) speed = 1.0;

        drive_msg.drive.steering_angle = static_cast<float>(steering_angle);
        drive_msg.drive.speed = static_cast<float>(speed);
        drive_pub_->publish(drive_msg);

        visualize(goal_x, goal_y, tree, path);
    }
}

std::vector<double> RRT::sample() {
    return {x_dist(gen), y_dist(gen)};
}

int RRT::nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point) {
    int nearest_node = 0;
    double min_d = 1e9;
    for (int i = 0; i < static_cast<int>(tree.size()); i++) {
        double d = std::hypot(tree[i].x - sampled_point[0], tree[i].y - sampled_point[1]);
        if (d < min_d) { min_d = d; nearest_node = i; }
    }
    return nearest_node;
}

RRT_Node RRT::steer(RRT_Node &nearest_node, std::vector<double> &sampled_point) {
    RRT_Node new_node;
    new_node.cost = 0.0;
    new_node.parent = -1;

    double dx = sampled_point[0] - nearest_node.x;
    double dy = sampled_point[1] - nearest_node.y;
    double d = std::hypot(dx, dy);

    if (d < step_size_) {
        new_node.x = sampled_point[0];
        new_node.y = sampled_point[1];
    } else {
        double theta = std::atan2(dy, dx);
        new_node.x = nearest_node.x + step_size_ * std::cos(theta);
        new_node.y = nearest_node.y + step_size_ * std::sin(theta);
    }
    return new_node;
}

bool RRT::check_collision(RRT_Node &nearest_node, RRT_Node &new_node) {
    double d = std::hypot(new_node.x - nearest_node.x, new_node.y - nearest_node.y);
    int steps = std::max(1, static_cast<int>(d / (grid_res_ / 2.0)));

    for (int i = 0; i <= steps; i++) {
        double t = static_cast<double>(i) / steps;
        double x = nearest_node.x + (new_node.x - nearest_node.x) * t;
        double y = nearest_node.y + (new_node.y - nearest_node.y) * t;

        int ix = static_cast<int>(x / grid_res_) + grid_x_offset_;
        int iy = static_cast<int>(y / grid_res_) + grid_y_offset_;

        if (ix >= 0 && ix < grid_height_ && iy >= 0 && iy < grid_width_) {
            if (occupancy_grid_[ix * grid_width_ + iy]) return true;
        } else {
            return true;
        }
    }
    return false;
}

bool RRT::is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y) {
    return std::hypot(latest_added_node.x - goal_x,
                      latest_added_node.y - goal_y) < goal_tolerance_;
}

std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node) {
    std::vector<RRT_Node> path;
    // latest_added_node is a copy of a tree node; trace back via parent indices
    path.push_back(latest_added_node);
    int parent_idx = latest_added_node.parent;
    while (parent_idx >= 0) {
        path.push_back(tree[parent_idx]);
        parent_idx = tree[parent_idx].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

void RRT::visualize(double gx, double gy,
                    const std::vector<RRT_Node> &tree,
                    const std::vector<RRT_Node> &path) {
    (void)gx; (void)gy;

    // Tree
    visualization_msgs::msg::MarkerArray tree_markers;
    visualization_msgs::msg::Marker t_marker;
    t_marker.header.frame_id = "laser";
    t_marker.header.stamp = this->get_clock()->now();
    t_marker.ns = "tree";
    t_marker.id = 0;
    t_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    t_marker.action = visualization_msgs::msg::Marker::ADD;
    t_marker.scale.x = 0.02;
    t_marker.color.g = 1.0f;
    t_marker.color.a = 0.5f;
    for (const auto &node : tree) {
        if (node.parent >= 0) {
            geometry_msgs::msg::Point p1, p2;
            p1.x = node.x; p1.y = node.y;
            p2.x = tree[node.parent].x; p2.y = tree[node.parent].y;
            t_marker.points.push_back(p1);
            t_marker.points.push_back(p2);
        }
    }
    tree_markers.markers.push_back(t_marker);
    tree_pub_->publish(tree_markers);

    // Path
    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = "laser";
    path_marker.header.stamp = this->get_clock()->now();
    path_marker.ns = "path";
    path_marker.id = 1;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.scale.x = 0.05;
    path_marker.color.b = 1.0f;
    path_marker.color.a = 1.0f;
    for (const auto &node : path) {
        geometry_msgs::msg::Point p;
        p.x = node.x; p.y = node.y;
        path_marker.points.push_back(p);
    }
    path_pub_->publish(path_marker);

    // Sampling bounds box
    visualization_msgs::msg::Marker sample_marker;
    sample_marker.header.frame_id = "laser";
    sample_marker.header.stamp = this->get_clock()->now();
    sample_marker.ns = "sample_bounds";
    sample_marker.id = 2;
    sample_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    sample_marker.action = visualization_msgs::msg::Marker::ADD;
    sample_marker.scale.x = 0.05;
    sample_marker.color.r = 1.0f;
    sample_marker.color.g = 1.0f;
    sample_marker.color.a = 1.0f;
    std::vector<std::pair<double, double>> bounds = {
        {-5.0, -5.0}, {5.0, -5.0}, {5.0, 5.0}, {-5.0, 5.0}, {-5.0, -5.0}};
    for (auto &b : bounds) {
        geometry_msgs::msg::Point p;
        p.x = b.first; p.y = b.second;
        sample_marker.points.push_back(p);
    }
    sample_viz_pub_->publish(sample_marker);
}

void RRT::publish_cur_waypoint(const std::array<double, 2> &goal_point) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->get_clock()->now();
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = goal_point[0];
    marker.pose.position.y = goal_point[1];
    marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2;
    marker.color.r = 1.0f; marker.color.a = 1.0f;
    cur_marker_pub_->publish(marker);
}

void RRT::publish_all_waypoints() {
    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = "map";
    path_marker.header.stamp = this->get_clock()->now();
    path_marker.ns = "smooth_path";
    path_marker.id = 0;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.scale.x = 0.05;
    path_marker.color.g = 1.0f;
    path_marker.color.a = 1.0f;
    for (const auto &p : waypoints_) {
        geometry_msgs::msg::Point pt;
        pt.x = p[0]; pt.y = p[1];
        path_marker.points.push_back(pt);
    }
    marker_pub_->publish(path_marker);

    visualization_msgs::msg::Marker raw_marker;
    raw_marker.header.frame_id = "map";
    raw_marker.header.stamp = this->get_clock()->now();
    raw_marker.ns = "raw_points";
    raw_marker.id = 1;
    raw_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    raw_marker.action = visualization_msgs::msg::Marker::ADD;
    raw_marker.scale.x = 0.15; raw_marker.scale.y = 0.15; raw_marker.scale.z = 0.15;
    raw_marker.color.r = 1.0f; raw_marker.color.a = 1.0f;
    for (const auto &p : raw_waypoints_) {
        geometry_msgs::msg::Point pt;
        pt.x = p[0]; pt.y = p[1];
        raw_marker.points.push_back(pt);
    }
    marker_pub_->publish(raw_marker);
}

// RRT* stubs
double RRT::cost(std::vector<RRT_Node> &tree, RRT_Node &node) {
    (void)tree; (void)node;
    return 0.0;
}

double RRT::line_cost(RRT_Node &n1, RRT_Node &n2) {
    (void)n1; (void)n2;
    return 0.0;
}

std::vector<int> RRT::near(std::vector<RRT_Node> &tree, RRT_Node &node) {
    (void)tree; (void)node;
    return {};
}
