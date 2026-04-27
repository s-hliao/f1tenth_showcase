#include "rclcpp/rclcpp.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"

class ReactiveFollowGap : public rclcpp::Node {
public:
    ReactiveFollowGap() : Node("reactive_node") {
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&ReactiveFollowGap::lidar_callback, this, std::placeholders::_1));
        drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/target_point_marker", 10);
        processed_lines_pub_ = create_publisher<visualization_msgs::msg::Marker>("/processed_scan_lines", 10);
        target_zone_pub_ = create_publisher<visualization_msgs::msg::Marker>("/target_zone_marker", 10);

        last_time_ = std::chrono::steady_clock::now();
    }

private:
    const int mean_window_size_ = 3;
    const float max_range_ = 5.0f;
    const float half_car_size_ = 0.4f;
    const float diff_value_ = 0.28f;
    const float considered_angle_ = M_PI / 2.0f;

    float angle_increment_ = 0.0f;
    float angle_min_ = 0.0f;
    int laser_count_ = 0;
    int start_angle_index_ = 0;
    int end_angle_index_ = 0;
    bool initialized_ = false;

    std::chrono::steady_clock::time_point last_time_;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr processed_lines_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_zone_pub_;

    // Clamp + mean-window convolution ("same" mode, zero-padded at edges)
    std::vector<float> preprocess_lidar(const std::vector<float>& ranges) {
        int n = ranges.size();
        std::vector<float> tmp(n);
        for (int i = 0; i < n; i++) {
            tmp[i] = (ranges[i] > max_range_) ? max_range_ : ranges[i];
        }
        int half = mean_window_size_ / 2;
        std::vector<float> result(n);
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = -half; j <= half; j++) {
                int idx = i + j;
                if (idx >= 0 && idx < n) sum += tmp[idx];
                // else: zero-pad, matches numpy "same" mode
            }
            result[i] = sum / mean_window_size_;
        }
        return result;
    }

    std::vector<float> disparity_extend(const std::vector<float>& ranges) {
        int n = ranges.size();
        std::vector<float> free_space(ranges);

        std::vector<int> indices;
        for (int i = 0; i < n - 1; i++) {
            if (std::abs(ranges[i + 1] - ranges[i]) > diff_value_) {
                indices.push_back(i);
            }
        }

        float left_closest  = *std::min_element(ranges.begin() + n / 2, ranges.end());
        float right_closest = *std::min_element(ranges.begin(), ranges.begin() + n / 2);

        const float tunnel_threshold = 0.4f;
        bool is_in_tunnel = (left_closest < tunnel_threshold) && (right_closest < tunnel_threshold);

        float active_half_width, max_angle_limit;
        if (is_in_tunnel) {
            RCLCPP_INFO(get_logger(), "tunnel detected!");
            active_half_width = half_car_size_ * 0.7f;
            max_angle_limit = M_PI / 2.0f;
        } else {
            active_half_width = half_car_size_;
            max_angle_limit = std::numeric_limits<float>::infinity();
        }

        for (int i : indices) {
            float left  = ranges[i];
            float right = (i + 1 < n) ? ranges[i + 1] : ranges[i];
            float closest = std::min(left, right);
            float ratio = std::max(-1.0f, std::min(1.0f, active_half_width / closest));
            float mask_angle = std::asin(ratio);
            mask_angle = std::max(0.0f, std::min(mask_angle, max_angle_limit));
            int mask_count = (int)std::ceil(mask_angle / angle_increment_);

            if (left < right) {
                int start = i;
                int end = std::min(i + 1 + mask_count, n);
                std::fill(free_space.begin() + start, free_space.begin() + end, 0.0f);
            } else {
                int start = std::max(i + 1 - mask_count, 0);
                int end = i + 1;
                std::fill(free_space.begin() + start, free_space.begin() + end, 0.0f);
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_time_).count() > 5.0) {
            RCLCPP_INFO(get_logger(), "disparity_extend debug tick");
            last_time_ = now;
        }

        return free_space;
    }

    // Mirrors Python: returns full range (the gap-splitting logic below it is dead code there too)
    std::pair<int, int> find_max_gap(const std::vector<float>& /*free_space_ranges*/) {
        return {0, laser_count_};
    }

    int find_best_point(int start_i, int end_i, const std::vector<float>& ranges) {
        int n = ranges.size();
        end_i = std::min(end_i, n - 1);
        int best = start_i;
        for (int i = start_i + 1; i <= end_i; i++) {
            if (ranges[i] > ranges[best]) best = i;
        }
        return best;
    }

    void lidar_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        const auto& raw = scan_msg->ranges;
        int total = (int)raw.size();

        if (!initialized_) {
            angle_increment_ = scan_msg->angle_increment;
            angle_min_ = scan_msg->angle_min;
            int end_idx   = (int)std::round(( considered_angle_ - angle_min_) / angle_increment_);
            int start_idx = (int)std::round((-considered_angle_ - angle_min_) / angle_increment_);
            start_angle_index_ = std::max(0, start_idx);
            end_angle_index_   = std::min(end_idx, total);
            laser_count_       = end_angle_index_ - start_angle_index_;
            initialized_ = true;
        }

        std::vector<float> ranges(raw.begin() + start_angle_index_,
                                   raw.begin() + end_angle_index_);

        auto proc       = preprocess_lidar(ranges);
        auto free_space = disparity_extend(proc);

        auto [start_idx, end_idx] = find_max_gap(free_space);

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = get_clock()->now();

        if (start_idx == -1 || end_idx == -1) {
            drive_msg.drive.speed = 0.0f;
            drive_pub_->publish(drive_msg);
            return;
        }

        int target_index = find_best_point(start_idx, end_idx, free_space);
        float angle = angle_min_ + angle_increment_ * (float)(target_index + start_angle_index_);

        float abs_angle_deg = std::abs(angle) * 180.0f / M_PI;
        float speed;
        if (abs_angle_deg < 10.0f)       speed = 3.0f;
        else if (abs_angle_deg < 20.0f)  speed = 2.0f;
        else                             speed = 1.0f;

        drive_msg.drive.steering_angle = angle;
        drive_msg.drive.speed = speed;
        drive_pub_->publish(drive_msg);

        float target_dist = proc[target_index];
        publish_processed_lines(free_space);
        publish_target_zone(target_index, free_space);
        publish_marker(angle, target_dist);
    }

    void publish_target_zone(int target_index, const std::vector<float>& ranges) {
        if (target_index < 0) return;
        int n = (int)ranges.size();

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "ego_racecar/laser_model";
        marker.header.stamp = get_clock()->now();
        marker.ns = "target_zone";
        marker.id = 2;
        marker.type  = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.02;
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0;

        int start = std::max(0, target_index - 10);
        int end   = std::min(n, target_index + 11);
        for (int i = start; i < end; i++) {
            float dist = ranges[i];
            if (dist <= 0.0f) continue;
            float ang = angle_min_ + (start_angle_index_ + i) * angle_increment_;
            geometry_msgs::msg::Point p1, p2;
            p1.x = 0.0; p1.y = 0.0; p1.z = 0.05;
            p2.x = dist * std::cos(ang); p2.y = dist * std::sin(ang); p2.z = 0.05;
            marker.points.push_back(p1);
            marker.points.push_back(p2);
        }
        target_zone_pub_->publish(marker);
    }

    void publish_processed_lines(const std::vector<float>& free_space_ranges) {
        visualization_msgs::msg::Marker lines;
        lines.header.frame_id = "ego_racecar/laser_model";
        lines.header.stamp = get_clock()->now();
        lines.ns = "processed_lines";
        lines.id = 1;
        lines.type  = visualization_msgs::msg::Marker::LINE_LIST;
        lines.action = visualization_msgs::msg::Marker::ADD;
        lines.scale.x = 0.01;
        lines.color.a = 0.5; lines.color.r = 0.0; lines.color.g = 0.8; lines.color.b = 1.0;

        int n = (int)free_space_ranges.size();
        for (int i = 0; i < n; i += 5) {
            float dist = free_space_ranges[i];
            float ang = angle_min_ + (i + start_angle_index_) * angle_increment_;
            geometry_msgs::msg::Point p1, p2;
            p1.x = 0.0; p1.y = 0.0; p1.z = 0.0;
            p2.x = dist * std::cos(ang); p2.y = dist * std::sin(ang); p2.z = 0.0;
            lines.points.push_back(p1);
            lines.points.push_back(p2);
        }
        processed_lines_pub_->publish(lines);
    }

    void publish_marker(float angle, float dist) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "ego_racecar/laser_model";
        marker.header.stamp = get_clock()->now();
        marker.type  = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = dist * std::cos(angle);
        marker.pose.position.y = dist * std::sin(angle);
        marker.pose.position.z = 0.0;
        marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2;
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;
        marker_pub_->publish(marker);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Gap follow Initialized");
    rclcpp::spin(std::make_shared<ReactiveFollowGap>());
    rclcpp::shutdown();
    return 0;
}
