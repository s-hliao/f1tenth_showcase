"""
depth_fusion.launch.py

Launches depth_fusion_node.  The node opens RealSense directly via
pyrealsense2 — no separate realsense2_camera ROS driver is required.

Preferred launch method (handles conda + ROS2 env):
    bash run_depth_fusion.sh [flags]

    --anchor-row R        Fix anchor to row R; -1 = auto-select (default)
    --max-gt-depth M      GT cap in metres (default 3.0)
    --max-depth M         Clip predicted depth (default 20.0)
    --inlier-rel-tol T    RANSAC inlier threshold (default 0.30)
    --display-vmax V      Colormap max, 0 = auto (default 0.0)
    --cam-fps N           RealSense FPS (default 30)

Direct ros2 launch (only if conda env is already active):
    ros2 launch depth_fusion depth_fusion.launch.py
    ros2 launch depth_fusion depth_fusion.launch.py anchor_row:=430 max_gt_depth:=3.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        # -1 = auto-select the row with the most GT depth spread per frame
        DeclareLaunchArgument("anchor_row",     default_value="-1",
                              description="Anchor row for scale fit. -1 = auto."),
        DeclareLaunchArgument("max_gt_depth",   default_value="3.0",
                              description="RealSense GT cap in metres (noisy beyond ~3 m)."),
        DeclareLaunchArgument("max_depth",      default_value="20.0",
                              description="Clip predicted metric depth at this value (m)."),
        DeclareLaunchArgument("inlier_rel_tol", default_value="0.30",
                              description="RANSAC inlier relative-error threshold."),
        DeclareLaunchArgument("display_vmax",   default_value="0.0",
                              description="Colormap max in m. 0 = auto from GT 95th pct."),
        DeclareLaunchArgument("cam_width",      default_value="848"),
        DeclareLaunchArgument("cam_height",     default_value="480"),
        DeclareLaunchArgument("cam_fps",        default_value="30"),
    ]

    node = Node(
        package="depth_fusion",
        executable="depth_fusion_node",
        name="depth_fusion",
        output="screen",
        parameters=[{
            "anchor_row":     LaunchConfiguration("anchor_row"),
            "max_gt_depth":   LaunchConfiguration("max_gt_depth"),
            "max_depth":      LaunchConfiguration("max_depth"),
            "inlier_rel_tol": LaunchConfiguration("inlier_rel_tol"),
            "display_vmax":   LaunchConfiguration("display_vmax"),
            "cam_width":      LaunchConfiguration("cam_width"),
            "cam_height":     LaunchConfiguration("cam_height"),
            "cam_fps":        LaunchConfiguration("cam_fps"),
        }],
    )

    return LaunchDescription(args + [node])
