from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scan_topic",    default_value="/scan"),
        DeclareLaunchArgument("output_dir",
                              default_value=os.path.expanduser("~/calibration_data")),
        DeclareLaunchArgument("use_realsense", default_value="true"),
        DeclareLaunchArgument("camera_topic",  default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("cam_width",     default_value="1280"),
        DeclareLaunchArgument("cam_height",    default_value="720"),
        DeclareLaunchArgument("cam_fps",       default_value="30"),

        Node(
            package="calibration",
            executable="collect_node",
            name="calibration_collect",
            parameters=[{
                "scan_topic":    LaunchConfiguration("scan_topic"),
                "output_dir":    LaunchConfiguration("output_dir"),
                "use_realsense": LaunchConfiguration("use_realsense"),
                "camera_topic":  LaunchConfiguration("camera_topic"),
                "cam_width":     LaunchConfiguration("cam_width"),
                "cam_height":    LaunchConfiguration("cam_height"),
                "cam_fps":       LaunchConfiguration("cam_fps"),
            }],
            output="screen",
        ),
    ])
