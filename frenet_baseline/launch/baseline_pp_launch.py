"""
Frenet pure pursuit Baseline (+ barrier-state avoidance).

    ros2 launch frenet_baseline baseline_pp_launch.py                       # car, venue ccw, stable line
    ros2 launch frenet_baseline baseline_pp_launch.py track:=venue_cw line:=raceline_fast.csv
    ros2 launch frenet_baseline baseline_pp_launch.py avoid:=true speed_scale:=0.9
    ros2 launch frenet_baseline baseline_pp_launch.py config:=sim track:=iros_2026_sim

Arguments
    config        car | sim       -> share/frenet_baseline/config/baseline_pp_<config>.yaml
    params_file   path            -> use this YAML instead (e.g. the src copy, to tune without rebuilding)
    track         folder in share/frenet_baseline/waypoints/
    line          CSV in that folder, or an absolute path to any raceline CSV
    avoid         true | false    -> AVOID_ENABLE      (empty = value from the YAML)
    speed_scale   float           -> SPEED_SCALE       (empty = value from the YAML)
    steer_law     PURE_PURSUIT | FRENET_PD            (empty = value from the YAML)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory('frenet_baseline')
    arg = lambda name: LaunchConfiguration(name).perform(context).strip()

    params_file = arg('params_file') or os.path.join(share, 'config', f'baseline_pp_{arg("config")}.yaml')
    line = arg('line')
    raceline = line if os.path.isabs(line) else os.path.join(share, 'waypoints', arg('track'), line)
    if not os.path.isfile(params_file):
        raise RuntimeError(f'params file not found: {params_file}')
    if not os.path.isfile(raceline):
        raise RuntimeError(f'raceline not found: {raceline}')

    overrides = {'raceline_file': raceline, 'track_id': arg('track')}
    if arg('avoid'):
        overrides['AVOID_ENABLE'] = arg('avoid').lower() in ('true', '1', 'yes')
    if arg('speed_scale'):
        overrides['SPEED_SCALE'] = float(arg('speed_scale'))
    if arg('steer_law'):
        overrides['STEER_LAW'] = arg('steer_law').upper()

    return [Node(
        package='frenet_baseline',
        executable='baseline_pp_node.py',
        name='baseline_pp',
        output='screen',
        emulate_tty=True,
        parameters=[params_file, overrides],   # overrides win over the YAML
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='car', description='car | sim'),
        DeclareLaunchArgument('params_file', default_value='', description='YAML path; overrides config'),
        DeclareLaunchArgument('track', default_value='venue_ccw', description='folder in waypoints/'),
        DeclareLaunchArgument('line', default_value='raceline_stable.csv', description='CSV name or absolute path'),
        DeclareLaunchArgument('avoid', default_value='', description='true | false (empty = YAML)'),
        DeclareLaunchArgument('speed_scale', default_value='', description='float (empty = YAML)'),
        DeclareLaunchArgument('steer_law', default_value='', description='PURE_PURSUIT | FRENET_PD (empty = YAML)'),
        OpaqueFunction(function=launch_setup),
    ])
