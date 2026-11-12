"""Ground station (laptop): one LLA estimator per car + the live GUI. Run after `source dds_env.sh laptop`
inside the LLA (JAX) environment.

    ros2 launch showcase_fleet ground_launch.py                         # car1 + car2
    ros2 launch showcase_fleet ground_launch.py cars:=car2              # one car only
    ros2 launch showcase_fleet ground_launch.py track:=venue_cw line:=raceline_fast.csv tube_radius:=0.25 log:=true

Nothing here can drive a car: the only thing sent to the cars is start/stop from the GUI buttons.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    get = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    env = os.environ
    acts = []
    if env.get('SHOWCASE_ROLE') != 'laptop':
        acts.append(LogInfo(msg='[showcase] WARNING: dds_env.sh laptop not sourced; the cars will not be visible'))
    cars = [c.strip() for c in get('cars').split(',') if c.strip()]
    if get('estimator').lower() in ('1', 'true', 'yes'):
        for car in cars:
            acts.append(Node(package='showcase_fleet', executable='lla_estimator', name='lla_estimator',
                             namespace=car, output='screen',
                             parameters=[{'log': get('log').lower() in ('1', 'true', 'yes'),
                                          'v_min': float(get('v_min'))}]))
    acts.append(Node(package='showcase_fleet', executable='ground_station', name='ground_station', output='screen',
                     parameters=[{'cars': ','.join(cars), 'track': get('track'), 'line': get('line'),
                                  'raceline_file': get('raceline_file'),
                                  'tube_radius': float(get('tube_radius')), 'trail_s': float(get('trail_s')),
                                  'pose_topic': get('pose_topic')}]))
    return acts


def generate_launch_description():
    args = [
        ('cars', os.environ.get('SHOWCASE_CARS', 'car1,car2'), 'comma-separated car names'),
        ('track', 'venue_ccw', 'frenet_baseline/waypoints/<track>'),
        ('line', 'raceline_stable.csv', 'raceline CSV inside the track folder'),
        ('raceline_file', '', 'explicit raceline CSV (overrides track/line)'),
        ('tube_radius', '0.3', 'half-width of the shaded tube around the raceline [m]'),
        ('trail_s', '3.0', 'trail length [s]'),
        ('pose_topic', 'odom', "relayed pose shown: 'odom' (EKF) or 'pf_odom' (raw PF)"),
        ('estimator', 'true', 'run the LLA estimators'),
        ('v_min', '0.3', 'LLA bank pauses below this speed [m/s]'),
        ('log', 'false', 'LLA estimators write ~/.ros/log/showcase_<car>_lla_*.npz (viz_ref_fiala replay)'),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args] +
                             [OpaqueFunction(function=_setup)])
