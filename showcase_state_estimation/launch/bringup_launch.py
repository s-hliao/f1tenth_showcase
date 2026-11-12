"""Car bringup with llampc-style state estimation (port of llampc/launch/bringup_launch.py):

    f1tenth_system  f1tenth_stack/launch/bringup_launch.py   VESC driver, lidar, ackermann_mux, joy
    particle_filter particle_filter/launch/localize_launch.py -> /pf/pose/odom
    imu_zupt_prep   /sensors/imu/raw + /odom                  -> /sensors/imu/data, /zupt
    ekf_node        PF pose + IMU + ZUPT                      -> /odometry/filtered

    ros2 launch showcase_state_estimation bringup_launch.py
    ros2 launch showcase_state_estimation bringup_launch.py localize:=false source:=mocap mocap_topic:=/car1/pose

f1tenth_stack and particle_filter are the car's own, distro-specific packages (cloned into the repo-level
f1tenth_system/ and particle_filter/ placeholders, or provided by an underlay workspace). A missing one is skipped
with a warning, so the EKF part can still be brought up on its own.

On a showcase car this is included by showcase_fleet's car_launch.py, which also checks that the shell is on the
car's loopback DDS domain. Run it directly only from a shell that sourced `dds_env.sh car <car>`.
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _share(pkg):
    try:
        return get_package_share_directory(pkg)
    except PackageNotFoundError:
        return None


def _setup(context):
    get = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    on = lambda k: get(k).strip().lower() in ('1', 'true', 'yes', 'on')  # noqa: E731
    acts = []
    for flag, pkg, rel in (('bringup', 'f1tenth_stack', 'launch/bringup_launch.py'),
                           ('localize', 'particle_filter', 'launch/localize_launch.py')):
        if not on(flag):
            continue
        share = _share(pkg)
        path = os.path.join(share, rel) if share else ''
        if path and os.path.isfile(path):
            acts.append(LogInfo(msg=f'[bringup] {pkg}: {path}'))
            acts.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(path)))
        else:
            acts.append(LogInfo(msg=f'[bringup] WARNING: {flag}:=true but {pkg}/{rel} is not installed '
                                    f'(clone it into the repo-level placeholder or source a workspace that has it)'))
    if on('ekf'):
        acts.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(_share('showcase_state_estimation'), 'launch',
                                                       'ekf_launch.py')),
            launch_arguments={k: get(k) for k in ('source', 'zupt', 'publish_static_tf', 'cg_x', 'imu_x', 'imu_z',
                                                  'mocap_topic', 'yaw_sign', 'gyro_scale', 'bias_calib_s')}.items()))
    return acts


def generate_launch_description():
    args = [
        ('bringup', 'true', 'include f1tenth_stack bringup_launch.py'),
        ('localize', 'true', 'include particle_filter localize_launch.py'),
        ('ekf', 'true', 'run the EKF (+ ZUPT prep, static TFs)'),
        ('source', 'pf', 'EKF input: pf (particle filter) | dr (dead reckoning: wheel speed + gyro) | mocap'),
        ('yaw_sign', '1.0', 'gyro z sign (dr; check with imu_check)'),
        ('gyro_scale', '0.017453292519943295', 'raw gyro -> rad/s (VESC: deg/s)'),
        ('bias_calib_s', '3.0', 'dr: gyro bias averaging at standstill [s]'),
        ('zupt', 'true', 'run imu_zupt_prep'),
        ('publish_static_tf', 'true', 'publish base_link->cg and base_link->imu_link (stock bringup has neither)'),
        ('cg_x', '0.15', 'base_link->cg x [m]'),
        ('imu_x', '0.065', 'base_link->imu_link x [m]'),
        ('imu_z', '0.09', 'base_link->imu_link z [m]'),
        ('mocap_topic', '/f1tenth/pose', 'rigid-body PoseStamped (source:=mocap)'),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args] +
                             [OpaqueFunction(function=_setup)])
