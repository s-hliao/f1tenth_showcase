"""Car side of the showcase, terminal 1. From a shell that ran `source dds_env.sh car <car1|car2>`:

    ros2 launch showcase_fleet car_launch.py              # bringup + PF + EKF + drive_gate + fleet_relay
    ros2 launch showcase_fleet lqr_launch.py              # terminal 2: the LQR, waits behind the gate
    ros2 run showcase_fleet carctl                        # terminal 3 (optional): s = start, space = stop

Everything runs on the car's LOOPBACK DDS domain (unreachable from Wi-Fi):
  showcase_state_estimation/bringup_launch.py   f1tenth_stack + particle_filter + llampc-style EKF
  drive_gate                                    start/stop latch: /showcase/drive_raw -> /drive
  fleet_relay                                   the only process that touches Wi-Fi (separate domain)
The gate boots STOPPED and holds zero until a start arrives from carctl (on the car) or the ground station.

controller:=bv3 (or <package>/<executable>) also starts the controller here, instead of from lqr_launch.py.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from showcase_fleet.common import check_car_env


def _truthy(v):
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _setup(context):
    get = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    car = check_car_env('car_launch')
    env = os.environ
    acts = [LogInfo(msg=f'[showcase] {car}: local domain {env["SHOWCASE_LOCAL_DOMAIN"]}, '
                        f'relay to net domain {env["SHOWCASE_NET_DOMAIN"]}')]

    dr = get('mode') == 'dr'          # dead reckoning: no particle filter, EKF integrates wheel speed + gyro
    on = _truthy
    parts = [p for p, f in (('f1tenth_stack', on(get('bringup'))),
                            ('particle_filter', on(get('localize')) and not dr)) if f]
    est = ('dead-reckoning EKF (wheel speed + gyro)' if dr else
           f"EKF ({get('ekf_source')} + IMU)") if on(get('ekf')) else 'no EKF'
    print('\n'.join([
        '',
        f'================================ SHOWCASE {car} ================================',
        ('  MODE      DEAD RECKONING: the pose starts at (0, 0) facing +x WHERE THE CAR STANDS NOW.\n'
         '            Keep the car STILL for ~3 s (gyro bias calibration). Relaunch to reset the origin.'
         if dr else '  MODE      particle filter localization (map frame)'),
        f"  STARTING  {', '.join(parts + [est, 'drive_gate'] + (['fleet_relay'] if on(get('relay')) else []))}",
        f"  NETWORK   driving stack on loopback DDS domain {env['SHOWCASE_LOCAL_DOMAIN']} (unreachable from Wi-Fi);",
        f"            fleet_relay mirrors it to Wi-Fi domain {env['SHOWCASE_NET_DOMAIN']} as /{car}/* for the "
        f"ground station.",
        '            Losing Wi-Fi / the laptop changes nothing on the car.',
        '  GATE      boots STOPPED: the motor gets zero until a START (lqr terminal "s", carctl, or ground station).',
        f"            speed control only (current/duty modes blocked); STOP ramps down at {get('stop_decel')} m/s^2;",
        '            controller silent for 0.2 s while running -> stop.',
        '  NEXT      T2:  ros2 run showcase_fleet lqr track:=<track> line:=<raceline.csv>' +
        (' pose:=dr' if dr else ''),
        '=' * 80, ''], ), flush=True)
    acts.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('showcase_state_estimation'),
                                                   'launch', 'bringup_launch.py')),
        launch_arguments={'bringup': get('bringup'), 'localize': 'false' if dr else get('localize'),
                          'ekf': get('ekf'), 'source': 'dr' if dr else get('ekf_source'),
                          'publish_static_tf': get('publish_static_tf'), 'yaw_sign': get('yaw_sign')}.items()))

    acts.append(Node(package='showcase_fleet', executable='drive_gate', name='drive_gate', output='screen',
                     parameters=[{'autostart': _truthy(get('autostart')),
                                  'stop_decel': float(get('stop_decel'))}]))
    if _truthy(get('relay')):
        relay_args = ['--car', car]
        if get('relay_config'):
            relay_args += ['--config', get('relay_config')]
        acts.append(Node(package='showcase_fleet', executable='fleet_relay', name='fleet_relay', output='screen',
                         arguments=relay_args, respawn=True, respawn_delay=2.0))

    if get('controller') not in ('', 'none'):
        acts.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('showcase_fleet'), 'launch',
                                                       'lqr_launch.py')),
            launch_arguments={'controller': get('controller'), 'params': get('controller_params')}.items()))
    return acts


def generate_launch_description():
    args = [
        ('mode', 'pf', 'pf (particle filter localization) | dr (dead reckoning from the start pose)'),
        ('yaw_sign', '1.0', 'gyro z sign for dr (check: ros2 run showcase_state_estimation imu_check)'),
        ('bringup', 'true', 'include f1tenth_stack bringup_launch.py (VESC, lidar, mux)'),
        ('localize', 'true', 'include particle_filter localize_launch.py'),
        ('ekf', 'true', 'run the llampc-style EKF'),
        ('ekf_source', 'pf', 'pf | mocap'),
        ('publish_static_tf', 'true', 'publish base_link->cg / imu_link for the EKF'),
        ('autostart', 'false', 'gate boots RUNNING instead of STOPPED'),
        ('stop_decel', '4.0', 'soft-stop speed ramp [m/s^2]'),
        ('relay', 'true', 'run fleet_relay (the car drives without it)'),
        ('relay_config', '', 'relay topic list (default: share/showcase_fleet/config/relay.yaml)'),
        ('controller', 'none', "none (start it with lqr_launch.py) | bv3 | <package>/<executable>"),
        ('controller_params', '', 'controller params file'),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args] +
                             [OpaqueFunction(function=_setup)])
