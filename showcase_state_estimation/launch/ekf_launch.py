"""EKF state estimation for the showcase cars (ported from llampc's ekf_launch.py / bringup_launch.py).

    ros2 launch showcase_state_estimation ekf_launch.py                  # particle filter input (default)
    ros2 launch showcase_state_estimation ekf_launch.py source:=dr       # dead reckoning (no map yet)
    ros2 launch showcase_state_estimation ekf_launch.py source:=mocap mocap_topic:=/car1/pose

source:=pf     /pf/pose/odom + /sensors/imu/data + /zupt -> /showcase/odometry (cg)          [llampc's ekf.yaml]
source:=dr     /odom wheel speed + gyro yaw rate (+ /zupt), integrated from (0, 0, 0) at start, in the odom frame:
               EKF (rear axle) -> /odometry/dr_base -> odom_points -> /dr/pose/odom (laser point, for the LQR)
                                                                   -> /showcase/odometry (cg, relay / LLA)
               The gyro bias is measured during the first bias_calib_s seconds at standstill: keep the car still.
source:=mocap  optitrack_node (/<rigid body>/pose -> /optitrack/odom) instead of the PF          [llampc's mocap.yaml]

The EKF works in `cg` / `base_link` and needs base_link->cg and base_link->imu_link. The LLA-era f1tenth_stack
bringup published those; the stock one does not, so publish_static_tf:=true (default) publishes them here.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CONFIGS = {'pf': 'ekf.yaml', 'mocap': 'ekf_mocap.yaml', 'dr': 'ekf_dr.yaml'}


def _setup(context):
    get = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    on = lambda k: get(k).strip().lower() in ('1', 'true', 'yes', 'on')  # noqa: E731
    source = get('source')
    if source not in CONFIGS:
        raise RuntimeError(f"ekf_launch: source must be one of {list(CONFIGS)} (got '{source}')")
    share = FindPackageShare('showcase_state_estimation').perform(context)
    dr = source == 'dr'
    acts = []
    if on('publish_static_tf'):
        acts += [
            Node(package='tf2_ros', executable='static_transform_publisher', name='tf_base_link_cg',
                 arguments=[get('cg_x'), '0', '0', '0', '0', '0', 'base_link', 'cg']),
            Node(package='tf2_ros', executable='static_transform_publisher', name='tf_base_link_imu',
                 arguments=[get('imu_x'), '0', get('imu_z'), '0', '0', '0', 'base_link', 'imu_link']),
        ]
    if on('zupt') or dr:
        acts.append(Node(package='showcase_state_estimation', executable='imu_zupt_prep', name='imu_zupt_prep',
                         output='screen',
                         parameters=[{'yaw_sign': float(get('yaw_sign')), 'gyro_scale': float(get('gyro_scale')),
                                      'bias_calib_s': float(get('bias_calib_s')) if dr else 0.0}]))
    if source == 'mocap':
        acts.append(Node(package='showcase_state_estimation', executable='optitrack_node', name='optitrack_node',
                         output='screen', parameters=[{'mocap_topic': get('mocap_topic')}]))
    # own node name and output topic: a car's own bringup may already run robot_localization (team10: imu_relay +
    # ekf_filter_node -> /odometry/filtered + odom->base_link TF); both must coexist untouched
    remaps = [('/set_pose', '/initialpose'),
              ('/odometry/filtered', '/odometry/dr_base' if dr else '/showcase/odometry')]
    if dr:
        acts.append(Node(package='showcase_state_estimation', executable='odom_points', name='odom_points',
                         output='screen',
                         parameters=[{'input': '/odometry/dr_base',
                                      'outputs': ['/dr/pose/odom', '/showcase/odometry'],
                                      'offsets': [float(get('laser_x')), float(get('cg_x'))],
                                      'frames': ['laser', 'cg']}]))
    acts.append(Node(package='robot_localization', executable='ekf_node', name='showcase_ekf', output='screen',
                     parameters=[f'{share}/config/{CONFIGS[source]}'], remappings=remaps))
    return acts


def generate_launch_description():
    args = [
        ('source', 'pf', 'pf (particle filter) | dr (dead reckoning) | mocap (OptiTrack)'),
        ('zupt', 'true', 'run imu_zupt_prep (always on for dr)'),
        ('publish_static_tf', 'true', 'publish base_link->cg and base_link->imu_link'),
        ('cg_x', '0.15', 'base_link->cg x [m]'),
        ('imu_x', '0.065', 'base_link->imu_link x [m]'),
        ('imu_z', '0.09', 'base_link->imu_link z [m]'),
        ('laser_x', '0.26', 'dr: rear axle -> tracked point (the LQR REF_X / laser) [m]'),
        ('yaw_sign', '1.0', 'gyro z sign (check with: ros2 run showcase_state_estimation imu_check)'),
        ('gyro_scale', '0.017453292519943295', 'raw gyro -> rad/s (VESC: deg/s)'),
        ('bias_calib_s', '3.0', 'dr: gyro bias averaging time at standstill [s]'),
        ('mocap_topic', '/f1tenth/pose', 'rigid-body PoseStamped (source:=mocap)'),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args] +
                             [OpaqueFunction(function=_setup)])
