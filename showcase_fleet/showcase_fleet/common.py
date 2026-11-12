"""Names, command protocol and helpers shared by the car-side and ground-side nodes.

Two DDS domains (set up by scripts/dds_env.sh):
  LOCAL  - loopback only, on each car. The whole driving stack (VESC, PF, EKF, controller, drive_gate) lives here,
           so nothing on Wi-Fi can reach it, block it, or stall it.
  NET    - Wi-Fi. Only fleet_relay (car side) and the ground station / LLA estimators (laptop) join it.
"""
import os

import numpy as np

# ---- car-local topics (LOCAL domain) ---------------------------------------------------------------------------
CMD_TOPIC = '/showcase/cmd'                # std_msgs/String "start[:source]" | "stop[:source]"  -> drive_gate
STATE_TOPIC = '/showcase/state'            # std_msgs/String "RUNNING" | "STOPPING:<why>" | "STOPPED:<why>"
DRIVE_RAW_TOPIC = '/showcase/drive_raw'    # controller output (remapped from /drive)
BRAKE_RAW_TOPIC = '/showcase/brake_raw'    # controller brake (remapped from /commands/motor/brake)
DRIVE_OUT_TOPIC = '/drive'                 # gated -> ackermann_mux -> VESC
BRAKE_OUT_TOPIC = '/commands/motor/brake'

# ---- network topics (NET domain): /<car>/<name> -----------------------------------------------------------------
NET_CMD = 'cmd'          # laptop -> car
NET_STATE = 'state'      # car -> laptop, re-sent at 5 Hz as a heartbeat
NET_ODOM = 'odom'
NET_VESC = 'vesc'
NET_DRIVE = 'drive'
NET_LLA = 'lla/params'   # laptop estimator -> GUI: [Cf, Cr, muf, mur, mu, model_idx, active]

VERBS = ('start', 'stop')

# /<car>/vesc layout (Float64MultiArray), packed on the car from vesc_msgs/VescStateStamped
VESC_FIELDS = ('t', 'avg_iq', 'current_motor', 'erpm', 'duty_cycle')
VESC_I = {k: i for i, k in enumerate(VESC_FIELDS)}


def pack_vesc(msg):
    """VescStateStamped -> Float64MultiArray. avg_iq is what LLA uses as its current input; older vesc_msgs
    without it fall back to current_motor. erpm is VescState.speed."""
    from std_msgs.msg import Float64MultiArray
    st = msg.state
    cur = float(getattr(st, 'current_motor', 0.0))
    iq = float(getattr(st, 'avg_iq', cur))
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return Float64MultiArray(data=[t, iq, cur, float(st.speed), float(getattr(st, 'duty_cycle', 0.0))])


PACKERS = {'vesc': pack_vesc}


def net_topic(car, name):
    return f'/{car}/{name}'


def parse_cmd(text):
    """'start:ground' -> ('start', 'ground'). Returns (None, None) for anything that is not a known verb."""
    verb, _, source = (text or '').strip().partition(':')
    verb = verb.strip().lower()
    if verb not in VERBS:
        return None, None
    return verb, (source.strip() or 'unknown')


def car_names_from_env():
    return [c for c in os.environ.get('SHOWCASE_CARS', 'car1,car2').split(',') if c]


def check_car_env(what):
    """Refuse to start car-side processes from a shell that is not on this car's loopback domain."""
    env = os.environ
    if env.get('SHOWCASE_ROLE') != 'car' or not env.get('SHOWCASE_CAR'):
        raise RuntimeError(f'{what}: run `source <showcase_fleet>/share/showcase_fleet/scripts/dds_env.sh car '
                           f'<car1|car2>` first (the driving stack must be on the car\'s loopback domain)')
    if env.get('ROS_DOMAIN_ID') != env.get('SHOWCASE_LOCAL_DOMAIN') or \
            env.get('SHOWCASE_LOCAL_DOMAIN') == env.get('SHOWCASE_NET_DOMAIN'):
        raise RuntimeError(f'{what}: ROS_DOMAIN_ID must equal SHOWCASE_LOCAL_DOMAIN and differ from the NET '
                           f'domain; re-source dds_env.sh car <car>')
    return env['SHOWCASE_CAR']


# ---- QoS (names that exist on both Foxy and Humble) -------------------------------------------------------------
def qos_best_effort(depth=1):
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    return QoSProfile(depth=depth, history=QoSHistoryPolicy.KEEP_LAST,
                      reliability=QoSReliabilityPolicy.BEST_EFFORT)


def qos_reliable(depth=10):
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    return QoSProfile(depth=depth, history=QoSHistoryPolicy.KEEP_LAST,
                      reliability=QoSReliabilityPolicy.RELIABLE)


def qos_latched():
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    return QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                      reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def yaw_from_quat(q):
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


# ---- racelines (frenet_baseline CSV format) ---------------------------------------------------------------------
def resolve_raceline(track='venue_ccw', line='raceline_stable.csv', raceline_file=''):
    """Explicit file wins; else share/frenet_baseline/waypoints/<track>/<line> (same files the LQR uses)."""
    if raceline_file:
        return os.path.expanduser(raceline_file)
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('frenet_baseline'), 'waypoints', track, line)


def load_raceline(path):
    """Load a frenet_baseline raceline/centerline CSV ('#' header, comma separated).

    9 columns: x_m, y_m, s_m, psi_rad, kappa_radpm, vx_mps, ax_mps2, w_left_m, w_right_m
    4 columns: x_m, y_m, w_left_m, w_right_m
    Returns dict(x, y, v or None, closed).
    """
    data = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    x, y = data[:, 0], data[:, 1]
    v = data[:, 5] if data.shape[1] >= 9 else None
    step = np.median(np.hypot(np.diff(x), np.diff(y))) if len(x) > 1 else 0.0
    closed = len(x) > 2 and np.hypot(x[-1] - x[0], y[-1] - y[0]) < 3.0 * max(step, 1e-3)
    return {'x': x, 'y': y, 'v': v, 'closed': bool(closed)}


def offset_curve(x, y, d, closed):
    """Offset a polyline by d metres along its left normal."""
    if closed:
        dx = np.roll(x, -1) - np.roll(x, 1)
        dy = np.roll(y, -1) - np.roll(y, 1)
    else:
        dx, dy = np.gradient(x), np.gradient(y)
    n = np.hypot(dx, dy)
    n[n < 1e-9] = 1e-9
    return x - d * dy / n, y + d * dx / n
