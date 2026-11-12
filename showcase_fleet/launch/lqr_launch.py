"""The car's controller, in its own terminal. From a shell that ran `source dds_env.sh car <car1|car2>`:

    ros2 launch showcase_fleet lqr_launch.py                                  # frenet_bv3 LQR, bv3_car.yaml
    ros2 launch showcase_fleet lqr_launch.py params:=/path/to/my_bv3.yaml
    ros2 launch showcase_fleet lqr_launch.py controller:=<package>/<executable> params:=...   # e.g. obstacle avoidance

The controller runs UNMODIFIED; only its outputs are remapped into the drive gate:
    /drive                 -> /showcase/drive_raw
    /commands/motor/brake  -> /showcase/brake_raw
So it can start (and warm up) at any time: nothing reaches the motor until the gate gets a start from carctl or
the ground station, and a stop ramps the car down even though the controller keeps running.
Do NOT start frenet_bv3.py by hand without these remaps: it would bypass the gate (the gate reports "!BYPASS").
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from showcase_fleet.common import check_car_env

DR_SPEED = '0.7'      # [m/s] default top speed with pose:=dr (first laps of an unmapped track)
GATE_REMAPS = [('/drive', '/showcase/drive_raw'), ('/commands/motor/brake', '/showcase/brake_raw')]


def _share(pkg):
    try:
        return get_package_share_directory(pkg)
    except PackageNotFoundError:
        return None


def _find_bv3(explicit):
    cands = [explicit] if explicit else []
    share = _share('frenet_baseline')
    if share:   # --symlink-install: share/config links back into the source tree, next to scripts/
        cands.append(os.path.join(os.path.dirname(os.path.realpath(os.path.join(share, 'config', 'bv3_car.yaml'))),
                                  '..', 'scripts', 'frenet_bv3.py'))
    here = os.path.realpath(__file__)   # source tree: <repo>/showcase_fleet/launch -> <repo>/frenet_baseline
    cands.append(os.path.join(os.path.dirname(here), '..', '..', 'frenet_baseline', 'scripts', 'frenet_bv3.py'))
    cands.append(os.path.expanduser('~/f1tenth_showcase/src/frenet_baseline/scripts/frenet_bv3.py'))
    for c in cands:
        if c and os.path.isfile(os.path.expanduser(c)):
            return os.path.normpath(os.path.expanduser(c))
    raise RuntimeError(f'lqr_launch: cannot find frenet_bv3.py; pass script:=/path/to/frenet_bv3.py (tried {cands})')


def _overrides(params_file, get):
    """Parameter overrides for bv3, written as a node-scoped params file passed AFTER bv3_car.yaml (a plain -p loses
    to the file's node-scoped values). Parameters only; the LQR code is untouched.
      track / line   -> track.NAME / track.RACELINE (map check skipped for any track but the venue)
      pose:=dr       -> pose_topic /dr/pose/odom (dead-reckoned laser point; the LQR's REF_X stays 0.26)
      speed_max / speed_scale -> planner.SPEED_MAX / planner.SPEED_SCALE
      track.ROOT/MAP from bv3_car.yaml missing on this machine -> the installed frenet_baseline share (same files)"""
    import yaml
    share = _share('frenet_baseline')
    try:
        with open(params_file) as f:
            doc = yaml.safe_load(f) or {}
        node_name = next(iter(doc))
        base = doc[node_name]['ros__parameters']
    except Exception:  # noqa: BLE001
        node_name, base = 'bv3', {}
    t0 = base.get('track', {})
    track, planner, notes, extra = {}, {}, [], {}
    root = os.path.expanduser(str(t0.get('ROOT', '')))
    if share and root and not os.path.isdir(root) and os.path.isdir(os.path.join(share, 'waypoints')):
        track['ROOT'] = os.path.join(share, 'waypoints')
        notes.append(f"track.ROOT {t0.get('ROOT')} missing -> {track['ROOT']}")
    mp = os.path.expanduser(str(t0.get('MAP', '')))
    alt = os.path.join(share, 'maps', os.path.basename(mp)) if (share and mp) else ''
    if mp and os.path.isabs(mp) and not os.path.isfile(mp) and os.path.isfile(alt):
        track['MAP'] = alt
        notes.append(f"track.MAP {t0.get('MAP')} missing -> {alt}")
    if get('track'):
        track['NAME'] = get('track')
        if not get('track').startswith('venue'):
            track['MAP'] = ''                    # the venue map says nothing about another track
    if get('line'):
        track['RACELINE'] = get('line')
    if get('pose') == 'dr':
        extra['pose_topic'] = '/dr/pose/odom'
    elif get('pose') not in ('', 'pf'):
        extra['pose_topic'] = get('pose')
    speed_max = get('speed_max') or (DR_SPEED if get('pose') == 'dr' else '')   # dead reckoning: slow by default
    if speed_max:
        planner['SPEED_MAX'] = float(speed_max)
        base_min = float(base.get('planner', {}).get('SPEED_MIN', 1.0))
        if planner['SPEED_MAX'] < base_min:           # bv3: v = max(SPEED_MIN, min(v_ref, SPEED_MAX))
            planner['SPEED_MIN'] = planner['SPEED_MAX']
    if get('speed_scale'):
        planner['SPEED_SCALE'] = float(get('speed_scale'))
    if get('avoid'):                             # obstacle avoidance (BaS) inside bv3: avoid.ENABLE (default ON)
        extra['avoid'] = {'ENABLE': get('avoid').strip().lower() in ('1', 'true', 'yes', 'on')}
    over = dict(extra)
    if track:
        over['track'] = track
    if planner:
        over['planner'] = planner
    if not over:
        return None, notes
    path = os.path.join(os.path.expanduser('~/.ros'), f'showcase_{node_name}_overrides.yaml')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump({node_name: {'ros__parameters': over}}, f)
    notes.append(f'overrides {over} -> {path}')
    return path, notes


def _setup(context):
    get = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    check_car_env('lqr_launch')
    ctrl, params = get('controller'), get('params')
    if ctrl == 'bv3':
        script = _find_bv3(get('script'))
        params = params or os.path.join(_share('frenet_baseline') or '', 'config', 'bv3_car.yaml')
        cmd = ['python3', script, '--ros-args', '--params-file', params]
        over, notes = _overrides(params, get)
        if over:
            cmd += ['--params-file', over]
        for a, b in GATE_REMAPS:
            cmd += ['-r', f'{a}:={b}']
        return [LogInfo(msg=f'[showcase] {n}') for n in notes] + \
            [LogInfo(msg=f'[showcase] LQR: {script} ({params}); output gated, waiting for start'),
             ExecuteProcess(cmd=cmd, output='screen', name='bv3')]
    pkg, _, exe = ctrl.partition('/')
    if not exe:
        raise RuntimeError("lqr_launch: controller must be 'bv3' or '<package>/<executable>'")
    return [LogInfo(msg=f'[showcase] controller {ctrl}; output gated, waiting for start'),
            Node(package=pkg, executable=exe, output='screen', parameters=[params] if params else [],
                 remappings=GATE_REMAPS)]


def generate_launch_description():
    args = [
        ('controller', 'bv3', "bv3 | <package>/<executable>"),
        ('params', '', 'params file (bv3 default: share/frenet_baseline/config/bv3_car.yaml)'),
        ('script', '', 'path to frenet_bv3.py (auto-detected)'),
        ('track', '', 'waypoints folder, e.g. ethzmobil_cw | ethzmobil_ccw | venue_ccw (default: bv3_car.yaml)'),
        ('line', '', 'raceline file in it, e.g. raceline_dr.csv | raceline_stable.csv'),
        ('pose', 'pf', 'pf (/pf/pose/odom) | dr (/dr/pose/odom, dead reckoning) | <topic>'),
        ('speed_max', '', f'planner.SPEED_MAX override [m/s] (pose:=dr default {DR_SPEED})'),
        ('speed_scale', '', 'planner.SPEED_SCALE override'),
        ('avoid', 'true', 'true | false: bv3 obstacle avoidance (avoid.ENABLE)'),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args] +
                             [OpaqueFunction(function=_setup)])
