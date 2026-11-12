"""lqr: the LQR terminal, with the start/stop gate keys built in (no separate carctl needed).

    ros2 run showcase_fleet lqr track:=ethzmobil_cw line:=raceline_dr.csv pose:=dr
    ros2 run showcase_fleet lqr                                   # bv3_car.yaml defaults (particle filter)

Arguments are passed straight to `ros2 launch showcase_fleet lqr_launch.py` (track, line, pose, speed_max, avoid ...).
Keys in this terminal:   s = START    space / x = STOP    q = stop the car, then quit
Ctrl-C also stops the car first. The ground station's START / STOP keep working at the same time.
Source `dds_env.sh car <car>` first, and have car_launch.py running (it owns the gate).
"""
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty

import rclpy

from showcase_fleet import common as C
from showcase_fleet.carctl import CarCtl
from showcase_fleet.common import check_car_env

DR_SPEED = 0.7      # keep in sync with lqr_launch.DR_SPEED


def _banner(car, argv):
    """What this terminal is about to run, resolved like lqr_launch.py does."""
    a = dict(x.split(':=', 1) for x in argv if ':=' in x)
    base = {}
    try:
        import yaml
        from ament_index_python.packages import get_package_share_directory
        cfg = a.get('params') or os.path.join(get_package_share_directory('frenet_baseline'), 'config',
                                              'bv3_car.yaml')
        with open(os.path.expanduser(cfg)) as f:
            doc = yaml.safe_load(f) or {}
        base = next(iter(doc.values()))['ros__parameters']
    except Exception:  # noqa: BLE001
        pass
    t, pl = base.get('track', {}), base.get('planner', {})
    dr = a.get('pose') == 'dr'
    track = a.get('track', t.get('NAME', '?'))
    line = a.get('line', t.get('RACELINE', '?'))
    if a.get('speed_max'):
        speed = f"max {a['speed_max']} m/s (speed_max)"
    elif dr:
        speed = f'max {DR_SPEED} m/s (dead-reckoning default; speed_max:= to change)'
    else:
        speed = f"regular LQR speeds: raceline profile, max {pl.get('SPEED_MAX', '?')} m/s (bv3_car.yaml)"
    avoid = a.get('avoid', 'true').strip().lower() in ('1', 'true', 'yes', 'on')
    ctrl = a.get('controller', 'bv3')
    print('\n'.join([
        '',
        f'============================== SHOWCASE LQR ({car}) ==============================',
        f"  CONTROLLER  {'frenet_bv3 LQR' if ctrl == 'bv3' else ctrl}  (output goes through the drive gate)",
        f'  TRACK       {track} / {line}',
        '  POSE        ' + ('DEAD RECKONING /dr/pose/odom (origin = where car_launch mode:=dr started)' if dr else
                            'particle filter /pf/pose/odom' if a.get('pose', 'pf') == 'pf' else a['pose']),
        f'  SPEED       {speed}',
        f"  AVOIDANCE   {'ON: LiDAR obstacles are steered around (avoid:=false to disable)' if avoid else 'OFF'}",
        '  The car stays STOPPED until you press s. It then laps continuously until a stop.',
        '  KEYS        s = START    space / x = STOP    q = stop the car, then quit    (Ctrl-C also stops first)',
        '=' * 80, ''], ), flush=True)


def _stop_launch(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)          # like Ctrl-C in a launch terminal
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main(args=None):
    # NOT rclpy's remove_ros_args: it treats bare name:=value (our launch arguments) as legacy ROS remaps and drops them
    argv = list(sys.argv[1:] if args is None else args[1:])
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    if any(a in ('-h', '--help') for a in argv):
        print(__doc__)
        return 0
    try:
        check_car_env('lqr')
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    # the launch runs in its own session: the keys and Ctrl-C come to us, and we decide the order (car first)
    _banner(os.environ.get('SHOWCASE_CAR', '?'), argv)
    proc = subprocess.Popen(['ros2', 'launch', 'showcase_fleet', 'lqr_launch.py', *argv], start_new_session=True)
    rclpy.init()
    node = CarCtl()
    ready = [False]

    def on_raw(_msg):      # the controller is producing commands: say so once, under the startup logs
        if not ready[0]:
            ready[0] = True
            print(f'\n[lqr] READY: the LQR is running and held by the gate (car: {node.state or "?"}). '
                  f'Press s to START, space to STOP, q to quit.\n', flush=True)

    from ackermann_msgs.msg import AckermannDriveStamped
    node.create_subscription(AckermannDriveStamped, C.DRIVE_RAW_TOPIC, on_raw, C.qos_best_effort(1))

    fd = sys.stdin.fileno()
    tty_ok = os.isatty(fd)
    old = termios.tcgetattr(fd) if tty_ok else None
    code = 0
    try:
        if tty_ok:
            tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if proc.poll() is not None:
                print(f'\n[lqr] LQR launch exited (code {proc.returncode}); stopping the car', flush=True)
                code = proc.returncode or 1
                break
            if tty_ok and select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1).lower()
                if k == 's':
                    node.send('start')
                elif k in (' ', 'x'):
                    node.send('stop')
                elif k == 'q':
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        try:                                            # always leave the car stopped
            node.send('stop')
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.5:
                rclpy.spin_once(node, timeout_sec=0.05)
            print('[lqr] stop sent; shutting down the LQR', flush=True)
        except Exception:  # noqa: BLE001
            pass
        _stop_launch(proc)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
