"""fleet_relay: the car's ONLY bridge between its loopback driving stack and the Wi-Fi.

    LOCAL domain (loopback)                         NET domain (Wi-Fi)
    /showcase/odometry, /sensors/core, /drive ...  ->  /<car>/odom, /<car>/vesc, /<car>/drive ...  (best effort)
    /showcase/state                                ->  /<car>/state  (re-sent at heartbeat_hz)
    /showcase/cmd   <-  "start:ground"/"stop:ground" <- /<car>/cmd   (only exact start/stop verbs pass)

Two child processes, one per DDS domain (works on Foxy, which has no per-context domain id), joined by bounded
queues that drop when full. The local side never waits on the network side: a dead, slow or absent Wi-Fi link can
only lose relayed samples, never delay the car. If the network side dies (e.g. Wi-Fi interface missing at boot)
it is restarted every few seconds; the local side keeps running.

Which topics go up is data, not code: config/relay.yaml (rows of local topic, message type, network name).
A row with `pack: vesc` is flattened on the car into std_msgs/Float64MultiArray (see common.VESC_FIELDS), so the
laptop never needs vesc_msgs and a Foxy car and a Humble car with different VescState definitions look the same.

    ros2 run showcase_fleet fleet_relay --car car1 [--config relay.yaml]
"""
import argparse
import multiprocessing as mp
import os
import queue
import signal
import sys
import time

import yaml


# ======================================================================================================================
# children
# ======================================================================================================================
def _child_init(domain_id, tag):
    os.environ['ROS_DOMAIN_ID'] = str(domain_id)   # read by rcl when the context / node is created
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent owns shutdown
    import rclpy
    rclpy.init()
    return rclpy


def _alive(stop_evt, ppid):
    """Keep running only while not told to stop AND the relay parent still exists (a SIGKILLed parent must not
    leave orphaned DDS participants behind)."""
    return not stop_evt.is_set() and os.getppid() == ppid


def _local_side(car, cfg, domain_id, up_q, down_q, stop_evt):
    ppid = os.getppid()
    rclpy = _child_init(domain_id, 'local')
    from rclpy.node import Node
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String

    from showcase_fleet import common as C

    node = Node(f'{car}_relay_local')
    log = node.get_logger()

    def make_cb(name, min_dt, pack=None):
        last = [0.0]

        def cb(msg):
            now = time.monotonic()
            if min_dt and now - last[0] < min_dt:
                return
            last[0] = now
            if pack is not None:
                msg = pack(msg)
            try:
                up_q.put_nowait((name, serialize_message(msg)))
            except queue.Full:
                pass                      # network side is behind: drop, never block the car
        return cb

    for row in cfg.get('up', []):
        try:
            cls = get_message(row['type'])
        except Exception as e:  # noqa: BLE001  (missing message package on this car)
            log.warn(f"skipping {row['local']}: cannot import {row['type']} ({e})")
            continue
        hz = float(row.get('max_hz', 0) or 0)
        pack = C.PACKERS[row['pack']] if row.get('pack') else None
        node.create_subscription(cls, row['local'], make_cb(row['name'], 1.0 / hz if hz > 0 else 0.0, pack),
                                 C.qos_best_effort(5))
        log.info(f"up: {row['local']} -> /{car}/{row['name']}" + (f' (<= {hz:g} Hz)' if hz > 0 else ''))

    # gate state is latched locally; forward it under the reserved name 'state'
    node.create_subscription(String, C.STATE_TOPIC, make_cb(C.NET_STATE, 0.0), C.qos_latched())

    cmd_pub = node.create_publisher(String, C.CMD_TOPIC, C.qos_reliable(10))

    def pump_down():
        while True:
            try:
                text = down_q.get_nowait()
            except queue.Empty:
                return
            cmd_pub.publish(String(data=text))
            log.info(f'ground command -> {C.CMD_TOPIC}: {text}')

    node.create_timer(0.02, pump_down)
    while rclpy.ok() and _alive(stop_evt, ppid):
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


def _net_side(car, cfg, domain_id, up_q, down_q, stop_evt):
    ppid = os.getppid()
    rclpy = _child_init(domain_id, 'net')
    import threading

    from rclpy.node import Node
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String

    from showcase_fleet import common as C

    node = Node(f'{car}_relay_net')
    log = node.get_logger()
    pubs = {}
    for row in cfg.get('up', []):
        try:
            cls = get_message('std_msgs/msg/Float64MultiArray' if row.get('pack') else row['type'])
        except Exception:  # noqa: BLE001
            continue
        pubs[row['name']] = (cls, node.create_publisher(cls, C.net_topic(car, row['name']), C.qos_best_effort(1)))
    state_pub = node.create_publisher(String, C.net_topic(car, C.NET_STATE), C.qos_best_effort(1))
    last_state = ['UNKNOWN:no gate']

    def on_cmd(msg):
        verb, _ = C.parse_cmd(msg.data)
        if verb is None:
            log.warn(f'ignoring network command {msg.data!r}')
            return
        try:
            down_q.put_nowait(f'{verb}:ground')
        except queue.Full:
            log.warn('command queue full, dropping')

    # reliable + volatile: a car that reconnects never receives an old command
    node.create_subscription(String, C.net_topic(car, C.NET_CMD), on_cmd, C.qos_reliable(10))

    def heartbeat():
        state_pub.publish(String(data=last_state[0]))

    node.create_timer(1.0 / float(cfg.get('heartbeat_hz', 5.0)), heartbeat)
    spin = threading.Thread(target=lambda: _spin_until(rclpy, node, stop_evt, ppid), daemon=True)
    spin.start()
    log.info(f'network side up on domain {domain_id} as /{car}/*')

    while rclpy.ok() and _alive(stop_evt, ppid):
        try:
            name, raw = up_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if name == C.NET_STATE:
            last_state[0] = deserialize_message(raw, String).data
            heartbeat()
            continue
        entry = pubs.get(name)
        if entry is not None:
            entry[1].publish(deserialize_message(raw, entry[0]))
    spin.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()


def _spin_until(rclpy, node, stop_evt, ppid):
    while rclpy.ok() and _alive(stop_evt, ppid):
        rclpy.spin_once(node, timeout_sec=0.1)


# ======================================================================================================================
# parent
# ======================================================================================================================
def _default_config():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('showcase_fleet'), 'config', 'relay.yaml')
    except Exception:  # noqa: BLE001
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'relay.yaml')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--ros-args' in argv:                       # launch_ros appends these; we take plain CLI args
        argv = argv[:argv.index('--ros-args')]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--car', default=os.environ.get('SHOWCASE_CAR', ''))
    ap.add_argument('--config', default='')
    ap.add_argument('--local-domain', type=int, default=int(os.environ.get('SHOWCASE_LOCAL_DOMAIN', -1)))
    ap.add_argument('--net-domain', type=int, default=int(os.environ.get('SHOWCASE_NET_DOMAIN', -1)))
    a = ap.parse_args(argv)
    if not a.car or a.local_domain < 0 or a.net_domain < 0:
        sys.exit('fleet_relay: need --car/--local-domain/--net-domain (or `source dds_env.sh car <car>` first)')
    if a.local_domain == a.net_domain:
        sys.exit('fleet_relay: local and net domains must differ, or the car stack would be on Wi-Fi')
    with open(a.config or _default_config()) as f:
        cfg = yaml.safe_load(f) or {}

    ctx = mp.get_context('spawn')
    qsize = int(cfg.get('queue_size', 256))
    up_q, down_q, stop_evt = ctx.Queue(qsize), ctx.Queue(16), ctx.Event()

    def start(fn, dom):
        p = ctx.Process(target=fn, args=(a.car, cfg, dom, up_q, down_q, stop_evt), daemon=True)
        p.start()
        return p

    def on_signal(*_):
        stop_evt.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    launcher = os.getppid()
    print(f'[fleet_relay] {a.car}: local domain {a.local_domain} <-> net domain {a.net_domain}', flush=True)
    local = start(_local_side, a.local_domain)
    net = start(_net_side, a.net_domain)
    while not stop_evt.is_set():
        time.sleep(0.5)
        if os.getppid() != launcher:
            print('[fleet_relay] launcher is gone; exiting', flush=True)
            break
        if not local.is_alive():
            print('[fleet_relay] local side exited; shutting down (launch respawns the relay)', flush=True)
            break
        if not net.is_alive():
            print(f'[fleet_relay] network side exited (code {net.exitcode}); retrying in 3 s', flush=True)
            for _ in range(6):
                if stop_evt.is_set():
                    break
                time.sleep(0.5)
            if not stop_evt.is_set():
                net = start(_net_side, a.net_domain)
    stop_evt.set()
    for p in (local, net):
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()
    return 0 if local.exitcode in (0, None) else 1


if __name__ == '__main__':
    sys.exit(main())
