"""carctl: start / stop THIS car from a terminal on the car (LOCAL domain; no laptop involved).

    ros2 run showcase_fleet carctl start
    ros2 run showcase_fleet carctl stop
    ros2 run showcase_fleet carctl status
    ros2 run showcase_fleet carctl            # interactive: s = start, space / x = stop, q = quit

Source `dds_env.sh car <car>` first so this joins the car's loopback domain.
"""
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from showcase_fleet import common as C


class CarCtl(Node):
    def __init__(self):
        super().__init__('carctl')
        self.pub = self.create_publisher(String, C.CMD_TOPIC, C.qos_reliable(10))
        self.state = None
        self.create_subscription(String, C.STATE_TOPIC, self._on_state, C.qos_latched())

    def _on_state(self, msg):
        if msg.data != self.state:
            self.state = msg.data
            print(f'\r[car] {msg.data}\r', flush=True)

    def wait_gate(self, timeout=3.0):
        t0 = time.monotonic()
        while self.pub.get_subscription_count() == 0 and time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.pub.get_subscription_count() > 0

    def send(self, verb):
        self.pub.publish(String(data=f'{verb}:local'))

    def spin_for(self, seconds):
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)


def _interactive(node):
    print('s = start   space/x = stop   q = quit', flush=True)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1).lower()
                if k == 's':
                    node.send('start')
                elif k in (' ', 'x'):
                    node.send('stop')
                elif k in ('q', '\x03'):
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(args=None):
    argv = rclpy.utilities.remove_ros_args(sys.argv if args is None else args)[1:]
    verb = argv[0].lower() if argv else ''
    rclpy.init(args=args)
    node = CarCtl()
    code = 0
    try:
        if not node.wait_gate():
            print('carctl: no drive_gate on this domain (is car_launch running, and did you source '
                  'dds_env.sh car <car>?)', file=sys.stderr)
            code = 1
        elif verb in C.VERBS:
            node.send(verb)
            node.spin_for(0.5)
        elif verb == 'status':
            node.spin_for(0.5)
            print(node.state or 'unknown')
        elif verb == '':
            node.spin_for(0.2)
            _interactive(node)
        else:
            print(__doc__)
            code = 2
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
