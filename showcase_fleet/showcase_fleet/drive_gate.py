"""drive_gate: car-local start/stop latch between the controller and the VESC.

    controller (/drive remapped) -> /showcase/drive_raw --+
    controller brake (remapped)  -> /showcase/brake_raw --+--> drive_gate --> /drive, /commands/motor/brake
    carctl (local) / fleet_relay (ground) -> /showcase/cmd --+

Runs on the car's LOCAL (loopback) domain only. Its state is latched: losing the laptop changes nothing.
  STOPPED   boot state (unless autostart). Publishes zero speed / zero steer at `rate_hz`.
  RUNNING   forwards every controller message unchanged, the moment it arrives (no added cadence).
  STOPPING  soft stop: keeps the controller's steering, ramps speed down at `stop_decel`, then -> STOPPED.
Watchdog: once the controller has spoken since the start, `watchdog_s` of silence -> STOPPING ("watchdog");
needs a new start. Before its first message (e.g. LQR still building its gain table) RUNNING just holds zero.
The controller itself is never touched; it keeps running (and keeps its state) while the gate is closed.
Bypass check: if anything else publishes /drive (e.g. the LQR started WITHOUT lqr_launch.py's remaps), the gate
cannot hold the car; it logs an error every second and appends "!BYPASS" to the reported state.
"""
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Float64, String

from showcase_fleet import common as C

RUNNING, STOPPING, STOPPED = 'RUNNING', 'STOPPING', 'STOPPED'


class DriveGate(Node):
    def __init__(self):
        super().__init__('drive_gate')
        self.declare_parameter('autostart', False)
        self.declare_parameter('stop_decel', 4.0)       # m/s^2 speed ramp on stop
        self.declare_parameter('watchdog_s', 0.2)       # controller silence that trips a stop while RUNNING
        self.declare_parameter('rate_hz', 40.0)         # output rate while STOPPING / STOPPED
        self.stop_decel = float(self.get_parameter('stop_decel').value)
        self.watchdog_s = float(self.get_parameter('watchdog_s').value)
        rate = float(self.get_parameter('rate_hz').value)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, C.DRIVE_OUT_TOPIC, 1)
        self.brake_pub = self.create_publisher(Float64, C.BRAKE_OUT_TOPIC, 1)
        self.state_pub = self.create_publisher(String, C.STATE_TOPIC, C.qos_latched())
        self.create_subscription(AckermannDriveStamped, C.DRIVE_RAW_TOPIC, self.on_drive, 1)
        self.create_subscription(Float64, C.BRAKE_RAW_TOPIC, self.on_brake, 1)
        self.create_subscription(String, C.CMD_TOPIC, self.on_cmd, C.qos_reliable(10))

        self.state, self.reason = STOPPED, 'boot'
        self.last_raw = None          # last controller message
        self.last_raw_t = 0.0         # monotonic receive time
        self.v_ramp = 0.0             # speed ceiling while STOPPING
        self.run_start_t = 0.0
        self.last_tick = time.monotonic()
        self.bypass = False
        self.create_timer(1.0 / rate, self.tick)
        self.create_timer(1.0, self.check_bypass)
        if self.get_parameter('autostart').value:
            self.set_state(RUNNING, 'autostart')
        else:
            self.set_state(STOPPED, 'boot')

    # ---- state ------------------------------------------------------------------------------------------------
    def set_state(self, state, reason):
        if state != self.state:          # side effects only on a real transition, never on a re-announce
            if state == RUNNING:
                self.run_start_t = time.monotonic()
            if state == STOPPING:
                # start the ramp from what the car was last told to do
                self.v_ramp = abs(self.last_raw.drive.speed) if self.last_raw is not None else 0.0
        self.state, self.reason = state, reason
        text = (state if state == RUNNING else f'{state}:{reason}') + (' !BYPASS' if self.bypass else '')
        self.state_pub.publish(String(data=text))
        self.get_logger().info(f'-> {text}')

    def check_bypass(self):
        others = self.count_publishers(C.DRIVE_OUT_TOPIC) - 1
        if others > 0:
            self.get_logger().error(f'{others} other publisher(s) on {C.DRIVE_OUT_TOPIC}: the controller is NOT going '
                                    f'through the gate (start it with lqr_launch.py so /drive is remapped)')
        if (others > 0) != self.bypass:
            self.bypass = others > 0
            self.set_state(self.state, self.reason)

    def on_cmd(self, msg):
        verb, source = C.parse_cmd(msg.data)
        if verb is None:
            self.get_logger().warn(f'ignoring command {msg.data!r}')
            return
        if verb == 'start' and self.state != RUNNING:
            self.set_state(RUNNING, source)
        elif verb == 'stop' and self.state == RUNNING:
            self.set_state(STOPPING, source)
        else:   # idempotent: re-announce so a late listener sees the current state
            self.set_state(self.state, self.reason)

    # ---- controller pass-through ------------------------------------------------------------------------------
    def on_drive(self, msg):
        self.last_raw, self.last_raw_t = msg, time.monotonic()
        if self.state == RUNNING:
            self.drive_pub.publish(msg)          # unchanged, same cadence as the controller

    def on_brake(self, msg):
        if self.state == RUNNING:
            self.brake_pub.publish(msg)

    def tick(self):
        now = time.monotonic()
        dt, self.last_tick = now - self.last_tick, now
        fresh = self.last_raw is not None and (now - self.last_raw_t) < self.watchdog_s
        if self.state == RUNNING:
            if fresh:
                return
            if self.last_raw is not None and self.last_raw_t > self.run_start_t:
                self.set_state(STOPPING, 'watchdog')
            # else: controller has not published since the start yet -> hold zero below
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        if self.state == STOPPING:
            self.v_ramp = max(0.0, self.v_ramp - self.stop_decel * dt)
            if fresh:                            # keep following the line while slowing down
                out.drive.steering_angle = self.last_raw.drive.steering_angle
            out.drive.speed = self.v_ramp
            if self.v_ramp <= 0.0:
                self.set_state(STOPPED, self.reason)
        self.drive_pub.publish(out)              # STOPPED / waiting: zero speed, zero steer


def main(args=None):
    rclpy.init(args=args)
    node = DriveGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:   # leave the car commanded to zero on shutdown
            node.drive_pub.publish(AckermannDriveStamped())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
