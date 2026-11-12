"""lla_estimator (ground station): LLA tyre-parameter estimation for one car, from what the car actually did.

Builds LLA's model bank exactly like llampc's multi_llampc_fiala.MPCNode.fiala_setup()/initialize_mpc()
(same Cf/Cr/muf/mur grid, window, gamma, reset interval, cost weights, 0.04 s RK4 step) and runs the same
bank cycle as its control_callback(), but with NO MPC: the bank is driven by the car's APPLIED control,
    u = [VESC avg_iq (current, A), steering_angle of the gated /drive (rad)],
instead of LLA's own u_opt (the cars drive on LQR, so LLA's commands were never applied).
llampc itself is imported, not modified.

Runs in a car namespace on the laptop (NET domain):
    sub  odom   (nav_msgs/Odometry, EKF)        sub  vesc  (Float64MultiArray, common.VESC_FIELDS)
    sub  drive  (AckermannDriveStamped, applied) pub  lla/params  [Cf, Cr, muf, mur, mu, model_idx, status, stale]
    status: 1 = estimating, 0 = holding (below v_min), -1 = holding (input missing/stale; `stale` is a bitmask
    1 = odom, 2 = vesc, 4 = drive)
Bank updates pause (estimates hold) when vx < v_min or any input is stale (car dropped off); history is reset
before resuming, so a car that rejoins starts clean.
"""
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from showcase_fleet import common as C

# ---- LLA bank configuration: copied from multi_llampc_fiala.py (declare_params / fiala_setup) -------------------
MEAN = {'Cf': 250, 'Cr': 225, 'muf': 0.6, 'mur': 0.6, 'Cro': 0.0}
VARIATION = {'Cf': 75, 'Cr': 75, 'muf': 0.5, 'mur': 0.5, 'Cro': 0.0}
DISCRETIZATION = {'Cf': 5, 'Cr': 5, 'muf': 7, 'mur': 7, 'Cro': 0.0}
COST_WEIGHTS = np.array([0.0, 0.0, 20.0, 1.0, 10.0, 0.1])     # x, y, theta, vx, vy, omega
LLA_WINDOW = 40
LLA_RESET_INTERVAL = 4
LLA_SKIP_INTERVAL = 1
LLA_GAMMA = 0.9
LLA_DT = 0.04          # lla_predict_horizon == control_callback_speed in LLA
MU_EMA = 0.1


class LLAEstimator(Node):
    def __init__(self):
        super().__init__('lla_estimator')
        self.declare_parameter('v_min', 0.3)        # LLA itself switches to pure pursuit (no bank) below 0.3 m/s
        self.declare_parameter('stale_s', 0.25)
        self.declare_parameter('log', False)        # write an LLALogger .npz (replay with viz_ref_fiala)
        self.declare_parameter('log_name', '')
        self.v_min = float(self.get_parameter('v_min').value)
        self.stale_s = float(self.get_parameter('stale_s').value)

        self._setup_bank()

        self.state = None            # [x, y, phi, vx, vy, omega]
        self.t_odom = self.t_vesc = self.t_drive = -1e9
        self.iq = 0.0
        self.omega_w = 0.0
        self.steer = 0.0
        self.need_reset = True
        self.reset_counter = 0
        self.last_mu = None
        self.best = self.dynamics_bank.get_model_params_arr(0)
        self.best_idx = 0

        self.logger_ = None
        if self.get_parameter('log').value:
            from llampc.lla_run_utils import LLALogger
            name = self.get_parameter('log_name').value or \
                f"showcase_{self.get_namespace().strip('/') or 'car'}_lla"
            self.logger_ = LLALogger(name)
            self.get_logger().info(f'logging to {self.logger_.log_file}')

        be = C.qos_best_effort(5)
        self.create_subscription(Odometry, C.NET_ODOM, self.on_odom, be)
        self.create_subscription(Float64MultiArray, C.NET_VESC, self.on_vesc, be)
        self.create_subscription(AckermannDriveStamped, C.NET_DRIVE, self.on_drive, be)
        self.pub = self.create_publisher(Float64MultiArray, C.NET_LLA, C.qos_reliable(10))
        self.create_timer(LLA_DT, self.step)
        self.get_logger().info(f'LLA bank ready ({self.num_models} models); waiting for '
                               f'{self.get_namespace()}/{{odom,vesc,drive}}')

    # ---- bank (identical construction to multi_llampc_fiala) -------------------------------------------------
    def _setup_bank(self):
        import llampc.rollout.dynamic_fiala as dynamics_fiala
        import llampc.rollout.multi_step_history as history
        from llampc.params import F110, get_param_dict_grid
        from llampc.rollout.rk6 import rk4Factory

        self.params_car = F110()
        param_dict = get_param_dict_grid(MEAN, VARIATION, discretization=DISCRETIZATION, ground_truth=True)
        self.num_models = len(param_dict['Cf'])
        p = self.params_car
        self.dynamics_bank = dynamics_fiala.DBMFialaBank(
            p['lf'], p['lr'], p['mass'], p['Iz'], p['rw'],
            param_dict['Cf'], param_dict['Cr'], param_dict['muf'], param_dict['mur'], param_dict['Cro'],
            self.num_models)
        self.lb_history = history.LBHistoryMultiStep(
            self.num_models, LLA_WINDOW, LLA_DT, COST_WEIGHTS, 6, rk4Factory,
            self.dynamics_bank, dynamics_fiala.diffequation,
            buffer_size=[0, 0], reset_interval=LLA_RESET_INTERVAL,
            skip_interval=LLA_SKIP_INTERVAL, gamma=LLA_GAMMA)
        # warm start / JIT compile, as LLA's initialize_mpc()
        self.get_logger().info('compiling LLA bank (JAX)...')
        self.lb_history.predict_states(np.zeros(6), np.zeros(2))
        self.lb_history.update_lookback_error(np.zeros(6))
        self.lb_history.get_best_model()
        self.lb_history.reset()

    # ---- inputs (relayed from the car) -----------------------------------------------------------------------
    def on_odom(self, msg):
        prev_phi = self.state[2] if self.state is not None else 0.0
        phi = np.unwrap([prev_phi, C.yaw_from_quat(msg.pose.pose.orientation)])[1]
        tw = msg.twist.twist
        self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, phi,
                               tw.linear.x, tw.linear.y, tw.angular.z])
        self.t_odom = time.monotonic()

    def on_vesc(self, msg):
        d = msg.data
        if len(d) < len(C.VESC_FIELDS):
            return
        self.iq = float(d[C.VESC_I['avg_iq']])
        # same conversion as LLA's sensor_callback: erpm -> wheel angular speed
        motor_omega = d[C.VESC_I['erpm']] / self.params_car['pole_pairs'] * (2.0 * np.pi / 60.0)
        self.omega_w = motor_omega / self.params_car['gear_ratio']
        self.t_vesc = time.monotonic()

    def on_drive(self, msg):
        self.steer = float(msg.drive.steering_angle)
        self.t_drive = time.monotonic()

    # ---- one LLA bank cycle --------------------------------------------------------------------------------
    def step(self):
        now = time.monotonic()
        stale = sum(bit for bit, t in ((1, self.t_odom), (2, self.t_vesc), (4, self.t_drive))
                    if now - t > self.stale_s)
        if stale or self.state is None or self.state[3] < self.v_min:
            self.need_reset = True
            self._publish(-1 if (stale or self.state is None) else 0, stale)
            return
        x = self.state.copy()
        if self.need_reset:      # first cycle after a gap / slow segment: start a clean history
            self.lb_history.reset()
            self.reset_counter = 0
            self.need_reset = False
            one_step_cost = None
        else:
            one_step_cost = self.lb_history.update_lookback_error(x)

        self.best_idx = int(self.lb_history.get_best_model())
        self.best = self.dynamics_bank.get_model_params_arr(self.best_idx)
        muf, mur = float(self.best[2]), float(self.best[3])
        lf, lr = self.params_car['lf'], self.params_car['lr']
        mu = (muf * lr + mur * lf) / (lf + lr)
        self.last_mu = mu if self.last_mu is None else MU_EMA * mu + (1.0 - MU_EMA) * self.last_mu

        u = np.array([self.iq, self.steer])
        self.reset_counter = (self.reset_counter + 1) % LLA_RESET_INTERVAL
        self.dynamics_bank.update_known_params(self.omega_w, 0.0)    # no dFz estimate (as in LLA)
        self.lb_history.predict_states(x, u, self.reset_counter == 0)
        self._publish(1, 0)

        if self.logger_ is not None:
            if one_step_cost is not None:
                self.logger_.log_rollout_data(self.lb_history, one_step_cost, True)
            self.logger_.log_lla_data(x, np.asarray(self.best), self.best_idx, u,
                                      np.array(self.dynamics_bank.get_known_params()), u, 0.0,
                                      mu_est=self.last_mu)

    def _publish(self, status, stale):
        b = self.best
        mu = self.last_mu if self.last_mu is not None else float('nan')
        self.pub.publish(Float64MultiArray(data=[float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                                 float(mu), float(self.best_idx), float(status), float(stale)]))


def main(args=None):
    rclpy.init(args=args)
    node = LLAEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.logger_ is not None:
            node.logger_.save_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
