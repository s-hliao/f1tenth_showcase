"""ground_station: live demo view + start/stop for the fleet (laptop, NET domain).

  * track: raceline (same CSV the LQR follows) with a shaded tube of +/- tube_radius around it
  * each car: heading marker + spline-interpolated trail that fades out over trail_s (no dots)
  * right column: LLA Cf, Cr and mu for every car (from lla_estimator, /<car>/lla/params)
  * buttons: START ALL (cars ticked "include"), STOP ALL (every car, always), per-car START / STOP
    keyboard: space = STOP ALL
  * a car silent for > link_timeout_s is greyed out as LINK LOST; the other keeps going

Commands go to /<car>/cmd ("start"/"stop"), which the car's fleet_relay hands to its drive_gate. The car's state
is latched on the car; this window only displays what the car reports on /<car>/state.

    ros2 run showcase_fleet ground_station --ros-args -p cars:="car1,car2" -p track:=venue_ccw
"""
import os
import threading
import time
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from showcase_fleet import common as C

CAR_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']   # car1 blue, car2 red, ...
GREY = '#9a9a9a'


# ======================================================================================================================
# ROS side
# ======================================================================================================================
class CarFeed:
    def __init__(self, name, color):
        self.name, self.color = name, color
        self.lock = threading.Lock()
        self.trail = deque()            # (car stamp, x, y)
        self.pose = None                # (x, y, yaw, v)
        self.t_rx = None                # laptop monotonic time of last message of any kind
        self.state = 'NO DATA'
        self.lla = deque()              # (laptop t, Cf, Cr, mu, active)


class GroundNode(Node):
    def __init__(self):
        super().__init__('ground_station')
        p = self.declare_parameter
        p('cars', ','.join(C.car_names_from_env()))
        p('pose_topic', C.NET_ODOM)          # relative to /<car>/ ; 'pf_odom' to show the raw PF pose
        p('track', 'venue_ccw')
        p('line', 'raceline_stable.csv')
        p('raceline_file', '')
        p('tube_radius', 0.3)
        p('trail_s', 3.0)
        p('history_s', 30.0)
        p('link_timeout_s', 1.0)
        p('fps', 30.0)                       # redraw rate of the live view
        p('blit', True)                      # redraw only the moving artists (much faster than full redraws)
        p('headless', False)                 # tests: Agg backend, no window
        p('snapshot', '')                    # tests: save a PNG after snapshot_after_s and exit
        p('snapshot_after_s', 8.0)
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.cfg = {k: g(k) for k in ('track', 'line', 'raceline_file', 'tube_radius', 'trail_s', 'history_s',
                                      'link_timeout_s', 'fps', 'blit', 'headless', 'snapshot',
                                      'snapshot_after_s')}
        names = [c.strip() for c in str(g('cars')).split(',') if c.strip()]
        self.cars = [CarFeed(n, CAR_COLORS[i % len(CAR_COLORS)]) for i, n in enumerate(names)]
        self.cmd_pubs = {}
        be = C.qos_best_effort(5)
        for car in self.cars:
            n = car.name
            self.create_subscription(Odometry, C.net_topic(n, g('pose_topic')), self._mk_pose(car), be)
            self.create_subscription(String, C.net_topic(n, C.NET_STATE), self._mk_state(car), be)
            self.create_subscription(Float64MultiArray, C.net_topic(n, C.NET_LLA), self._mk_lla(car),
                                     C.qos_reliable(10))
            self.cmd_pubs[n] = self.create_publisher(String, C.net_topic(n, C.NET_CMD), C.qos_reliable(10))

    def _mk_pose(self, car):
        def cb(msg):
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
            v = float(np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))
            keep = self.cfg['trail_s'] + 1.0
            with car.lock:
                if car.trail and st <= car.trail[-1][0]:
                    if st < car.trail[-1][0] - 5.0:     # car clock jumped back (restart): start over
                        car.trail.clear()
                    else:
                        return                           # out-of-order / duplicate
                car.trail.append((st, x, y))
                while car.trail and car.trail[0][0] < st - keep:
                    car.trail.popleft()
                car.pose = (x, y, C.yaw_from_quat(msg.pose.pose.orientation), v)
                car.t_rx = time.monotonic()
        return cb

    def _mk_state(self, car):
        def cb(msg):
            with car.lock:
                car.state = msg.data
                car.t_rx = time.monotonic()
        return cb

    def _mk_lla(self, car):
        def cb(msg):
            d = msg.data
            if len(d) < 7:
                return
            now = time.monotonic()
            with car.lock:
                stale = int(d[7]) if len(d) > 7 else 0
                car.lla.append((now, d[0], d[1], d[4], d[6], stale))
                while car.lla and car.lla[0][0] < now - self.cfg['history_s'] - 1.0:
                    car.lla.popleft()
        return cb

    def send(self, car_name, verb):
        self.cmd_pubs[car_name].publish(String(data=verb))
        self.get_logger().info(f'{verb} -> {car_name}')


# ======================================================================================================================
# GUI side
# ======================================================================================================================
def _pick_backend(headless):
    import matplotlib
    if headless:
        matplotlib.use('Agg')
        return
    import importlib
    for b, mod in (('QtAgg', 'backend_qtagg'), ('Qt5Agg', 'backend_qt5agg'), ('TkAgg', 'backend_tkagg')):
        try:   # import the backend first: matplotlib.use() alone defers a missing Qt/Tk to the first figure
            importlib.import_module('matplotlib.backends.' + mod)
            matplotlib.use(b, force=True)
            return
        except Exception:  # noqa: BLE001
            continue


TRAIL_BANDS = 12     # trail drawn as N polylines of stepped alpha/width: looks continuous, ~10x cheaper than a
                     # per-segment LineCollection


def _smooth_trail(pts, upsample=3):
    """Spline-interpolate (t, x, y) samples in time. Falls back to the raw polyline."""
    t, x, y = (np.asarray(c, dtype=float) for c in zip(*pts))
    if len(t) < 4:
        return t, x, y
    try:
        from scipy.interpolate import CubicSpline
        tt = np.linspace(t[0], t[-1], (len(t) - 1) * upsample + 1)
        return tt, CubicSpline(t, x)(tt), CubicSpline(t, y)(tt)
    except Exception:  # noqa: BLE001
        return t, x, y


class GroundGUI:
    def __init__(self, node):
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch, Polygon
        from matplotlib.path import Path
        from matplotlib.widgets import Button, CheckButtons
        self.plt, self.node, self.cfg = plt, node, node.cfg
        self.Polygon = Polygon
        for k in [k for k in plt.rcParams if k.startswith('keymap.')]:
            plt.rcParams[k] = []          # no accidental save / fullscreen / quit keys during a demo

        self.fig = plt.figure(figsize=(16, 9))
        try:
            self.fig.canvas.manager.set_window_title('Showcase ground station')
        except Exception:  # noqa: BLE001
            pass
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.7, 1.0], left=0.04, right=0.98, top=0.95,
                                   bottom=0.24, wspace=0.12, hspace=0.25)
        self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_p = [self.fig.add_subplot(gs[i, 1]) for i in range(3)]

        # ---- track + tube ----
        path = C.resolve_raceline(self.cfg['track'], self.cfg['line'], self.cfg['raceline_file'])
        rl = C.load_raceline(path)
        r = float(self.cfg['tube_radius'])
        lx, ly = C.offset_curve(rl['x'], rl['y'], r, rl['closed'])
        rx, ry = C.offset_curve(rl['x'], rl['y'], -r, rl['closed'])
        if rl['closed']:
            ring = lambda xs, ys: np.column_stack([np.append(xs, xs[0]), np.append(ys, ys[0])])  # noqa: E731
            a, b = ring(lx, ly), ring(rx, ry)[::-1]
            codes = lambda n: [Path.MOVETO] + [Path.LINETO] * (n - 2) + [Path.CLOSEPOLY]  # noqa: E731
            tube = PathPatch(Path(np.vstack([a, b]), codes(len(a)) + codes(len(b))),
                             facecolor='#7fb3d5', edgecolor='none', alpha=0.25, zorder=1)
        else:
            tube = Polygon(np.column_stack([np.r_[lx, rx[::-1]], np.r_[ly, ry[::-1]]]),
                           facecolor='#7fb3d5', edgecolor='none', alpha=0.25, zorder=1)
        self.ax.add_patch(tube)
        for ox, oy in ((lx, ly), (rx, ry)):
            self.ax.plot(ox, oy, color='#5d8aa8', lw=0.8, alpha=0.6, zorder=2)
        self.ax.plot(rl['x'], rl['y'], 'k--', lw=1.2, alpha=0.6, zorder=3,
                     label=f'raceline ({os.path.basename(path)}), tube ±{r:g} m')
        self.ax.set_aspect('equal')
        pad = 1.0
        self.extent = float(max(np.ptp(rl['x']), np.ptp(rl['y'])))
        self.ax.set_xlim(rl['x'].min() - pad, rl['x'].max() + pad)
        self.ax.set_ylim(rl['y'].min() - pad, rl['y'].max() + pad)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.grid(alpha=0.2)

        # ---- per-car artists ----
        self.trails, self.markers, self.labels = {}, {}, {}
        self.series = {}
        titles = ['Cf (LLA)', 'Cr (LLA)', 'μ (LLA)']
        ylims = [(150, 350), (125, 325), (0.0, 1.2)]
        for ax, ttl, yl in zip(self.ax_p, titles, ylims):
            ax.set_title(ttl, loc='left', fontsize=11)
            ax.set_xlim(-self.cfg['history_s'], 0)
            ax.set_ylim(*yl)
            ax.grid(alpha=0.25)
        self.ax_p[-1].set_xlabel('time [s]')
        for car in node.cars:
            self.trails[car.name] = [self.ax.plot([], [], color=car.color, lw=1, solid_capstyle='round',
                                                  zorder=5)[0] for _ in range(TRAIL_BANDS)]
            m = Polygon(np.zeros((3, 2)), closed=True, facecolor=car.color, edgecolor='k', lw=0.8, zorder=6)
            m.set_visible(False)
            self.ax.add_patch(m)
            self.markers[car.name] = m
            self.labels[car.name] = self.ax.text(0, 0, car.name, fontsize=10, fontweight='bold',
                                                 color=car.color, zorder=7, visible=False, clip_on=True)
            self.series[car.name] = [ax.plot([], [], color=car.color, lw=2, label=car.name)[0]
                                     for ax in self.ax_p]
        self.ax_p[0].legend(loc='upper left', fontsize=9, ncol=len(node.cars))
        self.ax.legend(loc='upper right', fontsize=9)

        # ---- controls ----
        self.widgets = []
        b_start = Button(self.fig.add_axes([0.04, 0.12, 0.10, 0.07]), 'START ALL', color='#c8e6c9',
                         hovercolor='#a5d6a7')
        b_stop = Button(self.fig.add_axes([0.04, 0.03, 0.10, 0.07]), 'STOP ALL\n(space)', color='#ffcdd2',
                        hovercolor='#ef9a9a')
        b_start.on_clicked(lambda _e: self.start_all())
        b_stop.on_clicked(lambda _e: self.stop_all())
        self.widgets += [b_start, b_stop]
        self.include = {}
        self.status = {}
        row_h = 0.19 / max(1, len(node.cars))
        for i, car in enumerate(node.cars):
            y = 0.03 + (len(node.cars) - 1 - i) * row_h
            h = min(0.07, row_h * 0.8)
            chk = CheckButtons(self.fig.add_axes([0.16, y, 0.07, h]), [car.name], [True])
            for t in chk.labels:
                t.set_color(car.color)
                t.set_fontweight('bold')
            bs = Button(self.fig.add_axes([0.24, y, 0.06, h]), 'start', color='#e8f5e9')
            bx = Button(self.fig.add_axes([0.31, y, 0.06, h]), 'stop', color='#ffebee')
            bs.on_clicked(lambda _e, n=car.name: self.node.send(n, 'start'))
            bx.on_clicked(lambda _e, n=car.name: self.node.send(n, 'stop'))
            self.widgets += [chk, bs, bx]
            self.include[car.name] = chk
            self.status[car.name] = self.fig.text(0.39, y + h / 2, '', va='center', fontsize=11,
                                                  family='monospace')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # ---- real-time redraw: blit only what moves -----------------------------------------------------
        self.fps_text = self.fig.text(0.985, 0.99, '', ha='right', va='top', fontsize=9, color='#777')
        self.dyn = ([self.fps_text] + [b for v in self.trails.values() for b in v] + list(self.markers.values()) +
                    list(self.labels.values()) + [ln for v in self.series.values() for ln in v] +
                    list(self.status.values()))
        self.blit = bool(self.cfg['blit']) and not self.cfg['headless'] and \
            getattr(self.fig.canvas, 'supports_blit', False)
        self.bg = None
        self.frames = deque(maxlen=90)
        self._set_regions()
        if self.blit:
            for a in self.dyn:
                a.set_animated(True)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)

    def _set_regions(self):
        # only these regions change each frame; blitting them instead of the whole window is much cheaper
        from matplotlib.transforms import Bbox
        fb = self.fig.bbox
        strip = Bbox.from_extents(0.38 * fb.width, 0.0, fb.width, 0.23 * fb.height)       # status lines
        top = Bbox.from_extents(0.85 * fb.width, 0.965 * fb.height, fb.width, fb.height)  # fps
        # one blit of their union: per-call overhead dominates on TkAgg
        self.regions = [Bbox.union([self.ax.bbox] + [a.bbox for a in self.ax_p] + [strip, top])]

    def _on_draw(self, _ev):
        # a full redraw (first show, resize, widget repaint) excludes animated artists: grab the static
        # background, then put the moving artists back on top
        self._set_regions()                     # window may have been resized
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for a in self.dyn:
            self.fig.draw_artist(a)

    def save(self, path):
        for a in self.dyn:
            a.set_animated(False)
        self.fig.savefig(path, dpi=90)

    # ---- commands ----
    def start_all(self):
        for car in self.node.cars:
            if self.include[car.name].get_status()[0]:
                self.node.send(car.name, 'start')

    def stop_all(self):
        for car in self.node.cars:       # safety: STOP ALL ignores the include boxes
            self.node.send(car.name, 'stop')

    def on_key(self, ev):
        if ev.key == ' ':
            self.stop_all()

    # ---- redraw ----
    def update(self, *_):
        now = time.monotonic()
        trail_s = float(self.cfg['trail_s'])
        hist = float(self.cfg['history_s'])
        for car in self.node.cars:
            with car.lock:
                pts = list(car.trail)
                pose, t_rx, state = car.pose, car.t_rx, car.state
                lla = list(car.lla)
            age = None if t_rx is None else max(0.0, now - t_rx)
            lost = age is None or age > self.cfg['link_timeout_s']
            color = GREY if lost else car.color
            rgb = np.array(self._rgb(color))

            # trail: interpolated polyline, fading with age on the CAR's clock (no laptop/car clock sync needed)
            bands = self.trails[car.name]
            pts = [p for p in pts if p[0] >= pts[-1][0] - trail_s] if pts else pts
            if len(pts) >= 2:
                t, x, y = _smooth_trail(pts)
                edges = np.linspace(0, len(t) - 1, len(bands) + 1).astype(int)
                for b, line in enumerate(bands):
                    i0, i1 = edges[b], edges[b + 1]
                    if i1 <= i0:
                        line.set_data([], [])
                        continue
                    a = float(np.clip(1.0 - (t[-1] - 0.5 * (t[i0] + t[i1])) / trail_s, 0.0, 1.0))
                    line.set_data(x[i0:i1 + 1], y[i0:i1 + 1])        # bands share end points: no gaps
                    line.set_color((*rgb, a * (0.35 if lost else 1.0)))
                    line.set_linewidth(0.8 + 3.2 * a)
            else:
                for line in bands:
                    line.set_data([], [])

            m, lab = self.markers[car.name], self.labels[car.name]
            if pose is not None:
                x0, y0, yaw, v = pose
                L = max(0.55, 0.03 * self.extent)
                W = 0.6 * L
                body = np.array([[L * 0.6, 0.0], [-L * 0.4, W / 2], [-L * 0.4, -W / 2]])
                c, s = np.cos(yaw), np.sin(yaw)
                m.set_xy(body @ np.array([[c, s], [-s, c]]) + [x0, y0])
                m.set_facecolor(color)
                m.set_alpha(0.5 if lost else 1.0)
                m.set_visible(True)
                xl = self.ax.get_xlim()
                right = x0 > xl[0] + 0.85 * (xl[1] - xl[0])        # keep the name inside the plot
                lab.set_position((x0 - 0.6 * L if right else x0 + 0.6 * L, y0 + 0.6 * L))
                lab.set_horizontalalignment('right' if right else 'left')
                lab.set_color(color)
                lab.set_visible(True)

            # LLA series
            if lla:
                arr = np.array(lla)
                tt = arr[:, 0] - now
                keep = np.flatnonzero(tt >= -hist)
                keep = keep[::max(1, len(keep) // 300)]          # ~300 points per curve is plenty on screen
                for line, col in zip(self.series[car.name], (1, 2, 3)):
                    line.set_data(tt[keep], arr[keep, col])
                    line.set_color(color)
                last = arr[-1]
                mu = f'{last[3]:4.2f}' if np.isfinite(last[3]) else ' -- '
                est = f'Cf {last[1]:6.1f}  Cr {last[2]:6.1f}  μ {mu}'
                if last[4] == 0:
                    est += '  (hold: slow)'
                elif last[4] < 0:
                    missing = [n for bit, n in ((1, 'odom'), (2, 'vesc'), (4, 'drive')) if int(last[5]) & bit]
                    est += f"  (hold: no {'/'.join(missing) or 'data'})"
            else:
                est = 'LLA: waiting'

            if lost:
                link = 'NO DATA ' if age is None else f'LINK LOST ({age:4.1f}s)'
                txt = f'{car.name}  {link:<16} last {state:<16} {est}'
            else:
                v = pose[3] if pose is not None else 0.0
                txt = f'{car.name}  {state:<16} link {age * 1000:4.0f} ms  v {v:4.1f} m/s   {est}'
            self.status[car.name].set_text(txt)
            self.status[car.name].set_color(color)

        self.frames.append(now)
        if len(self.frames) > 1:
            self.fps_text.set_text(f'{(len(self.frames) - 1) / (self.frames[-1] - self.frames[0]):4.1f} fps'
                                   + ('  (blit)' if self.blit else ''))
        canvas = self.fig.canvas
        if not self.blit:
            canvas.draw_idle()
        elif self.bg is None:
            canvas.draw()                       # -> _on_draw grabs the background
            canvas.blit(self.fig.bbox)
        else:
            canvas.restore_region(self.bg)
            for a in self.dyn:
                self.fig.draw_artist(a)
            for r in self.regions:
                canvas.blit(r)

    @staticmethod
    def _rgb(c):
        from matplotlib.colors import to_rgb
        return to_rgb(c)


def main(args=None):
    rclpy.init(args=args)
    node = GroundNode()
    headless = bool(node.cfg['headless'])
    _pick_backend(headless)
    import matplotlib.pyplot as plt

    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    gui = GroundGUI(node)
    try:
        period = 1.0 / max(1.0, float(node.cfg['fps']))
        snap = node.cfg['snapshot']
        if headless:
            t_end = time.monotonic() + float(node.cfg['snapshot_after_s'])
            while time.monotonic() < t_end:
                gui.update()
                time.sleep(period)
            gui.update()
            gui.save(snap or '/tmp/ground_station.png')
            node.get_logger().info(f"snapshot saved to {snap} ({gui.fps_text.get_text()})")
        else:
            timer = gui.fig.canvas.new_timer(interval=int(period * 1000))
            timer.add_callback(gui.update)
            timer.start()
            if snap:   # test hook: save + close after snapshot_after_s with the real GUI loop running
                def _snap():
                    gui.save(snap)
                    node.get_logger().info(f"snapshot saved to {snap} ({gui.fps_text.get_text()})")
                    plt.close(gui.fig)
                t2 = gui.fig.canvas.new_timer(interval=int(float(node.cfg['snapshot_after_s']) * 1000))
                t2.single_shot = True
                t2.add_callback(_snap)
                t2.start()
                gui._snap_timer = t2
            gui._timer = timer
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
