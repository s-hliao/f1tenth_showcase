# showcase_fleet

Two-car live demo. Each car drives on its own stack (PF → EKF → frenet_bv3 LQR → VESC). A ground-station laptop shows both cars in a real-time matplotlib view: live positions, the raceline with a ±r tube, interpolated trails that fade with time, and LLA Cf/Cr/μ for each car. It can also start and stop each car. **The cars do not depend on the laptop or the network to drive.**

| car | host | ROS |
|---|---|---|
| car1 | `dl-car.wifi.local.cmu.edu` | Foxy (20.04) |
| car2 | `team10jetsonorin.wifi.local.cmu.edu` | Humble (22.04) |
| ground | `spirit-of-fire.wifi.local.cmu.edu` (`LAPTOP_HOST` in `config/fleet.env`) | LLA container (Foxy + JAX) |

```
CAR  ── loopback-only DDS domain (car1: 42, car2: 43); nothing on Wi-Fi can reach these processes ──
  car_launch.py
    showcase_state_estimation/bringup_launch.py
      f1tenth_stack bringup   (VESC, lidar, mux)        /sensors/core, /odom, /scan
      particle_filter         localize                  /pf/pose/odom
      EKF (llampc-style)      PF + IMU + ZUPT           /odometry/filtered
    drive_gate   start/stop latch                       /showcase/drive_raw ──► /drive ──► mux ──► VESC
    fleet_relay  local process ⇄ bounded queue ⇄ network process ───────────────┐
  lqr_launch.py  (own terminal) frenet_bv3, unmodified; /drive remapped ──► /showcase/drive_raw
  carctl         (own terminal) start / stop from the car                       │
                                                                                │ Wi-Fi, NET domain 7
GROUND (NET domain 7)                                                           │
  lla_estimator ×2   /<car>/odom, vesc, drive ──► /<car>/lla/params  ◄──────────┤ best effort, car → laptop
  ground_station     live view; START/STOP buttons ──► /<car>/cmd    ───────────┘ reliable, volatile → car
```

## What keeps the cars independent
- **Domains.** The driving stack sits on a loopback-only CycloneDDS domain. Only `fleet_relay`'s network child process joins Wi-Fi.
- **Relay queue.** The relay's local side hands samples over through a bounded queue that drops when full, so a dead, slow or absent link can never delay the car. If the network side crashes, it is restarted every 3 s. If the relay's launcher dies, all relay processes exit.
- **State is latched on the car.** `drive_gate` keeps the car's state, so link loss changes nothing: a running car keeps lapping, and a stopped car stays stopped.
- **Commands are volatile.** Laptop commands are never replayed when a car reconnects. Only the exact verbs `start` and `stop` get in; nothing else from the network can reach the car.
- **No MPC or LLA on the cars.** LLA runs only on the laptop, as a bank-only estimator that publishes no drive commands.
- **Mixed ROS distros.** car1 runs Foxy and car2 runs Humble.
  - The relay uses one process per domain, so it works on both.
  - VESC data is flattened to a `Float64MultiArray` on the car, so different `vesc_msgs` versions don't matter, and the laptop doesn't need `vesc_msgs`.
  - `dds_env.sh` writes the Cyclone interface element each distro's Cyclone version understands.

## Repository layout (the repo is the workspace's `src/`)
```
<ws>/src = f1tenth_showcase/
  frenet_baseline/            LQR (frenet_bv3), racelines            (unchanged)
  showcase_state_estimation/  EKF + car bringup launch               (port of llampc's EKF)
  showcase_fleet/             relay, gate, carctl, ground station, LLA estimator, viz_ref_fiala, DDS configs
  f1tenth_system/             PLACEHOLDER: clone the car's distro version here (gitignored)
  particle_filter/            PLACEHOLDER: clone the car's distro version here (gitignored)
```
Each car clones its own distro's `f1tenth_system` and `particle_filter` into the placeholders, or sources a workspace that already has them. team10's `~/f1tenth_ws` does. The clones are gitignored.

## Setup
**Cars.** Install the DDS and EKF packages. team10 is missing `rmw_cyclonedds_cpp`; `robot_localization` is already there.
```bash
sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp ros-$ROS_DISTRO-robot-localization
git clone <this repo> ~/f1tenth_showcase/src     # ws = ~/f1tenth_showcase (matches bv3_car.yaml track.ROOT)
cd ~/f1tenth_showcase && colcon build --symlink-install
```

**Laptop.** Build the same repo inside the LLA container, which has JAX and llampc. The container must use `--network host`; DDS cannot reach the cars through Docker's NAT.
```bash
docker run -it --network host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/docker_workspaces/lla_drive_ws:/lla_ws -v ~/docker_workspaces/f1tenth_showcase:/showcase_ws/src lla:latest
source /lla_ws/install/setup.bash && cd /showcase_ws && colcon build --symlink-install
pip install pyqt5            # optional: QtAgg is faster than the TkAgg fallback
```

## Run
**Each car.** Run `source dds_env.sh car car1` (or `car2`) first in **every** terminal; it puts the shell on the car's loopback domain.
```bash
source ~/f1tenth_showcase/install/setup.bash
source ~/f1tenth_showcase/install/showcase_fleet/share/showcase_fleet/scripts/dds_env.sh car car2

ros2 launch showcase_fleet car_launch.py     # T1: f1tenth_system + PF + EKF + gate + relay (boots STOPPED)
                                             #     (cars are set up in ~/showcase_demo_ws: source its install/)
ros2 run showcase_fleet lqr                  # T2: the LQR + start/stop keys: s = start, space = stop, q = stop+quit
ros2 run showcase_fleet carctl               # optional, any other terminal: s / space (works with no laptop)
```
`car_launch` and `lqr_launch` refuse to start unless the shell has sourced `dds_env.sh car <car>`. Never run `frenet_bv3.py` by hand: without `lqr_launch`'s remaps it would bypass the gate. If that happens, the gate reports `!BYPASS` in its state.

To bring up only the sensors and the EKF (no gate or relay), run `ros2 launch showcase_state_estimation bringup_launch.py`.

**Laptop.**
```bash
source /showcase_ws/install/showcase_fleet/share/showcase_fleet/scripts/dds_env.sh laptop
ros2 launch showcase_fleet ground_launch.py                  # cars:=car2 to run one car
```
- **START ALL** starts the ticked cars. **STOP ALL** (or the space bar) stops every car.
- Each car also has its own start and stop buttons.
- A car that drops off is greyed out as LINK LOST. The other car is unaffected.
- The status line shows each car's state, link latency and speed, and why LLA is holding (slow, or no odom/vesc/drive data).

### Arguments
| launch | arg | default | |
|---|---|---|---|
| car_launch | `bringup` / `localize` / `ekf` | true | f1tenth_stack / particle_filter / EKF |
| car_launch | `ekf_source` | `pf` | `pf` or `mocap` |
| car_launch | `autostart` | false | gate boots RUNNING |
| car_launch | `stop_decel` | 4.0 | soft stop: speed ramp [m/s²] while still following the line |
| car_launch | `relay_config` | `config/relay.yaml` | topics broadcast to the laptop; add rows here, no code changes needed |
| lqr_launch | `avoid` | `true` | bv3 obstacle avoidance (BaS) on the LiDAR |
| lqr_launch | `pose` | `pf` | `dr` = dead reckoning; then `speed_max` defaults to 0.7 m/s (otherwise regular LQR speeds) |
| lqr_launch | `controller` | `bv3` | or `<package>/<executable>`; the obstacle-avoidance stack plugs in here |
| lqr_launch | `params` | `frenet_baseline/config/bv3_car.yaml` | |
| ground_launch | `track` / `line` | `venue_ccw` / `raceline_stable.csv` | the raceline the LQR follows |
| ground_launch | `tube_radius` / `trail_s` | 0.3 / 3.0 | |
| ground_launch | `log` | false | the LLA estimators write `.npz` files for `viz_ref_fiala` replay |

### Gate behaviour
- **STOPPED** (boot state): holds zero.
- **RUNNING:** forwards the controller output unchanged.
- **STOPPING:** ramps the speed down, then goes to STOPPED.
- **Watchdog:** once the controller has published since the start, 0.2 s of silence stops the car. A new start is needed.

## ETHZMobil by dead reckoning (first shot, before the track has a map)
`frenet_baseline/waypoints/ethzmobil_cw` and `ethzmobil_ccw` hold BayesRace's ETHZMobil centreline ×4.3 (ORCA 1:43 → F1TENTH 1:10): 12.9 × 7.7 m, 55.3 m lap, tightest radius 1.28 m. `scripts/make_track.py` regenerates them.

**(0, 0) is the middle of the longest straight** (8.8 m long), and +x is the driving direction there. Place the car on that mark facing along the straight, with the **LiDAR over the mark** (the LQR tracks the laser point). Then, after `source dds_env.sh car carN`:
```bash
ros2 run showcase_state_estimation imu_check        # once per car (bringup running): turn the car left -> OK
ros2 launch showcase_fleet car_launch.py mode:=dr localize:=false      # T1: keep the car still ~3 s (gyro bias)
ros2 run showcase_fleet lqr track:=ethzmobil_cw line:=raceline_dr.csv pose:=dr   # T2: s = start, space = stop, q = quit
```
Ground station: `ros2 launch showcase_fleet ground_launch.py track:=ethzmobil_cw line:=raceline_dr.csv`.

How dead reckoning works:
- The EKF integrates VESC wheel speed plus gyro yaw rate from the start pose (`ekf_dr.yaml`).
- The gyro bias is measured during the first 3 s at standstill.
- `odom_points` publishes the laser point the LQR tracks (`/dr/pose/odom`) and the cg (`/odometry/filtered`).

Accuracy: in sim, with a 0.4 deg/s gyro bias, the dead-reckoned pose drifted at most 42 cm over 3.4 laps. Relaunch `car_launch` to reset the origin. If `imu_check` says the gyro is negative, add `yaw_sign:=-1` to `car_launch`.

After the first laps: build the physical track around the driven line, map it with slam_toolbox, then run the particle filter on that map and switch back to `mode:=pf` / `pose:=pf`. The racelines are in the dead-reckoning frame, so they have to be re-registered to the map frame.

## LLA estimator
`lla_estimator` builds LLA's model bank exactly as llampc's `multi_llampc_fiala.py` does: same grid, window, γ, reset interval and 0.04 s step. llampc is imported, not modified.

The bank is driven by what the car applied: VESC `avg_iq` and the gated `/drive` steer. It is not driven by LLA's own MPC output. The estimator pauses when the car is below 0.3 m/s or its inputs are stale, and resets its history when it resumes. Each estimator uses about 30% of one core.

To replay a run:
```bash
ros2 run showcase_fleet viz_ref_fiala ~/.ros/log/showcase_car1_lla_<ts>.npz --ref <raceline.csv|track.npz>
```
The bank uses llampc's `F110()` vehicle parameters. If a showcase car differs, its estimates will be biased.

## Tests run
- **Docker, cross-distro.** A Foxy "car", a Humble "car" and the Foxy LLA "laptop" on a bridge network, using the real (non-loopback) Cyclone configs.
  - The real `frenet_bv3` LQR drove simulated cars through the gates, with the real EKF.
  - The laptop saw only `/carN/*` topics.
  - Both cars ran at 40 Hz. One was started from the car itself, the other from the ground.
  - With a car's network cut for 10–16 s, its local `/drive` stayed at 40 Hz (max gap 29 ms) and it stayed RUNNING. On reconnect it resumed with no command replayed.
  - The gate flagged a bypassing `/drive` publisher.
  - The LLA estimators ran on relayed VESC data, and the live GUI ran under Xvfb.
- **On team10 (Humble).** A simulated run only: `fake_car` replaced VESC, lidar and PF, nothing touched hardware, it used FastDDS localhost on spare domains, and the scratch workspace was removed afterwards.
  - The build was clean on the `f1tenth_ws` underlay.
  - EKF, gate, relay and the real LQR worked: a ground start reached RUNNING with `/drive` at 40 Hz, and a local stop gave STOPPING, then STOPPED.
- **dl-car (Foxy):** unreachable at the time of testing, so it is covered only by the Docker Foxy run.

## Desk test (one machine, no cars)
Every line runs in its own terminal, after `source dds_env.sh local <who>`:
```bash
ros2 run showcase_fleet fake_car --ros-args -p start_s:=0.0   # local car1; use start_s:=0.5 for car2
ros2 run showcase_fleet drive_gate                            # local car1 / car2
ros2 run showcase_fleet fleet_relay                           # local car1 / car2
ros2 launch showcase_fleet ground_launch.py                   # local laptop
```
`fake_car controller:=false` hands steering to the real LQR (`lqr_launch.py`). `fake_car ekf:=false` hands `/odometry/filtered` to the real EKF.
