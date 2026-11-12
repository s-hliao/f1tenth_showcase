# frenet_baseline

Frenet-frame pure pursuit Baseline with barrier-state (BaS) obstacle avoidance, per-cycle telemetry and post-run tools.

```
ros2 launch frenet_baseline baseline_pp_launch.py                                   # car, venue_ccw, stable line
ros2 launch frenet_baseline baseline_pp_launch.py track:=venue_cw line:=raceline_fast.csv
ros2 launch frenet_baseline baseline_pp_launch.py avoid:=true speed_scale:=0.9
ros2 launch frenet_baseline baseline_pp_launch.py params_file:=$HOME/f1tenth_ws/src/frenet_baseline/config/baseline_pp_car.yaml
ros2 run frenet_baseline telemetry_report.py <bag> --plot
ros2 run frenet_baseline grip_from_bag.py <bag>
```

| Path | Content |
|---|---|
| `frenet_baseline/baseline_pp.py` | controller module (params, raceline, Frenet tracking, BaS, steering, speed, ROS node) |
| `scripts/` | `baseline_pp_node.py` entry point, post-run tools |
| `config/` | `baseline_pp_car.yaml`, `baseline_pp_sim.yaml` |
| `launch/` | `baseline_pp_launch.py` |
| `waypoints/<track>/` | racelines from `make_racelines.py` (stable / optimal / fast, centerline, report) |
| `maps/` | reference copy of the venue map (the particle filter uses its own copy) |

Depends on `race_msgs` (Telemetry). Build: `colcon build --packages-select race_msgs frenet_baseline`.
