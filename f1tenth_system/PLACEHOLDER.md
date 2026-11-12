# f1tenth_system (placeholder — clone here, contents are gitignored)

Clone the f1tenth_system for **this car's ROS distro** into this folder:

| car | host | distro |
|---|---|---|
| car1 | dl-car.wifi.local.cmu.edu | Foxy |
| car2 | team10jetsonorin.wifi.local.cmu.edu | Humble |

```bash
cd <ws>/src            # this repo is the workspace's src/
rm f1tenth_system/PLACEHOLDER.md && rmdir f1tenth_system
git clone -b <foxy|humble branch> <f1tenth_system repo> f1tenth_system
```
Must provide `f1tenth_stack` (launch/bringup_launch.py: VESC driver, lidar, ackermann_mux) and `vesc_msgs`.
Everything in this folder except this file is gitignored: the clone stays on the car and is never committed.
If the car already has these packages in another workspace, leave this empty and source that workspace instead.
