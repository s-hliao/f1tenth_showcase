# particle_filter (placeholder — clone here, contents are gitignored)

Clone the particle_filter for **this car's ROS distro** into this folder (car1 dl-car: Foxy, car2 team10: Humble):

```bash
cd <ws>/src            # this repo is the workspace's src/
rm particle_filter/PLACEHOLDER.md && rmdir particle_filter
git clone -b <foxy|humble branch> <particle_filter repo> particle_filter
```
Must provide `launch/localize_launch.py` publishing `/pf/pose/odom` (~40 Hz: the LQR runs once per pose).
Everything in this folder except this file is gitignored: the clone stays on the car and is never committed.
If the car already has this package in another workspace, leave this empty and source that workspace instead.
