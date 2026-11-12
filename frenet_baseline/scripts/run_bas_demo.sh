#!/usr/bin/env bash
# One demo run of bv1 + BaS at a given speed cap, with a bag for bas_plot.py.
#   ./run_bas_demo.sh 2.0          (then 3.0, 4.0, ...)   add "cw" as 2nd arg to drive clockwise
# Start the map server / particle filter / EKF first, as for a normal bv1 run.
# Ctrl+C stops the node and the recording together.
set -e
V=${1:?usage: ./run_bas_demo.sh <speed_max m/s>}
[[ $V == *.* ]] || V=$V.0          # ROS 2 needs a float here (3 -> 3.0)
DIRN=${2:-ccw}                     # ./run_bas_demo.sh 3.0 cw  for the clockwise direction
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${BAG_DIR:-$HOME/bas_bags}
NAME=bas_v${V}_$(date +%H%M%S)
mkdir -p "$OUT"
source ~/f1tenth_ws/install/setup.bash

ros2 bag record -o "$OUT/$NAME" /bas/debug /bas/markers /pf/pose/odom /drive /scan /odom /odometry/filtered &
BAG=$!
trap 'kill -INT $BAG 2>/dev/null; wait $BAG 2>/dev/null; echo; echo "bag: $OUT/$NAME"; \
      echo "plot: python3 $DIR/bas_plot.py $OUT/$NAME --waypoints $DIR/aims_centerline_$DIRN.csv"' EXIT
sleep 1
python3 "$DIR/frenet_pursuit_bas.py" --ros-args --params-file "$DIR/bas_demo.yaml" -p speed_max:=$V \
    -p waypoint_file:=$DIR/aims_centerline_$DIRN.csv
