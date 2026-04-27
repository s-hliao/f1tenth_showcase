#!/usr/bin/env bash
# Run the calibration collect node inside the vision_lab_v2 conda environment,
# which has pyrealsense2.  Mirrors the pattern in final/run_depth_fusion.sh.
#
# Usage:
#   bash run_collect.sh
#   bash run_collect.sh --output-dir ~/my_calib_data

set -e

export DISPLAY="${DISPLAY:-:0}"

# Parse flags → ROS2 params
ROS_PARAMS=()
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) ROS_PARAMS+=("-p" "output_dir:=$2"); shift 2 ;;
        --scan-topic) ROS_PARAMS+=("-p" "scan_topic:=$2"); shift 2 ;;
        --cam-width)  ROS_PARAMS+=("-p" "cam_width:=$2");  shift 2 ;;
        --cam-height) ROS_PARAMS+=("-p" "cam_height:=$2"); shift 2 ;;
        --cam-fps)    ROS_PARAMS+=("-p" "cam_fps:=$2");    shift 2 ;;
        *)            PASSTHROUGH+=("$1");                 shift   ;;
    esac
done

ROS_ARGS=()
if [[ ${#ROS_PARAMS[@]} -gt 0 ]]; then
    ROS_ARGS+=("--ros-args" "${ROS_PARAMS[@]}")
fi
ROS_ARGS+=("${PASSTHROUGH[@]}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/calibration/collect_node.py"

export DISPLAY="${DISPLAY:-:0}"

source /opt/ros/humble/setup.bash
WS_SETUP="$SCRIPT_DIR/../../install/setup.bash"
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

# Use system python3 (has GTK3 cv2 → windows work) with vision_lab_v2's
# pyrealsense2 injected via PYTHONPATH.
RS2_SITE="/home/team10/miniconda3/envs/vision_lab_v2/lib/python3.10/site-packages"
SYS_CV2="/usr/lib/python3/dist-packages"
export PYTHONPATH="$SYS_CV2:$RS2_SITE:${PYTHONPATH}"

echo "=== calibration collect_node ==="
echo "  DISPLAY:  $DISPLAY"
echo "  output:   ${ROS_PARAMS[*]:-~/calibration_data (default)}"
echo ""

/usr/bin/python3 "$NODE_SCRIPT" "${ROS_ARGS[@]}"
