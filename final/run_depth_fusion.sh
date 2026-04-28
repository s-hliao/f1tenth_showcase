#!/usr/bin/env bash
# run_depth_fusion.sh
#
# Launch depth_fusion_node inside the conda vision_lab_v2 environment
# (CUDA torch + DAv2) with the ROS2 Humble Python packages active.
#
# ---- Guidance source -------------------------------------------------------
#   --guidance camera|lidar   Scale anchor: RealSense depth (camera, default)
#                             or projected 2D LiDAR scan (lidar).
#
# ---- LiDAR extrinsics (required with --guidance lidar) --------------------
#   --tx M  --ty M  --tz M    LiDAR→camera translation in metres.
#                             e.g. --tx 0.05 --ty 0.1 --tz 0.3
#   --rx R  --ry R  --rz R    LiDAR→camera rotation, ZYX Euler angles in RADIANS
#                             (roll, pitch, yaw).
#                             Defaults: rx=1.5708  ry=-1.5708  rz=0.0
#                             (standard LiDAR x-fwd/y-left/z-up → camera x-right/y-down/z-fwd)
#   --scan-topic TOPIC        LaserScan topic.  Default: /scan
#
# ---- DAv2 / RealSense ------------------------------------------------------
#   --anchor-row R        Fix scale-fit anchor row (camera mode only).
#                         Omit (or pass -1) for auto-select per frame.
#   --max-gt-depth M      Cap RealSense GT at M metres.  Default: 3.0
#   --max-depth M         Clip predicted depth at M metres.  Default: 20.0
#   --inlier-rel-tol T    RANSAC inlier relative-error threshold.  Default: 0.30
#   --display-vmax V      Colormap max in metres (0 = auto).  Default: 0.0
#   --cam-fps N           RealSense FPS.  Default: 30
#   --trt                 Load TRT engine instead of .pth
#
# Any unrecognised flags (including raw --ros-args) are forwarded to the node.
#
# Examples:
#   bash run_depth_fusion.sh
#   bash run_depth_fusion.sh --guidance camera --max-gt-depth 3.0 --anchor-row 250 --trt
#   bash run_depth_fusion.sh --guidance lidar --tx 0.05 --ty 0.1 --tz 0.3 --trt --cam-fps 15 --inlier-rel-tol 0.1 --max-gt-depth 3.0
#   bash run_depth_fusion.sh --guidance lidar --tx 0.05 --ty 0.1 --tz 0.3 \
#                            --rx 0.0 --ry -0.17 --rz 0.0 --scan-topic /scan
#   bash run_depth_fusion.sh --trt --cam-fps 15

set -e

# ---- Display ---------------------------------------------------------------
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts}"

# ---- Parse named flags → ROS2 params ---------------------------------------
ROS_PARAMS=()
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guidance)       ROS_PARAMS+=("-p" "guidance_mode:=$2");  shift 2 ;;
        --scan-topic)     ROS_PARAMS+=("-p" "scan_topic:=$2");     shift 2 ;;
        --tx)             ROS_PARAMS+=("-p" "tx:=$2");             shift 2 ;;
        --ty)             ROS_PARAMS+=("-p" "ty:=$2");             shift 2 ;;
        --tz)             ROS_PARAMS+=("-p" "tz:=$2");             shift 2 ;;
        --rx)             ROS_PARAMS+=("-p" "rx:=$2");             shift 2 ;;
        --ry)             ROS_PARAMS+=("-p" "ry:=$2");             shift 2 ;;
        --rz)             ROS_PARAMS+=("-p" "rz:=$2");             shift 2 ;;
        --anchor-row)     ROS_PARAMS+=("-p" "anchor_row:=$2");     shift 2 ;;
        --max-gt-depth)   ROS_PARAMS+=("-p" "max_gt_depth:=$2");   shift 2 ;;
        --max-depth)      ROS_PARAMS+=("-p" "max_depth:=$2");      shift 2 ;;
        --inlier-rel-tol) ROS_PARAMS+=("-p" "inlier_rel_tol:=$2"); shift 2 ;;
        --display-vmax)   ROS_PARAMS+=("-p" "display_vmax:=$2");   shift 2 ;;
        --cam-fps)        ROS_PARAMS+=("-p" "cam_fps:=$2");        shift 2 ;;
        --cam-width)      ROS_PARAMS+=("-p" "cam_width:=$2");      shift 2 ;;
        --cam-height)     ROS_PARAMS+=("-p" "cam_height:=$2");     shift 2 ;;
        --trt)            ROS_PARAMS+=("-p" "use_trt:=true");      shift   ;;
        *)                PASSTHROUGH+=("$1");                     shift   ;;
    esac
done

# Build final ROS2 args: named params first, then anything unrecognised
ROS_ARGS=()
if [[ ${#ROS_PARAMS[@]} -gt 0 ]]; then
    ROS_ARGS+=("--ros-args" "${ROS_PARAMS[@]}")
fi
ROS_ARGS+=("${PASSTHROUGH[@]}")

# ---- Paths -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/depth_fusion/depth_fusion/depth_fusion_node.py"

# ---- ROS2 environment ------------------------------------------------------
source /opt/ros/humble/setup.bash

WS_SETUP="$SCRIPT_DIR/../../install/setup.bash"
if [[ -f "$WS_SETUP" ]]; then
    source "$WS_SETUP"
fi

# ---- DAv2 checkpoint -------------------------------------------------------
export DAV2_PATH="${DAV2_PATH:-$HOME/Depth-Anything-V2}"
export DAV2_CHECKPOINT="${DAV2_CHECKPOINT:-$SCRIPT_DIR/checkpoints/depth_anything_v2_vits.pth}"
export DAV2_CHECKPOINT_TRT="${DAV2_CHECKPOINT_TRT:-$SCRIPT_DIR/checkpoints/depth_anything_v2_vits.engine}"

# ---- Detect key flags from collected ROS_PARAMS ----------------------------
USE_TRT=false
GUIDANCE=camera
for p in "${ROS_PARAMS[@]}"; do
    [[ "$p" == "use_trt:=true"         ]] && USE_TRT=true
    [[ "$p" == "guidance_mode:=lidar"  ]] && GUIDANCE=lidar
    [[ "$p" == "guidance_mode:=camera" ]] && GUIDANCE=camera
done

if [[ "$USE_TRT" == "true" ]]; then
    if [[ ! -f "$DAV2_CHECKPOINT_TRT" ]]; then
        echo "[ERROR] TRT engine not found: $DAV2_CHECKPOINT_TRT"
        echo "        Run: conda run -n vision_lab_v2 python export_trt.py"
        exit 1
    fi
else
    if [[ ! -f "$DAV2_CHECKPOINT" ]]; then
        echo "[ERROR] DAv2 checkpoint not found: $DAV2_CHECKPOINT"
        echo "        Run bash setup/setup.sh first, or set DAV2_CHECKPOINT."
        exit 1
    fi
fi

# ---- Warn if LiDAR guidance selected without setting tx/ty/tz ---------------
if [[ "$GUIDANCE" == "lidar" ]]; then
    _has_tx=false; _has_ty=false; _has_tz=false
    for p in "${ROS_PARAMS[@]}"; do
        [[ "$p" == tx:=* ]] && _has_tx=true
        [[ "$p" == ty:=* ]] && _has_ty=true
        [[ "$p" == tz:=* ]] && _has_tz=true
    done
    if [[ "$_has_tx" == false || "$_has_ty" == false || "$_has_tz" == false ]]; then
        echo "[WARN] --guidance lidar selected but --tx/--ty/--tz not all set."
        echo "       LiDAR beams will be projected assuming zero translation."
        echo "       Pass --tx 0.05 --ty 0.1 --tz 0.3 (and --ry for pitch)."
    fi
fi

# ---- Summary ---------------------------------------------------------------
echo "=== depth_fusion_node ==="
echo "  DISPLAY:         $DISPLAY"
echo "  DAV2_PATH:       $DAV2_PATH"
if [[ "$USE_TRT" == "true" ]]; then
    echo "  BACKEND:         TensorRT  ($DAV2_CHECKPOINT_TRT)"
else
    echo "  BACKEND:         PyTorch   ($DAV2_CHECKPOINT)"
fi
echo "  GUIDANCE:        $GUIDANCE"
echo "  ROS_DISTRO:      $ROS_DISTRO"
if [[ ${#ROS_PARAMS[@]} -gt 0 ]]; then
    echo "  ROS params:      ${ROS_PARAMS[*]}"
fi
echo ""

# ---- Launch ----------------------------------------------------------------
PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 --no-capture-output \
    python3 "$NODE_SCRIPT" "${ROS_ARGS[@]}"
