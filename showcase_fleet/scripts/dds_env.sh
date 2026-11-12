#!/bin/bash
# dds_env.sh: CycloneDDS + domain setup for the showcase fleet. SOURCE it (do not run it) in every terminal:
#
#   source dds_env.sh car car1        # on a car: driving stack on loopback domain, relay bridges to Wi-Fi
#   source dds_env.sh laptop          # on the ground station: Wi-Fi domain, peers = every car that resolves
#   source dds_env.sh local car1      # desk test on ONE machine: same domains, everything on loopback
#   source dds_env.sh local laptop
#
# Addresses and domain ids come from ../config/fleet.env (or $SHOWCASE_FLEET_ENV).
# Adapted from ilc_f1tenth_hw/scripts/dds_env.sh; the XML templates are ports of llampc's cyclonedds{car,laptop}.xml.
#
# Exports: RMW_IMPLEMENTATION, CYCLONEDDS_URI, ROS_DOMAIN_ID (car: its LOCAL domain; laptop: NET domain),
#          SHOWCASE_ROLE, SHOWCASE_CAR, SHOWCASE_CARS, SHOWCASE_LOCAL_DOMAIN, SHOWCASE_NET_DOMAIN.

_sf_fail() { echo "dds_env.sh: $*" >&2; }
_sf_main() {
  local here cfg envf role car mode
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  cfg=$(cd "$here/../config" && pwd)
  envf=${SHOWCASE_FLEET_ENV:-$cfg/fleet.env}
  [ -f "$envf" ] || { _sf_fail "missing $envf"; return 1; }
  # shellcheck disable=SC1090
  source "$envf"

  mode=${1:-}; car=${2:-}
  case $mode in
    car)    role=car ;;
    laptop) role=laptop ;;
    local)  role=$([ "$car" = laptop ] && echo laptop || echo car) ;;
    *) _sf_fail "usage: source dds_env.sh car <car1|car2> | laptop | local <car1|car2|laptop>"; return 1 ;;
  esac
  if [ "$role" = car ]; then
    case $car in
      car1|car2) ;;
      *) _sf_fail "car name must be car1 or car2 (got '$car')"; return 1 ;;
    esac
  else
    car=""
  fi

  local idx local_dom net_dom
  net_dom=${NET_DOMAIN:-7}
  idx=${car#car}; idx=${idx:-1}
  local_dom=$(( ${LOCAL_DOMAIN_BASE:-42} + idx - 1 ))

  # ---- resolve ---------------------------------------------------------------------------------------------
  _sf_resolve() { [ -n "$1" ] && getent ahostsv4 "$1" 2>/dev/null | awk 'NR==1{print $1}'; }
  # `ip` first; /proc and `hostname -I` fallbacks for minimal images without iproute2
  _sf_iface_to() { ip route get "$1" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1; }
  _sf_src_to() {
    local s; s=$(ip route get "$1" 2>/dev/null | sed -n 's/.* src \([^ ]*\).*/\1/p' | head -1)
    [ -n "$s" ] || s=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo "$s"
  }
  _sf_default_iface() {
    local d; d=$(ip route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
    [ -n "$d" ] || d=$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)
    echo "$d"
  }

  local tmpl out iface peers ip h self
  local distro=${ROS_DISTRO:-$(ls /opt/ros 2>/dev/null | head -1)}
  mkdir -p ~/.ros
  if [ "$mode" = local ]; then
    tmpl=$cfg/cyclonedds_local.xml.in
    out=~/.ros/showcase_cyclonedds_local.xml
  elif [ "$role" = car ]; then
    local want_host
    want_host=$([ "$car" = car1 ] && echo "$CAR1_HOST" || echo "$CAR2_HOST")
    # CMU-DEVICE names differ from the local hostname (dl-car vs kathy-desktop): compare addresses instead
    local want_ip; want_ip=$(_sf_resolve "$want_host")
    if [ -n "$want_ip" ] && ! { ip -4 -o addr show 2>/dev/null; hostname -I 2>/dev/null; } | grep -qw "$want_ip"; then
      _sf_fail "WARNING: fleet.env says $car is $want_host ($want_ip), which is not an address of this machine. Wrong car name?"
    fi
    ip=$(_sf_resolve "$LAPTOP_HOST")
    if [ -n "$ip" ]; then
      peers="        <Peer Address=\"$ip\"/>"
      iface=$(_sf_iface_to "$ip")
    else
      _sf_fail "WARNING: LAPTOP_HOST '${LAPTOP_HOST}' does not resolve; the car still drives, but the relay has no peer."
      peers=""
    fi
    iface=${iface:-$(_sf_default_iface)}
    iface=${iface:-127.0.0.1}       # no network at all: keep the relay's network side harmless
    tmpl=$cfg/cyclonedds_car.xml.in
    out=~/.ros/showcase_cyclonedds_${car}.xml
  else
    local first=""
    for h in "$CAR1_HOST" "$CAR2_HOST"; do
      ip=$(_sf_resolve "$h")
      if [ -n "$ip" ]; then
        peers+="        <Peer Address=\"$ip\"/>"$'\n'
        first=${first:-$ip}
        echo "dds_env.sh: peer $h -> $ip"
      elif [ -n "$h" ]; then
        _sf_fail "WARNING: $h does not resolve; that car will not be reachable this session"
      fi
    done
    [ -n "$first" ] && iface=$(_sf_iface_to "$first")
    iface=${iface:-$(_sf_default_iface)}
    self=$(_sf_src_to "${first:-8.8.8.8}")
    [ -n "$self" ] && peers+="        <Peer Address=\"$self\"/>"    # this laptop's own processes
    iface=${iface:-127.0.0.1}
    tmpl=$cfg/cyclonedds_laptop.xml.in
    out=~/.ros/showcase_cyclonedds_laptop.xml
  fi
  # Interface element: Foxy ships CycloneDDS 0.7 (NetworkInterfaceAddress only); Humble's 0.10 wants
  # General/Interfaces and warns about the old element in every process.
  _sf_iface_xml() {
    if [ "$distro" = foxy ]; then echo "<NetworkInterfaceAddress>$1</NetworkInterfaceAddress>"
    elif [[ $1 =~ ^[0-9.]+$ ]]; then echo "<Interfaces><NetworkInterface address=\"$1\"/></Interfaces>"
    else echo "<Interfaces><NetworkInterface name=\"$1\"/></Interfaces>"; fi
  }
  awk -v peers="$peers" -v nx="$(_sf_iface_xml "${iface:-127.0.0.1}")" -v lx="$(_sf_iface_xml 127.0.0.1)" \
      -v ld="$local_dom" -v nd="$net_dom" '
    /@PEERS@/ { if (peers != "") print peers; next }
    { gsub(/@NET_IFACE_XML@/, nx); gsub(/@LO_IFACE_XML@/, lx); gsub(/@LOCAL_DOMAIN@/, ld); gsub(/@NET_DOMAIN@/, nd); print }' \
    "$tmpl" > "$out.$$" && mv -f "$out.$$" "$out"     # atomic: several terminals may source at once

  # ---- export ----------------------------------------------------------------------------------------------
  if [ -n "$distro" ] && [ ! -e "/opt/ros/$distro/lib/librmw_cyclonedds_cpp.so" ]; then
    _sf_fail "rmw_cyclonedds_cpp is not installed: sudo apt install ros-$distro-rmw-cyclonedds-cpp"
    return 1
  fi
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file://$out
  export SHOWCASE_ROLE=$role
  export SHOWCASE_CAR=$car
  export SHOWCASE_CARS=${SHOWCASE_CARS:-car1,car2}
  export SHOWCASE_LOCAL_DOMAIN=$local_dom
  export SHOWCASE_NET_DOMAIN=$net_dom
  if [ "$role" = car ]; then export ROS_DOMAIN_ID=$local_dom; else export ROS_DOMAIN_ID=$net_dom; fi
  unset ROS_LOCALHOST_ONLY
  echo "DDS: $role${car:+ $car} | ROS_DOMAIN_ID=$ROS_DOMAIN_ID (local $local_dom, net $net_dom) | iface ${iface:-lo} | $out"
  ros2 daemon stop >/dev/null 2>&1 || true    # a running daemon keeps the old domain / discovery settings
}
_sf_main "$@"
_sf_rc=$?
unset -f _sf_main _sf_fail _sf_resolve _sf_iface_to _sf_src_to _sf_default_iface _sf_iface_xml
return $_sf_rc 2>/dev/null || exit $_sf_rc
