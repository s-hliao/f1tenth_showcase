cmake_minimum_required(VERSION 3.8)
project(frenet_baseline)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python module: controller, raceline, barrier state (frenet_baseline/baseline_pp.py)
ament_python_install_package(${PROJECT_NAME})

# Executables: ros2 run frenet_baseline <name>
install(PROGRAMS
  scripts/baseline_pp_node.py
  scripts/telemetry_report.py
  scripts/grip_from_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

# Params, launch files, racelines, reference map
install(DIRECTORY
  config
  launch
  waypoints
  maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
