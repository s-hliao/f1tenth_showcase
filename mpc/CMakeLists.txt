cmake_minimum_required(VERSION 3.5)
project(mpc)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# Fetch OSQP
include(FetchContent)
FetchContent_Declare(
  osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG        v1.0.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(osqp)

# Find ROS dependencies
foreach(PACKAGE
  ament_cmake
  ament_cmake_python
  rclcpp
  rclpy
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  tf2_ros)
  find_package(${PACKAGE} REQUIRED)
endforeach()

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIRS})

# C++ executable
add_executable(mpc_node src/mpc_node.cpp)

ament_target_dependencies(mpc_node
  rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2_ros)

target_link_libraries(mpc_node osqpstatic)

# Install C++ executables
install(TARGETS mpc_node DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/mpc_node.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
