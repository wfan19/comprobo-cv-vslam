cmake_minimum_required(VERSION 3.8)
project(symforce_vslam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(vslam_node src/vslam_node.cpp)
ament_target_dependencies(vslam_node rclcpp)

install(
  TARGETS vslam_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch cameras config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  symforce_vslam/main.py
  symforce_vslam/slam_offline.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_python_install_package(symforce_vslam)


ament_package()
