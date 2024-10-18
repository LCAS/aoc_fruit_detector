cmake_minimum_required(VERSION 3.8)
project(aoc_fruit_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(std_msgs REQUIRED)

find_package(geometry_msgs REQUIRED)

find_package(sensor_msgs REQUIRED)

find_package(visualization_msgs REQUIRED)

find_package(image_geometry REQUIRED)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FruitInfoMessage.msg"
  "msg/FruitInfoArray.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs visualization_msgs image_geometry
)

# Install Python scripts
install(PROGRAMS
  scripts/fruit_detection.py
  DESTINATION lib/${PROJECT_NAME}
)

foreach (dir launch config)
    install(DIRECTORY ${dir} DESTINATION share/${PROJECT_NAME})
endforeach(dir)

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

ament_package()
