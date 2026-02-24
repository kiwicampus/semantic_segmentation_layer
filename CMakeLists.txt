cmake_minimum_required(VERSION 3.8)
project(semantic_segmentation_layer)

# Default to C++17
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_ros_common REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

# Include directories
include_directories(include)

# Create the semantic segmentation layer library
add_library(semantic_segmentation_layer SHARED
  src/semantic_segmentation_layer.cpp
  src/segmentation_buffer.cpp
)

target_include_directories(semantic_segmentation_layer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(semantic_segmentation_layer
  PUBLIC
    message_filters::message_filters
    nav2_costmap_2d::nav2_costmap_2d_core
    nav2_util::nav2_util_core
    nav2_ros_common::nav2_ros_common
    pluginlib::pluginlib
    rclcpp::rclcpp
    rclcpp_lifecycle::rclcpp_lifecycle
    tf2::tf2
    tf2_ros::tf2_ros
    tf2_sensor_msgs::tf2_sensor_msgs
    ${vision_msgs_TARGETS}
)

# Install library
install(TARGETS semantic_segmentation_layer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Install plugin description file
install(FILES costmap_plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

# Export plugin description file for pluginlib
pluginlib_export_plugin_description_file(nav2_costmap_2d costmap_plugins.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

