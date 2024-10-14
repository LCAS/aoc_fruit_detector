import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from ament_index_python.packages import get_package_prefix
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get the installation path of the package
    package_prefix = get_package_prefix('aoc_fruit_detector')

    # Path to the fruit_detection.py script in the installed directory
    fruit_detection_script_installed = os.path.join(
        package_prefix,
        'lib',
        'aoc_fruit_detector',
        'fruit_detection.py'
    )

    declare_constant_depth_value = DeclareLaunchArgument(
        'constant_depth_value',
        default_value='1.0',  # Dummy depth value
        description='Constant depth value to use when depth channel is unavailable'
    )

    # Run the Python script with the -O optimization flag
    fruit_detection_node = ExecuteProcess(
        cmd=['python3', '-O', fruit_detection_script_installed,
            '--ros-args',
            '--param', ['constant_depth_value:=', LaunchConfiguration('constant_depth_value')],
            '--remap', '/camera/image_raw:=/flir_camera/image_raw', # /zed/zed_node/rgb_raw/image_raw_color or /front_camera/image_raw
            '--remap', '/camera/depth:=/zed/zed_node/depth/depth_registered', # /zed/zed_node/depth/depth_registered or /front_camera/depth
            '--remap', '/camera/camera_info:=/flir_camera/camera_info'
            ],
        output='screen'
    )

    return LaunchDescription([
        declare_constant_depth_value,
        fruit_detection_node
    ])
