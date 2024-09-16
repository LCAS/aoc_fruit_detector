import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_prefix

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

    # Run the Python script with the -O optimization flag
    fruit_detection_node = ExecuteProcess(
        cmd=['python3', '-O', fruit_detection_script_installed,
             '--ros-args',
            '--remap', '/camera/image_raw:=/front_camera/image_raw',
            '--remap', '/camera/depth:=/front_camera/depth'
            ],
        output='screen'
    )

    return LaunchDescription([
        fruit_detection_node
    ])
