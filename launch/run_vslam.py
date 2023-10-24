from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import numpy as np

import os

def generate_launch_description():
    apriltag_config = os.path.join(get_package_share_directory('symforce_vslam'), 'config', "apriltag_detector.yaml")

    return LaunchDescription([
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_detector',
            parameters=[
                apriltag_config,
            ],
            remappings=[
                ('image_rect', '/camera/image_raw'),
                ('camera_info', '/camera/camera_info'),
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera',
            arguments=["-0.1016", "0", "0.0889", str(-np.pi/2), "0", str(-np.pi/2), "base_link", "camera"]
        ),
    ])