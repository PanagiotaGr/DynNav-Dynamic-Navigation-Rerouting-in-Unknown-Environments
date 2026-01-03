from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')

    # === Δείχνει στο δικό σου YAML ===
    params_file = LaunchConfiguration(
        'params_file',
        default=os.path.join(
            get_package_share_directory('nav_research'),
            'config',
            'nav2_params.yaml'
        )
    )

    # === SLAM Toolbox ===
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            )
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # === Nav2 bringup ===
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav2_bringup'),
                'launch',
                'bringup_launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': 'True',
            'params_file': params_file
        }.items()
    )

    # === RViz2 για οπτικοποίηση ===
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav2_bringup'),
                'launch',
                'rviz_launch.py'
            )
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    return LaunchDescription([slam, nav2, rviz])
