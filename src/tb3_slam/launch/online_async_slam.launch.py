from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Το δικό σου params αρχείο
    params_file = os.path.join(
        get_package_share_directory('tb3_slam'),
        'config',
        'slam_toolbox_params.yaml'
    )

    # Το official launch του slam_toolbox (έχει lifecycle manager κλπ.)
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'slam_params_file': params_file
        }.items()
    )

    return LaunchDescription([
        slam_launch
    ])

