from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Πακέτο που περιέχει τα Gazebo launch files του TurtleBot3
    tb3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')

    # Το world file σου
    world_file = os.path.join(
        get_package_share_directory('tb3_sim_bringup'),
        'worlds',
        'lab_world.sdf'
    )

    # Ορισμός TurtleBot3 μοντέλου (burger)
    os.environ['TURTLEBOT3_MODEL'] = 'burger'

    tb3_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_pkg, 'launch', 'turtlebot3_world.launch.py')
        ),
        launch_arguments={
            'world': world_file
        }.items()
    )

    return LaunchDescription([
        tb3_world_launch
    ])
