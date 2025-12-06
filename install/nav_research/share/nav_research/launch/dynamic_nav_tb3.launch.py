from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Πακέτο προσομοίωσης TurtleBot3
    tb3_sim_pkg = "nav2_minimal_tb3_sim"

    # Πακέτο του δικού μας node
    dynamic_nav_pkg = "dynamic_nav"

    # Path στο launch του turtlebot3
    tb3_spawn_launch = PathJoinSubstitution(
        [FindPackageShare(tb3_sim_pkg), "launch", "spawn_tb3.launch.py"]
    )

    # Path στο YAML με τις παραμέτρους
    dynamic_nav_params = PathJoinSubstitution(
        [FindPackageShare(dynamic_nav_pkg), "config", "dynamic_nav_params.yaml"]
    )

    # Launch περιγραφή
    return LaunchDescription(
        [
            # 1) Προσομοίωση TurtleBot3
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(tb3_spawn_launch)
            ),

            # 2) Δικός μας κόμβος δυναμικής πλοήγησης
            Node(
                package="dynamic_nav",
                executable="dynamic_nav_node",
                name="dynamic_nav_node",
                output="screen",
                parameters=[dynamic_nav_params],
            ),
        ]
    )
