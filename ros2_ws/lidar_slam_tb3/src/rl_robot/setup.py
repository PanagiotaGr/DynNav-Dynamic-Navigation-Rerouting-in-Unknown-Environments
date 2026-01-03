from setuptools import setup

package_name = 'rl_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='panagiotagrosd',
    maintainer_email='iitsbooklover@gmail.com',
    description='RL experiments with TurtleBot3 in Gazebo',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'basic_controller = rl_robot.basic_controller:main',
            'rl_agent = rl_robot.rl_agent:main',
        'fsm_controller = rl_robot.fsm_controller:main',
        'benchmark = rl_robot.benchmark_node:main',

'astar_planner = rl_robot.astar_planner:main',
        ],
    },
)

