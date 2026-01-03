from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tb3_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # εδώ προσθέτουμε τα launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # εδώ προσθέτουμε τα config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='panagiotagrosd',
    maintainer_email='panagiotagrosd@todo.todo',
    description='SLAM Toolbox integration for TurtleBot3 in Gazebo',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)

