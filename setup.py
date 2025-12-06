from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'nav_research'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Research bringup (Nav2 + SLAM + RViz) for dynamic re-routing',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cpp_node = cpp_extension.cpp_node:main',
            'rl_rrt_planner = nav_research.rl_rrt_planner_node:main',
            'active_calib_node = nav_research.calib_active.calibration_node:main',
        ],
    },
)
