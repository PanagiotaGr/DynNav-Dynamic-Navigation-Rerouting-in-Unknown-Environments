from setuptools import setup
import os
from glob import glob

package_name = 'tb3_sim_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ΕΔΩ: install τα launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # ΕΔΩ: install τα world files
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='panagiota',
    maintainer_email='your_email@example.com',
    description='TurtleBot3 simulation bringup with Gazebo',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)

