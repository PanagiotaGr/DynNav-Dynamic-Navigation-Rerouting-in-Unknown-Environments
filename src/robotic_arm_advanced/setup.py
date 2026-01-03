from setuptools import setup

package_name = 'robotic_arm_advanced'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Advanced kinematics (FK + IK) for a 6-DOF manipulator in ROS 2',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fk_node = robotic_arm_advanced.fk_node:main',
            'ik_dls_node = robotic_arm_advanced.ik_dls_node:main',
            'dummy_joint_pub = robotic_arm_advanced.dummy_joint_pub:main',
        ],
    },
)
