import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/panagiotagrosd/ros2_ws/install/robotic_arm_advanced'
