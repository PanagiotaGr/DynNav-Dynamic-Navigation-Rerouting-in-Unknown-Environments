import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from .active_calibration_planner import ActiveCalibrationPlanner
from .calibration_state import CalibrationState
from .calibration_ekf import CalibrationEKF

class CalibrationROS(Node):
    def __init__(self, planner):
        super().__init__('active_calibration_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.planner = planner

    def send_motion_command(self, motion):
        msg = Twist()
        msg.linear.x = float(motion[0])
        msg.linear.y = float(motion[1])
        msg.angular.z = float(motion[2])
        self.cmd_pub.publish(msg)
