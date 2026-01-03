#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class DummyJointPublisher(Node):
    def __init__(self):
        super().__init__('dummy_joint_pub')
        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz
        self.t = 0.0

        self.joint_names = [
            'joint1', 'joint2', 'joint3',
            'joint4', 'joint5', 'joint6'
        ]

        self.get_logger().info('DummyJointPublisher started.')

    def timer_callback(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names

        # Παράδειγμα κινηματικής πορείας
        theta1 = 0.5 * math.sin(self.t)
        theta2 = 0.5 * math.cos(self.t)
        theta3 = 0.3 * math.sin(0.5 * self.t)
        theta4 = 0.4 * math.sin(0.8 * self.t)
        theta5 = 0.3 * math.cos(0.7 * self.t)
        theta6 = 0.2 * math.sin(1.2 * self.t)

        js.position = [theta1, theta2, theta3, theta4, theta5, theta6]

        self.pub.publish(js)
        self.t += 0.02


def main(args=None):
    rclpy.init(args=args)
    node = DummyJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
