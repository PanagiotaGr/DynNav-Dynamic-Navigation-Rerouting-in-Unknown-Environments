#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Quaternion


def rot_to_quaternion(R: np.ndarray) -> Quaternion:
    """Μετατροπή 3x3 rotation matrix σε quaternion."""
    q = Quaternion()
    tr = np.trace(R)

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        q.w = 0.25 * S
        q.x = (R[2, 1] - R[1, 2]) / S
        q.y = (R[0, 2] - R[2, 0]) / S
        q.z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q.w = (R[2, 1] - R[1, 2]) / S
            q.x = 0.25 * S
            q.y = (R[0, 1] + R[1, 0]) / S
            q.z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q.w = (R[0, 2] - R[2, 0]) / S
            q.x = (R[0, 1] + R[1, 0]) / S
            q.y = 0.25 * S
            q.z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q.w = (R[1, 0] - R[0, 1]) / S
            q.x = (R[0, 2] + R[2, 0]) / S
            q.y = (R[1, 2] + R[2, 1]) / S
            q.z = 0.25 * S

    return q


class FKNode(Node):
    def __init__(self):
        super().__init__('fk_node')

        # Παράμετροι DH: [a, alpha, d, theta_offset]
        dh_default = [
            [0.0,  math.pi/2, 0.1, 0.0],
            [0.2,  0.0,       0.0, 0.0],
            [0.2,  0.0,       0.0, 0.0],
            [0.0,  math.pi/2, 0.1, 0.0],
            [0.0, -math.pi/2, 0.0, 0.0],
            [0.0,  0.0,       0.1, 0.0],
        ]
        self.dh_params = np.array(dh_default, dtype=float)

        self.joint_names = [
            self.declare_parameter('joint1_name', 'joint1').value,
            self.declare_parameter('joint2_name', 'joint2').value,
            self.declare_parameter('joint3_name', 'joint3').value,
            self.declare_parameter('joint4_name', 'joint4').value,
            self.declare_parameter('joint5_name', 'joint5').value,
            self.declare_parameter('joint6_name', 'joint6').value,
        ]

        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            'ee_pose',
            10
        )

        self.get_logger().info('FKNode for 6-DOF manipulator started.')

    def dh_transform(self, a, alpha, d, theta):
        """Υπολογισμός T_i^{i+1} από DH παραμέτρους."""
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        ct = math.cos(theta)
        st = math.sin(theta)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,    sa,       ca,      d ],
            [0.0,   0.0,      0.0,    1.0 ],
        ])

    def forward_kinematics(self, thetas: np.ndarray) -> np.ndarray:
        """Επιστρέφει τον 4x4 πίνακα T_0^6."""
        T = np.eye(4)
        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = thetas[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
        return T

    def joint_state_callback(self, msg: JointState):
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}

        thetas = []
        for jn in self.joint_names:
            if jn not in name_to_pos:
                self.get_logger().warn(f'Missing joint {jn} in JointState')
                return
            thetas.append(name_to_pos[jn])

        thetas = np.array(thetas, dtype=float)

        T = self.forward_kinematics(thetas)
        pos = T[:3, 3]
        R = T[:3, :3]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'

        pose_msg.pose.position.x = float(pos[0])
        pose_msg.pose.position.y = float(pos[1])
        pose_msg.pose.position.z = float(pos[2])

        pose_msg.pose.orientation = rot_to_quaternion(R)

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
