#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


class IKDLSNode(Node):
    def __init__(self):
        super().__init__('ik_dls_node')

        # Ιδιο DH με το FK node
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

        # αρχική εικασία για τις γωνίες
        self.theta_init = np.zeros(6)

        # Damping factor για DLS
        self.lambda_dls = self.declare_parameter('lambda_dls', 0.05).value

        # Publisher για joint commands
        self.joint_pub = self.create_publisher(JointState, 'joint_cmds', 10)

        # Συνδρομή στο target pose
        self.target_sub = self.create_subscription(
            PoseStamped,
            'ee_target',
            self.target_callback,
            10
        )

        self.get_logger().info('IKDLSNode started (position-only IK).')

    def dh_transform(self, a, alpha, d, theta):
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
        T = np.eye(4)
        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = thetas[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
        return T

    def position_from_thetas(self, thetas: np.ndarray) -> np.ndarray:
        T = self.forward_kinematics(thetas)
        return T[:3, 3]

    def numerical_jacobian(self, thetas: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Αριθμητικός Jacobian για τη θέση x,y,z ως προς τις γωνίες θ.
        J είναι 3x6.
        """
        J = np.zeros((3, 6))
        f0 = self.position_from_thetas(thetas)
        for i in range(6):
            dtheta = np.zeros(6)
            dtheta[i] = eps
            f1 = self.position_from_thetas(thetas + dtheta)
            J[:, i] = (f1 - f0) / eps
        return J

    def damped_least_squares_step(self, thetas: np.ndarray, x_d: np.ndarray) -> np.ndarray:
        """
        Ένα βήμα DLS: θ_{k+1} = θ_k + Δθ
        """
        x = self.position_from_thetas(thetas)
        e = x_d - x  # 3x1

        J = self.numerical_jacobian(thetas)  # 3x6

        lam = self.lambda_dls
        # Δθ = J^T (J J^T + λ^2 I)^(-1) e
        JJt = J @ J.T
        I3 = np.eye(3)
        inv_term = np.linalg.inv(JJt + (lam ** 2) * I3)
        delta_theta = J.T @ (inv_term @ e)

        return thetas + delta_theta

    def solve_ik(self, x_d: np.ndarray, max_iters: int = 200, tol: float = 1e-4) -> np.ndarray:
        thetas = self.theta_init.copy()
        for k in range(max_iters):
            x = self.position_from_thetas(thetas)
            e = x_d - x
            err_norm = np.linalg.norm(e)
            if err_norm < tol:
                self.get_logger().info(f'IK converged in {k} iterations, error={err_norm:.6f}')
                return thetas
            thetas = self.damped_least_squares_step(thetas, x_d)
        self.get_logger().warn(f'IK did not fully converge (final error={err_norm:.6f})')
        return thetas

    def target_callback(self, msg: PoseStamped):
        # επιθυμητή θέση από το μήνυμα
        x_d = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)

        thetas_sol = self.solve_ik(x_d)

        # Αποθηκεύουμε λύση ως νέα "initial guess" για επόμενο στόχο
        self.theta_init = thetas_sol.copy()

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = thetas_sol.tolist()

        self.joint_pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = IKDLSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
