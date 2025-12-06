import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import numpy as np

from .calibration_state import CalibrationState
from .calibration_ekf import CalibrationEKF
from .active_calibration_planner import ActiveCalibrationPlanner
from .sensor_models import vo_scale_residual


def pose_to_xy(pose):
    """Helper: πάρε x,y από geometry_msgs/Pose."""
    return pose.position.x, pose.position.y


def distance(p1, p2):
    """Απόσταση 2D μεταξύ δύο (x,y)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class ActiveCalibrationNode(Node):
    def __init__(self):
        super().__init__('active_calibration_node')

        # --- Calibration core ---
        self.calib_state = CalibrationState()
        self.ekf = CalibrationEKF(self.calib_state)
        self.planner = ActiveCalibrationPlanner()

        # --- Publisher κίνησης ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- Subscribers ---

        # ΠΡΟΣ ΤΟ ΠΑΡΟΝ: Χρησιμοποιούμε /odom και σαν VO και σαν ground-truth,
        # απλά για να δούμε ότι δουλεύουν τα callbacks και το EKF.
        self.vo_sub = self.create_subscription(
            Odometry,
            '/odom',            # "VO" odometry
            self.vo_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',            # ground-truth / wheel / fused odom
            self.odom_callback,
            10
        )

        # --- Internal state for incremental distances ---
        self.last_vo_pose = None       # τελευταίο Pose από VO
        self.prev_vo_pose = None       # προηγούμενο Pose από VO

        self.last_odom_pose = None     # τελευταίο Pose από odom
        self.prev_odom_pose = None     # προηγούμενο Pose από odom

        # Timer για κίνηση (active planner)
        self.motion_timer = self.create_timer(0.5, self.motion_timer_callback)

        self.get_logger().info("Active calibration node started.")

    # =====================
    #   MOTION PLANNING
    # =====================
    def motion_timer_callback(self):
        """
        Προς το παρόν κρατάμε τον απλό active planner:
        - dummy H για κάθε motion primitive
        - επιλέγει το κίνημα με μέγιστη observability
        """
        H_candidates = [
            np.eye(3),           # primitive 1
            np.eye(3) * 0.5,     # primitive 2
            np.eye(3) * 0.2,     # primitive 3
            np.eye(3) * 2.0,     # primitive 4
        ]

        motion, score = self.planner.choose_best_motion(H_candidates)

        msg = Twist()
        msg.linear.x = float(motion[0])
        msg.linear.y = float(motion[1])
        msg.angular.z = float(motion[2])

        self.cmd_pub.publish(msg)
        self.get_logger().info(
            f"Sent motion {motion}, observability score={score:.3f}"
        )

    # =====================
    #   ODOMETRY CALLBACKS
    # =====================
    def vo_callback(self, msg: Odometry):
        """
        Callback για "VO" odometry (προς το παρόν /odom).
        """
        pose = msg.pose.pose

        self.prev_vo_pose = self.last_vo_pose
        self.last_vo_pose = pose

        self.get_logger().info("VO callback received.")
        self.try_vo_scale_update()

    def odom_callback(self, msg: Odometry):
        """
        Callback για ground-truth / fused odom.
        """
        pose = msg.pose.pose

        self.prev_odom_pose = self.last_odom_pose
        self.last_odom_pose = pose

        self.get_logger().info("Odom callback received.")
        self.try_vo_scale_update()

    # =====================
    #   VO SCALE EKF UPDATE
    # =====================
    def try_vo_scale_update(self):
        """
        Όταν έχουμε ΚΑΙ προηγούμενη και τωρινή VO pose
        ΚΑΙ προηγούμενη και τωρινή odom pose,
        υπολογίζουμε τις incremental αποστάσεις και κάνουμε EKF update.
        """
        if (self.prev_vo_pose is None or self.last_vo_pose is None or
                self.prev_odom_pose is None or self.last_odom_pose is None):
            # δεν έχουμε αρκετά δεδομένα ακόμα
            self.get_logger().info(
                "try_vo_scale_update: waiting for enough poses "
                f"(prev_vo={self.prev_vo_pose is not None}, "
                f"last_vo={self.last_vo_pose is not None}, "
                f"prev_odom={self.prev_odom_pose is not None}, "
                f"last_odom={self.last_odom_pose is not None})"
            )
            return

        vo_prev_xy = pose_to_xy(self.prev_vo_pose)
        vo_last_xy = pose_to_xy(self.last_vo_pose)
        d_vo = distance(vo_prev_xy, vo_last_xy)

        odom_prev_xy = pose_to_xy(self.prev_odom_pose)
        odom_last_xy = pose_to_xy(self.last_odom_pose)
        d_odom = distance(odom_prev_xy, odom_last_xy)

        if d_vo < 1e-4 and d_odom < 1e-4:
            self.get_logger().info(
                f"try_vo_scale_update: small motion, "
                f"d_vo={d_vo:.6f}, d_odom={d_odom:.6f} -> skip"
            )
            return

        residual_value = vo_scale_residual(d_vo, d_odom, self.calib_state)
        residual = np.array([residual_value])  # shape (1,)

        H = np.zeros((1, 13))
        H[0, 12] = d_vo

        R = np.array([[0.05 ** 2]])

        self.ekf.update(residual, H, R)

        vo_scale_est = self.calib_state.x[12]
        self.get_logger().info(
            f"VO scale update: d_vo={d_vo:.4f}, d_odom={d_odom:.4f}, "
            f"residual={residual_value:.4f}, s_est={vo_scale_est:.4f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ActiveCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
