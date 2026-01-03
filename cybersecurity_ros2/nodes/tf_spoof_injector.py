#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float):
    # z-rotation quaternion
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class TFSpoofInjector(Node):
    """
    Publishes a spoofed TF transform parent->child with controlled attack patterns.
    For evaluation only (simulation / controlled tests).

    Default: odom -> base_link
    Attack modes:
      - none: publish identity (or pass-through style if you adapt later)
      - drift: slow bias accumulating in x/y and yaw
      - step: sudden jump at attack_start_sec
    """

    def __init__(self):
        super().__init__("tf_spoof_injector")
        self.br = TransformBroadcaster(self)

        # Parameters
        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_link_spoofed")
        self.declare_parameter("rate_hz", 30.0)

        self.declare_parameter("mode", "drift")  # none | drift | step
        self.declare_parameter("attack_start_sec", 8.0)

        # Drift params (per second)
        self.declare_parameter("drift_vx", 0.02)     # m/s
        self.declare_parameter("drift_vy", -0.015)   # m/s
        self.declare_parameter("drift_wz", 0.01)     # rad/s

        # Step params
        self.declare_parameter("step_dx", 0.8)       # meters
        self.declare_parameter("step_dy", -0.4)
        self.declare_parameter("step_dyaw", 0.35)    # rad

        self.parent = self.get_parameter("parent_frame").value
        self.child = self.get_parameter("child_frame").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.mode = str(self.get_parameter("mode").value)
        self.attack_start = float(self.get_parameter("attack_start_sec").value)

        self.drift_vx = float(self.get_parameter("drift_vx").value)
        self.drift_vy = float(self.get_parameter("drift_vy").value)
        self.drift_wz = float(self.get_parameter("drift_wz").value)

        self.step_dx = float(self.get_parameter("step_dx").value)
        self.step_dy = float(self.get_parameter("step_dy").value)
        self.step_dyaw = float(self.get_parameter("step_dyaw").value)

        self.t0 = self.get_clock().now()
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.step_applied = False

        period = 1.0 / max(self.rate_hz, 1e-6)
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(
            f"TF Spoof Injector publishing {self.parent} -> {self.child} | mode={self.mode}"
        )

    def tick(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        if self.mode == "none":
            # identity transform
            self.x, self.y, self.yaw = 0.0, 0.0, 0.0

        elif self.mode == "drift":
            if t >= self.attack_start:
                dt = 1.0 / self.rate_hz
                self.x += self.drift_vx * dt
                self.y += self.drift_vy * dt
                self.yaw += self.drift_wz * dt

        elif self.mode == "step":
            if (t >= self.attack_start) and (not self.step_applied):
                self.x += self.step_dx
                self.y += self.step_dy
                self.yaw += self.step_dyaw
                self.step_applied = True

        # Publish TF
        msg = TransformStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.parent
        msg.child_frame_id = self.child

        msg.transform.translation.x = float(self.x)
        msg.transform.translation.y = float(self.y)
        msg.transform.translation.z = 0.0

        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw

        self.br.sendTransform(msg)


def main():
    rclpy.init()
    node = TFSpoofInjector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
