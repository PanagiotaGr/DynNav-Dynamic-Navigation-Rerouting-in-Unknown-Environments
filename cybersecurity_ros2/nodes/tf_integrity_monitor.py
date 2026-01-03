#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from tf2_ros import Buffer, TransformListener


def quat_to_yaw(q):
    # yaw from quaternion (z-rotation)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y*q.y + q.z*q.z))


def angle_wrap(a):
    # wrap to [-pi, pi]
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class TFIntegrityMonitor(Node):
    """
    Checks TF continuity between parent->child and raises an alarm if motion is physically implausible.

    Detection signals:
      - linear speed bound: |dp|/dt > v_max
      - angular speed bound: |dyaw|/dt > w_max
      - optional: timestamp monotonicity and dt sanity

    Publishes:
      /ids/tf_alarm (Bool)
      /ids/tf_score (Float32)  score = max(v_ratio, w_ratio) where ratio >1 indicates violation
    """

    def __init__(self):
        super().__init__("tf_integrity_monitor")

        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_link_spoofed")
        self.declare_parameter("check_rate_hz", 30.0)

        self.declare_parameter("v_max", 1.5)   # m/s
        self.declare_parameter("w_max", 1.2)   # rad/s
        self.declare_parameter("dt_min", 1e-3)
        self.declare_parameter("alarm_hold_steps", 30)  # latch alarm

        self.parent = self.get_parameter("parent_frame").value
        self.child = self.get_parameter("child_frame").value
        self.rate_hz = float(self.get_parameter("check_rate_hz").value)

        self.v_max = float(self.get_parameter("v_max").value)
        self.w_max = float(self.get_parameter("w_max").value)
        self.dt_min = float(self.get_parameter("dt_min").value)
        self.hold_steps = int(self.get_parameter("alarm_hold_steps").value)

        self.pub_alarm = self.create_publisher(Bool, "/ids/tf_alarm", 10)
        self.pub_score = self.create_publisher(Float32, "/ids/tf_score", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.prev_t = None
        self.prev_x = None
        self.prev_y = None
        self.prev_yaw = None

        self.alarm_countdown = 0

        period = 1.0 / max(self.rate_hz, 1e-6)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"Monitoring TF {self.parent} -> {self.child}")

    def tick(self):
        try:
            tr = self.tf_buffer.lookup_transform(self.parent, self.child, rclpy.time.Time())
        except Exception:
            # TF might not be available yet
            return

        stamp = tr.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        x = tr.transform.translation.x
        y = tr.transform.translation.y
        yaw = quat_to_yaw(tr.transform.rotation)

        # First sample
        if self.prev_t is None:
            self.prev_t, self.prev_x, self.prev_y, self.prev_yaw = t, x, y, yaw
            return

        dt = t - self.prev_t
        if dt < self.dt_min:
            # too small/invalid dt; skip update
            return

        dx = x - self.prev_x
        dy = y - self.prev_y
        dyaw = angle_wrap(yaw - self.prev_yaw)

        v = math.sqrt(dx*dx + dy*dy) / dt
        w = abs(dyaw) / dt

        v_ratio = v / max(self.v_max, 1e-6)
        w_ratio = w / max(self.w_max, 1e-6)
        score = max(v_ratio, w_ratio)

        violated = (v_ratio > 1.0) or (w_ratio > 1.0)

        if violated:
            self.alarm_countdown = self.hold_steps
        else:
            if self.alarm_countdown > 0:
                self.alarm_countdown -= 1

        alarm = self.alarm_countdown > 0

        self.pub_alarm.publish(Bool(data=bool(alarm)))
        self.pub_score.publish(Float32(data=float(score)))

        # update prev
        self.prev_t, self.prev_x, self.prev_y, self.prev_yaw = t, x, y, yaw


def main():
    rclpy.init()
    node = TFIntegrityMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
