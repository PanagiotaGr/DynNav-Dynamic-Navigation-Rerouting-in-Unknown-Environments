#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from tf2_ros import Buffer, TransformListener


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y*q.y + q.z*q.z))


def angle_wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class CUSUM:
    def __init__(self, k: float, h: float):
        self.k = float(k)
        self.h = float(h)
        self.g = 0.0

    def reset(self):
        self.g = 0.0

    def update(self, x: float) -> float:
        # one-sided
        self.g = max(0.0, self.g + (float(x) - self.k))
        return self.g

    def alarm(self) -> bool:
        return self.g >= self.h


class TFIntegrityMonitorCUSUM(Node):
    """
    Stealth-drift TF IDS:
      score_t = max(v_ratio, w_ratio)
      cusum_t = max(0, cusum_{t-1} + (score_t - k))
      alarm if cusum_t >= h

    score_t is normalized, so typical nominal motion should produce score < 1.
    For stealth attacks: score may stay < 1 but slightly elevated -> CUSUM accumulates.
    """

    def __init__(self):
        super().__init__("tf_integrity_monitor_cusum")

        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_link_spoofed")
        self.declare_parameter("check_rate_hz", 30.0)

        # bounds used to build score ratios
        self.declare_parameter("v_max", 0.25)   # m/s
        self.declare_parameter("w_max", 0.35)   # rad/s
        self.declare_parameter("dt_min", 1e-3)

        # CUSUM parameters (tune these)
        self.declare_parameter("cusum_k", 0.08)
        self.declare_parameter("cusum_h", 2.5)

        # latch alarm for a bit (optional)
        self.declare_parameter("alarm_hold_steps", 30)

        self.parent = self.get_parameter("parent_frame").value
        self.child = self.get_parameter("child_frame").value
        self.rate_hz = float(self.get_parameter("check_rate_hz").value)

        self.v_max = float(self.get_parameter("v_max").value)
        self.w_max = float(self.get_parameter("w_max").value)
        self.dt_min = float(self.get_parameter("dt_min").value)

        k = float(self.get_parameter("cusum_k").value)
        h = float(self.get_parameter("cusum_h").value)
        self.cusum = CUSUM(k=k, h=h)

        self.hold_steps = int(self.get_parameter("alarm_hold_steps").value)
        self.alarm_countdown = 0

        self.pub_alarm = self.create_publisher(Bool, "/ids/tf_alarm", 10)
        self.pub_score = self.create_publisher(Float32, "/ids/tf_score", 10)
        self.pub_cusum = self.create_publisher(Float32, "/ids/tf_cusum", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.prev_t = None
        self.prev_x = None
        self.prev_y = None
        self.prev_yaw = None

        period = 1.0 / max(self.rate_hz, 1e-6)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"Monitoring TF {self.parent}->{self.child} | "
            f"score bounds v_max={self.v_max}, w_max={self.w_max} | "
            f"CUSUM(k={k}, h={h})"
        )

    def tick(self):
        try:
            tr = self.tf_buffer.lookup_transform(self.parent, self.child, rclpy.time.Time())
        except Exception:
            return

        stamp = tr.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        x = tr.transform.translation.x
        y = tr.transform.translation.y
        yaw = quat_to_yaw(tr.transform.rotation)

        if self.prev_t is None:
            self.prev_t, self.prev_x, self.prev_y, self.prev_yaw = t, x, y, yaw
            return

        dt = t - self.prev_t
        if dt < self.dt_min:
            return

        dx = x - self.prev_x
        dy = y - self.prev_y
        dyaw = angle_wrap(yaw - self.prev_yaw)

        v = math.sqrt(dx*dx + dy*dy) / dt
        w = abs(dyaw) / dt

        v_ratio = v / max(self.v_max, 1e-6)
        w_ratio = w / max(self.w_max, 1e-6)
        score = max(v_ratio, w_ratio)

        g = self.cusum.update(score)

        if self.cusum.alarm():
            self.alarm_countdown = self.hold_steps
        else:
            if self.alarm_countdown > 0:
                self.alarm_countdown -= 1

        alarm = self.alarm_countdown > 0

        self.pub_score.publish(Float32(data=float(score)))
        self.pub_cusum.publish(Float32(data=float(g)))
        self.pub_alarm.publish(Bool(data=bool(alarm)))

        self.prev_t, self.prev_x, self.prev_y, self.prev_yaw = t, x, y, yaw


def main():
    rclpy.init()
    node = TFIntegrityMonitorCUSUM()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
