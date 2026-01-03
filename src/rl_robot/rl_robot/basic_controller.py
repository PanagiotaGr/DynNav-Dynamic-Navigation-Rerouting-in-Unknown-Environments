import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan


class BasicController(Node):
    def __init__(self):
        super().__init__('basic_controller')

        # Publisher για εντολές κίνησης (TwistStamped, επειδή το θέλει το ros_gz_bridge)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Subscriber για το LIDAR
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer: εκτελεί τη control_loop κάθε 0.1s (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Τελευταίο LaserScan
        self.last_scan = None

    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg

    def control_loop(self):
        # Μήνυμα κίνησης
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_footprint'  # ή 'base_link'

        if self.last_scan is None:
            # Αν δεν έχουμε ακόμα scan, σταμάτα
            self.get_logger().info('No scan data yet')
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist_msg)
            return

        ranges = self.last_scan.ranges
        n = len(ranges)

        # Θέλουμε να κοιτάξουμε μπροστά ±30°
        angle_range_deg = 30
        angle_increment = self.last_scan.angle_increment  # rad
        angle_range_rad = math.radians(angle_range_deg)

        # index που αντιστοιχεί περίπου στο "μπροστά"
        center_index = int((-self.last_scan.angle_min) / angle_increment)
        half_span = int(angle_range_rad / angle_increment)

        start_idx = max(0, center_index - half_span)
        end_idx = min(n - 1, center_index + half_span)

        front_ranges = [
            r for r in ranges[start_idx:end_idx]
            if not math.isinf(r) and not math.isnan(r)
        ]

        if not front_ranges:
            min_front = float('inf')
        else:
            min_front = min(front_ranges)

        safe_distance = 0.5  # μέτρα

        if min_front < safe_distance:
            # Αν έχει εμπόδιο μπροστά κοντά → γύρνα
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.angular.z = 0.5
            self.get_logger().info(f'Obstacle at {min_front:.2f} m -> Turning')
        else:
            # Αλλιώς πήγαινε ευθεία
            twist_msg.twist.linear.x = 0.15
            twist_msg.twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BasicController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

