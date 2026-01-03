import csv
import time
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class BenchmarkNode(Node):
    def __init__(self):
        super().__init__('benchmark_node')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.start_time = time.time()
        self.min_clearance = float('inf')
        self.cmd_count = 0
        self.current_pose = None

        self.goal_x = 1.0
        self.goal_y = 0.0

        self.timer = self.create_timer(0.1, self.update)

        # CSV output
        self.csv_file = '/home/panagiotagrosd/benchmark_results.csv'
        self.get_logger().info(f"Logging to {self.csv_file}")

    def scan_cb(self, msg):
        filtered = [r for r in msg.ranges if not math.isinf(r) and not math.isnan(r)]
        if len(filtered):
            self.min_clearance = min(self.min_clearance, min(filtered))

    def odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def dist_to_goal(self):
        if self.current_pose is None:
            return float('inf')
        dx = self.goal_x - self.current_pose.position.x
        dy = self.goal_y - self.current_pose.position.y
        return math.hypot(dx, dy)

    def update(self):
        if self.dist_to_goal() < 0.15:
            end_t = time.time()
            duration = end_t - self.start_time

            with open(self.csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([duration, self.min_clearance])

            self.get_logger().info("TRIAL FINISHED!")
            self.get_logger().info(f"Time: {duration:.2f} sec")
            self.get_logger().info(f"Min clearance: {self.min_clearance:.3f} m")


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
