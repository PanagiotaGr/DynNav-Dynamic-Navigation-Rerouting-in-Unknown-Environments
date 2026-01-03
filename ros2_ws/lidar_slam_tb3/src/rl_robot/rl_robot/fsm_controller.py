import math
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class State(Enum):
    SEARCH_GOAL = auto()
    GO_TO_GOAL = auto()
    AVOID_OBSTACLE = auto()
    REROUTE = auto()
    RECOVERY = auto()


class FSMController(Node):
    def __init__(self):
        super().__init__('fsm_controller')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.last_scan = None
        self.last_odom = None

        self.state = State.SEARCH_GOAL

        # Hardcoded goal for now
        self.goal_x = 1.0
        self.goal_y = 0.0

        # Thresholds
        self.safe_dist = 0.5   # start avoiding
        self.clear_dist = 0.8  # return to goal seeking
        self.goal_tol = 0.15   # consider goal reached

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    def odom_cb(self, msg: Odometry):
        self.last_odom = msg

    def min_front_distance(self):
        if self.last_scan is None:
            return float('inf')

        ranges = self.last_scan.ranges
        n = len(ranges)
        angle_min = self.last_scan.angle_min
        angle_inc = self.last_scan.angle_increment

        min_deg, max_deg = -30, 30
        min_rad, max_rad = math.radians(min_deg), math.radians(max_deg)

        vals = []
        angle = angle_min
        for i in range(n):
            if min_rad <= angle <= max_rad:
                r = ranges[i]
                if not math.isinf(r) and not math.isnan(r):
                    vals.append(r)
            angle += angle_inc

        return min(vals) if vals else float('inf')

    def distance_to_goal(self):
        if self.last_odom is None:
            return float('inf')
        x = self.last_odom.pose.pose.position.x
        y = self.last_odom.pose.pose.position.y
        dx = self.goal_x - x
        dy = self.goal_y - y
        return math.hypot(dx, dy)

    def control_loop(self):
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_footprint'

        d_front = self.min_front_distance()
        d_goal = self.distance_to_goal()

        # STATE TRANSITIONS -----------------------
        if d_goal < self.goal_tol:
            self.get_logger().info("GOAL REACHED!")
            self.cmd_pub.publish(twist)
            return

        if d_front < self.safe_dist:
            self.state = State.AVOID_OBSTACLE
        elif self.state == State.AVOID_OBSTACLE and d_front > self.clear_dist:
            self.state = State.GO_TO_GOAL
        elif self.state == State.SEARCH_GOAL:
            self.state = State.GO_TO_GOAL

        # FSM ACTIONS -----------------------------
        if self.state == State.GO_TO_GOAL:
            twist.twist.linear.x = 0.15
            twist.twist.angular.z = 0.0

        elif self.state == State.AVOID_OBSTACLE:
            twist.twist.linear.x = 0.0
            twist.twist.angular.z = 0.5

        elif self.state == State.SEARCH_GOAL:
            twist.twist.linear.x = 0.05
            twist.twist.angular.z = 0.3

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = FSMController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
