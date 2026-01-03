#!/usr/bin/env python3
import math, time, rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

def dist(p1, p2):
    dx = p1[0]-p2[0]; dy = p1[1]-p2[1]
    return math.hypot(dx, dy)

class MetricsNode(Node):
    def __init__(self, goal_xy):
        super().__init__('nav2_metrics')
        self.goal_xy = goal_xy
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.cb_odom, 10)
        self.plan_sub = self.create_subscription(Path, '/plan', self.cb_plan, 10)
        self.action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.prev_xy = None
        self.path_len = 0.0
        self.replans = 0
        self.last_plan_stamp = None
        self.start_t = None

    def cb_odom(self, msg):
        xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.prev_xy is not None:
            self.path_len += dist(xy, self.prev_xy)
        self.prev_xy = xy

    def cb_plan(self, msg):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if self.last_plan_stamp is None:
            self.last_plan_stamp = stamp
        elif stamp != self.last_plan_stamp:
            self.replans += 1
            self.last_plan_stamp = stamp

    def send_goal_and_measure(self):
        self.get_logger().info('Waiting for NavigateToPose action server...')
        self.action_client.wait_for_server()
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = float(self.goal_xy[0])
        goal.pose.pose.position.y = float(self.goal_xy[1])
        goal.pose.pose.orientation.w = 1.0
        self.start_t = time.time()
        send_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        end_t = time.time()
        ok = (result_future.result().status == 4)  # SUCCEEDED
        t = end_t - self.start_t
        print(f'=== METRICS ===\nSuccess={ok}\nTimeToGoal_s={t:.3f}\nPathLength_m={self.path_len:.3f}\nReplans={self.replans}')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=2.0)
    parser.add_argument('--y', type=float, default=0.5)
    args = parser.parse_args()
    rclpy.init()
    node = MetricsNode((args.x, args.y))
    # δώσε 1s να ξεκινήσουν τα topics
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < 1.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.send_goal_and_measure()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
