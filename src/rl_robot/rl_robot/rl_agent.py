import math
import random
from collections import defaultdict

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan


class QLearningAgent(Node):
    def __init__(self):
        super().__init__('q_learning_agent')

        # ROS publishers / subscribers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer loop (RL βήμα) στα 10 Hz
        self.timer = self.create_timer(0.1, self.step)

        # Τελευταίο LaserScan
        self.last_scan = None

        # Q-table: dict[(state, action)] -> value
        self.Q = defaultdict(float)

        # RL hyperparameters
        self.alpha = 0.2      # learning rate
        self.gamma = 0.95     # discount factor
        self.epsilon = 0.2    # exploration rate

        # Για να κάνουμε update: (s, a, r, s')
        self.prev_state = None
        self.prev_action = None

    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg

    # ----- Helper: απόσταση σε sectors & discretization -----

    def get_state_from_scan(self):
        """Μετατρέπει το LaserScan σε διακριτό state (left, front, right)."""
        if self.last_scan is None:
            # State "άγνοιας"
            return (0, 0, 0)

        ranges = self.last_scan.ranges
        n = len(ranges)
        angle_min = self.last_scan.angle_min
        angle_increment = self.last_scan.angle_increment

        # Ορίζουμε 3 sectors: left, front, right
        # Π.χ. front = [-30°, +30°], left = [30°, 90°], right = [-90°, -30°]
        def sector_min_deg(min_deg, max_deg):
            min_rad = math.radians(min_deg)
            max_rad = math.radians(max_deg)

            indices = []
            angle = angle_min
            for i in range(n):
                if min_rad <= angle <= max_rad:
                    indices.append(i)
                angle += angle_increment

            vals = []
            for i in indices:
                r = ranges[i]
                if not math.isinf(r) and not math.isnan(r):
                    vals.append(r)

            if len(vals) == 0:
                return float('inf')
            return min(vals)

        d_front = sector_min_deg(-30, 30)
        d_left = sector_min_deg(30, 90)
        d_right = sector_min_deg(-90, -30)

        # Discretization σε 3 επίπεδα
        def discretize(d):
            if d < 0.5:
                return 2  # near
            elif d < 1.0:
                return 1  # medium
            else:
                return 0  # far

        s_left = discretize(d_left)
        s_front = discretize(d_front)
        s_right = discretize(d_right)

        return (s_left, s_front, s_right)

    # ----- Helper: policy & Q-learning -----

    def choose_action(self, state):
        """ε-greedy επιλογή action."""
        if random.random() < self.epsilon:
            # exploration
            return random.choice([0, 1, 2])

        # exploitation: argmax_a Q(s,a)
        q_vals = [self.Q[(state, a)] for a in [0, 1, 2]]
        max_q = max(q_vals)
        # Αν πολλά έχουν ίδιο Q, επέλεξε τυχαίο από αυτά
        best_actions = [a for a, q in zip([0, 1, 2], q_vals) if q == max_q]
        return random.choice(best_actions)

    def compute_reward(self, state):
        """Reward με βάση το state."""
        _, front, _ = state
        if front == 2:  # near μπροστά
            return -1.0
        else:
            return 0.1

    def publish_action(self, action):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_footprint'  # ή 'base_link'

        if action == 0:  # go_forward
            twist_msg.twist.linear.x = 0.15
            twist_msg.twist.angular.z = 0.0
        elif action == 1:  # turn_left
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.angular.z = 0.5
        elif action == 2:  # turn_right
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.angular.z = -0.5

        self.cmd_vel_pub.publish(twist_msg)

    def step(self):
        """Ένα RL βήμα: observe s, get r, update Q, choose a, act."""
        # Χωρίς scan, δεν κάνουμε τίποτα
        if self.last_scan is None:
            stop_msg = TwistStamped()
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_footprint'
            stop_msg.twist.linear.x = 0.0
            stop_msg.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_msg)
            return

        # Τρέχον state
        current_state = self.get_state_from_scan()

        # Αν έχουμε προηγούμενη (s, a), μπορούμε να κάνουμε Q update
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.compute_reward(current_state)

            # Q-learning update
            prev_sa = (self.prev_state, self.prev_action)

            max_next_Q = max(self.Q[(current_state, a)] for a in [0, 1, 2])

            td_target = reward + self.gamma * max_next_Q
            td_error = td_target - self.Q[prev_sa]

            self.Q[prev_sa] += self.alpha * td_error

            # (προαιρετικά) log
            # self.get_logger().info(
            #     f"s={self.prev_state}, a={self.prev_action}, r={reward:.2f}, Q={self.Q[prev_sa]:.3f}"
            # )

        # Επιλογή επόμενης δράσης (ε-greedy)
        action = self.choose_action(current_state)

        # Εκτέλεση δράσης
        self.publish_action(action)

        # Αποθήκευση για το επόμενο βήμα
        self.prev_state = current_state
        self.prev_action = action


def main(args=None):
    rclpy.init(args=args)
    node = QLearningAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

