import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class DynamicNavNode(Node):
    """
    Τοπικός πλοηγός για TurtleBot3 με:
    - ROS2 parameters (forward_speed, rotate_speed, obstacle_threshold,
      clear_threshold, control_period)
    - adaptive λογική (obstacle_density) από το LiDAR
    - finite-state μηχανή: FORWARD / ROTATING
    - logging metrics: χρόνος, re-routing events, min clearance, control freq
    """

    def __init__(self):
        super().__init__("dynamic_nav_node")

        # --- Δήλωση παραμέτρων με default τιμές ---
        self.declare_parameter("forward_speed", 0.2)
        self.declare_parameter("rotate_speed", 0.5)
        self.declare_parameter("obstacle_threshold", 0.5)
        self.declare_parameter("clear_threshold", 0.8)
        self.declare_parameter("control_period", 0.1)

        # Ανάγνωση τιμών παραμέτρων
        self.forward_speed = self.get_parameter("forward_speed").value
        self.rotate_speed = self.get_parameter("rotate_speed").value
        self.obstacle_threshold = self.get_parameter("obstacle_threshold").value
        self.clear_threshold = self.get_parameter("clear_threshold").value
        control_period = self.get_parameter("control_period").value

        # --- Μεταβλητές για adaptive λογική ---
        self.obstacle_density = 0.0

        # Βασικές σταθερές για adaptive mapping
        self.v_min = 0.1
        self.v_max = self.forward_speed  # "κανονική" μέγιστη ταχύτητα

        self.d_min = 0.4
        self.d_max = 0.8

        self.w_min = 0.5
        self.w_max = self.rotate_speed

        # --- Metrics για αξιολόγηση απόδοσης ---
        self.start_time = self.get_clock().now()
        self.reroute_count = 0
        self.step_count = 0
        self.min_clearance = float("inf")

        self.get_logger().info(
            "Dynamic navigation node started with parameters:\n"
            f"  forward_speed      = {self.forward_speed:.2f} m/s\n"
            f"  rotate_speed       = {self.rotate_speed:.2f} rad/s\n"
            f"  obstacle_threshold = {self.obstacle_threshold:.2f} m\n"
            f"  clear_threshold    = {self.clear_threshold:.2f} m\n"
            f"  control_period     = {control_period:.2f} s"
        )

        # Publisher για ταχύτητες
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Subscriber για LaserScan
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        # Ελάχιστη απόσταση μπροστά (ενημερώνεται από το scan)
        self.min_front_dist = None

        # Κατάσταση: "FORWARD" ή "ROTATING"
        self.state = "FORWARD"

        # Timer loop ελέγχου
        self.control_timer = self.create_timer(control_period, self.control_loop)

    def scan_callback(self, msg: LaserScan):
        """
        Επεξεργασία LaserScan:
        - υπολογίζει min_front_dist σε ένα παράθυρο μπροστά
        - εκτιμά obstacle_density στο [0,1] μέσα σε ακτίνα R
        - ενημερώνει min_clearance
        """
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0:
            self.min_front_dist = None
            self.obstacle_density = 0.0
            return

        # Παράθυρο μπροστά από το ρομπότ (π.χ. ~60°)
        window_size = max(10, n // 6)
        center = n // 2
        start = max(0, center - window_size // 2)
        end = min(n, center + window_size // 2)

        front_ranges = []
        valid_all = []

        for idx, r in enumerate(ranges):
            if math.isfinite(r) and r > 0.0:
                valid_all.append(r)
                if start <= idx < end:
                    front_ranges.append(r)

        # ελάχιστη απόσταση μπροστά
        if front_ranges:
            self.min_front_dist = min(front_ranges)
        else:
            self.min_front_dist = None

        # ενημέρωση ελάχιστης απόστασης (clearance)
        if self.min_front_dist is not None:
            self.min_clearance = min(self.min_clearance, self.min_front_dist)

        # Εκτίμηση obstacle density
        if valid_all:
            R = 1.5  # [m] "κοντινά" εμπόδια
            close = [r for r in valid_all if r < R]
            density = len(close) / len(valid_all)
            # clamp στο [0,1]
            self.obstacle_density = max(0.0, min(1.0, density))
        else:
            self.obstacle_density = 0.0

    def update_adaptive_parameters(self):
        """
        Προσαρμογή παραμέτρων με βάση obstacle_density στο [0,1].
        - όσο πιο πυκνό περιβάλλον (density→1):
            -> μικρότερη forward_speed
            -> μεγαλύτερο obstacle_threshold
            -> μεγαλύτερη rotate_speed
        """
        d = self.obstacle_density

        # Προσαρμοσμένη ταχύτητα
        self.forward_speed = self.v_min + (self.v_max - self.v_min) * (1.0 - d)

        # Προσαρμοσμένο obstacle threshold
        self.obstacle_threshold = self.d_min + (self.d_max - self.d_min) * d

        # Προσαρμοσμένη rotate speed
        self.rotate_speed = self.w_min + (self.w_max - self.w_min) * d

    def control_loop(self):
        """Finite-state λογική πλοήγησης με obstacle avoidance + adaptive tuning."""
        twist = Twist()

        # μετράμε βήματα ελέγχου
        self.step_count += 1

        # Αν δεν έχουμε ακόμα έγκυρο scan, δεν κινούμαστε
        if self.min_front_dist is None:
            self.cmd_pub.publish(twist)
            return

        # Adaptive update
        self.update_adaptive_parameters()

        clear_threshold = self.clear_threshold

        if self.state == "FORWARD":
            if self.min_front_dist < self.obstacle_threshold:
                # Εμπόδιο πολύ κοντά -> κατάσταση περιστροφής
                self.get_logger().info(
                    f"Obstacle detected at {self.min_front_dist:.2f} m "
                    f"(density={self.obstacle_density:.2f}), re-routing..."
                )
                self.state = "ROTATING"
                self.reroute_count += 1
                twist.linear.x = 0.0
                twist.angular.z = self.rotate_speed
            else:
                # Καθαρός δρόμος -> κίνηση ευθεία με adaptive ταχύτητα
                twist.linear.x = self.forward_speed
                twist.angular.z = 0.0

        elif self.state == "ROTATING":
            if self.min_front_dist > clear_threshold:
                # Ο δρόμος μπροστά καθάρισε -> συνεχίζουμε ευθεία
                self.get_logger().info(
                    f"Path is clear again ({self.min_front_dist:.2f} m, "
                    f"density={self.obstacle_density:.2f}), resuming forward motion."
                )
                self.state = "FORWARD"
                twist.linear.x = self.forward_speed
                twist.angular.z = 0.0
            else:
                # Συνεχίζουμε να περιστρέφουμε μέχρι να βρεθεί κενός χώρος
                twist.linear.x = 0.0
                twist.angular.z = self.rotate_speed

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        end_time = node.get_clock().now()
        duration = (end_time - node.start_time).nanoseconds / 1e9  # σε sec

        if node.step_count > 0 and duration > 0.0:
            avg_freq = node.step_count / duration
        else:
            avg_freq = 0.0

        if node.min_clearance == float("inf"):
            min_clearance = float("nan")
        else:
            min_clearance = node.min_clearance

        node.get_logger().info(
            "\n===== Navigation run statistics =====\n"
            f"  Total time:           {duration:.2f} s\n"
            f"  Control steps:        {node.step_count}\n"
            f"  Avg control frequency:{avg_freq:.2f} Hz\n"
            f"  Re-routing events:    {node.reroute_count}\n"
            f"  Min clearance:        {min_clearance:.2f} m\n"
            "=====================================\n"
        )

        node.get_logger().info("Shutting down dynamic navigation node.")
        node.destroy_node()
        rclpy.shutdown()


