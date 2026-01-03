import math
import heapq

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class AStarPlanner(Node):
    def __init__(self):
        super().__init__("astar_planner")

        # Συνδρομή στον χάρτη SLAM
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            10
        )

        self.map = None
        self.get_logger().info("A* planner node started. Waiting for /map ...")

        # Παράδειγμα start/goal (για δοκιμή).
        # Αργότερα θα τα παίρνουμε από RViz ή άλλο node.
        self.example_start = (0.0, 0.0)
        self.example_goal = (1.0, 0.0)

        # Μόλις λάβουμε χάρτη, θα καλέσουμε A* μία φορά
        self.planned_once = False

    def map_callback(self, msg: OccupancyGrid):
        self.map = msg

        if not self.planned_once:
            self.planned_once = True
            self.get_logger().info("Map received, running example A* plan...")
            path = self.plan(self.example_start, self.example_goal)
            if path is None:
                self.get_logger().warn("A* could not find a path.")
            else:
                self.get_logger().info(f"A* path length (in grid cells): {len(path)}")
                # Προαιρετικά: εκτύπωση πρώτων μερικών σημείων
                for i, p in enumerate(path[:10]):
                    self.get_logger().info(f"  step {i}: grid={p}")

    # ---------------- A* πάνω σε OccupancyGrid ----------------

    def plan(self, start_world, goal_world):
        """
        Υλοποίηση A* πάνω σε OccupancyGrid.
        start_world, goal_world: (x, y) σε μέτρα, στο frame του χάρτη.
        Επιστρέφει λίστα από grid cells [(gx, gy), ...] ή None.
        """
        if self.map is None:
            self.get_logger().error("No map available for A* planning.")
            return None

        width = self.map.info.width
        height = self.map.info.height
        resolution = self.map.info.resolution
        origin = self.map.info.origin.position

        data = np.array(self.map.data, dtype=np.int8).reshape((height, width))

        def world_to_grid(x, y):
            gx = int((x - origin.x) / resolution)
            gy = int((y - origin.y) / resolution)
            return gx, gy

        start_g = world_to_grid(start_world[0], start_world[1])
        goal_g = world_to_grid(goal_world[0], goal_world[1])
        self.get_logger().info(f"Start grid: {start_g}, Goal grid: {goal_g}")

        # Έλεγχος ότι start/goal είναι μέσα στον χάρτη
        if not self.in_bounds(start_g, width, height) or not self.in_bounds(goal_g, width, height):
            self.get_logger().error("Start or goal is outside map bounds.")
            return None

        # A* δομές
        open_set = []
        heapq.heappush(open_set, (0.0, start_g))

        came_from = {}
        g_score = {start_g: 0.0}

        def heuristic(a, b):
            # Ευκλείδεια απόσταση σε grid coordinates
            return math.dist(a, b)

        neighbors_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_g:
                # Ανακατασκευή διαδρομής
                return self.reconstruct_path(came_from, current)

            for dx, dy in neighbors_4:
                nx, ny = current[0] + dx, current[1] + dy

                if not self.in_bounds((nx, ny), width, height):
                    continue

                # Εμπόδια: θεωρούμε >50 ως occupied
                if data[ny, nx] > 50:
                    continue

                tentative_g = g_score[current] + 1.0  # σταθερό κόστος ανά βήμα

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic((nx, ny), goal_g)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current

        return None

    @staticmethod
    def in_bounds(cell, width, height):
        x, y = cell
        return 0 <= x < width and 0 <= y < height

    @staticmethod
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
