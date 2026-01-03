import heapq
import math
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        self.map = None
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

    def map_callback(self, msg):
        self.map = msg
        self.get_logger().info("Map received! Ready for A* planning.")

    def plan(self, start, goal):
        if self.map is None:
            self.get_logger().error("No map available!")
            return None

        width = self.map.info.width
        height = self.map.info.height
        resolution = self.map.info.resolution
        origin = self.map.info.origin.position

        grid = np.array(self.map.data).reshape((height, width))

        # Convert world → grid indices
        def world_to_grid(x, y):
            gx = int((x - origin.x) / resolution)
            gy = int((y - origin.y) / resolution)
            return gx, gy

        start_g = world_to_grid(start[0], start[1])
        goal_g = world_to_grid(goal[0], goal[1])

        # A* --------------------------
        def heuristic(a, b):
            return math.dist(a, b)

        open_set = []
        heapq.heappush(open_set, (0, start_g))

        came_from = {}
        g_score = {start_g: 0}

        neighbors = [(1,0), (-1,0), (0,1), (0,-1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_g:
                return self.reconstruct_path(came_from, current)

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy

                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue

                # check obstacle
                if grid[ny][nx] > 50:
                    continue

                tentative_g = g_score[current] + 1

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic((nx, ny), goal_g)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def main(args=None):
    rclpy.init(args=args)
    planner = AStarPlanner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
