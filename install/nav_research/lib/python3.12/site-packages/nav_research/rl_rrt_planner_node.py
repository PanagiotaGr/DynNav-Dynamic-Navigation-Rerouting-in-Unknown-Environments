import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from .advanced_planners import (
    train_q_learning,
    extract_policy_path,
)


class RLPlannerNode(Node):
    """
    RL-based Path Planner (GridWorld + Q-learning) με ROS2 interface.

    - Εκπαιδεύει μία φορά Q-learning agent σε GridWorld.
    - Περιμένει goal από /goal_pose (PoseStamped).
    - Υπολογίζει διαδρομή στο grid με το RL policy.
    - Δημοσιεύει nav_msgs/Path στο /planned_path.

    Σημείωση:
    - Το περιβάλλον είναι 2D grid [0..grid_size-1] x [0..grid_size-1].
    - Θεωρούμε fixed start στο (0, 0).
    - Το goal από RViz προβάλλεται σε κοντινό grid cell.
    """

    def __init__(self):
        super().__init__("rl_rrt_planner")

        # Παράμετροι
        self.declare_parameter("grid_size", 10)
        self.grid_size = self.get_parameter("grid_size").get_parameter_value().integer_value
        if self.grid_size <= 1:
            self.grid_size = 10

        self.get_logger().info(f"[RLPlannerNode] Initializing with grid_size={self.grid_size}")

        # Εκπαίδευση Q-learning agent σε GridWorld
        # (χρησιμοποιεί internally GridWorldEnv από advanced_planners)
        episodes = 500
        self.env, self.agent = train_q_learning(
            episodes=episodes,
            width=self.grid_size,
            height=self.grid_size,
            start=(0, 0),
            goal=(self.grid_size - 1, self.grid_size - 1),
            obstacles=None,  # μπορείς να βάλεις εμπόδια αν θέλεις
        )

        self.get_logger().info(f"[RLPlannerNode] Q-learning training done after {episodes} episodes.")

        # Publisher για path
        self.path_pub = self.create_publisher(Path, "/planned_path", 10)

        # Subscriber για goal από RViz / άλλο node
        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_callback,
            10,
        )

        self.get_logger().info("[RLPlannerNode] Waiting for /goal_pose messages...")


    def goal_callback(self, msg: PoseStamped):
        """
        Callback όταν έρχεται νέο goal από /goal_pose.
        """
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        self.get_logger().info(
            f"[RLPlannerNode] Received goal pose: x={gx:.2f}, y={gy:.2f}"
        )

        # Προβολή goal σε grid [0 .. grid_size-1]
        goal_cell = self._world_to_grid(gx, gy)
        self.get_logger().info(f"[RLPlannerNode] Mapped goal to grid cell {goal_cell}")

        # (Προσεγγιστικά) ενημερώνουμε το goal του περιβάλλοντος
        self.env.goal = goal_cell

        # Επαναφορά στο start
        self.env.reset()

        # Εξαγωγή path με το policy του agent
        grid_path = extract_policy_path(self.env, self.agent, max_steps=self.grid_size * self.grid_size)

        self.get_logger().info(f"[RLPlannerNode] RL grid path length = {len(grid_path)}")

        # Μετατροπή από grid path -> nav_msgs/Path
        path_msg = self._grid_path_to_ros_path(grid_path)

        self.path_pub.publish(path_msg)
        self.get_logger().info("[RLPlannerNode] Published RL-based /planned_path")


    def _world_to_grid(self, x: float, y: float):
        """
        Πολύ απλή χαρτογράφηση world → grid:
        Θεωρούμε ότι ο χρήστης κλικάρει κοντά στο [0, grid_size-1].

        - Κόβουμε (clamp) στις άκρες.
        - Στρογγυλοποιούμε στο κοντινότερο cell.
        """
        gx = int(round(x))
        gy = int(round(y))

        gx = max(0, min(self.grid_size - 1, gx))
        gy = max(0, min(self.grid_size - 1, gy))

        return (gx, gy)


    def _grid_path_to_ros_path(self, grid_path):
        """
        Μετατρέπει μια λίστα από grid states [(ix, iy), ...]
        σε nav_msgs/Path στο frame "map".
        Υποθέτουμε cell_size = 1.0 (1 unit = 1 meter).
        """
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for (ix, iy) in grid_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(ix)
            pose.pose.position.y = float(iy)
            # Τα orientation τα αφήνουμε default (0,0,0,1)
            path_msg.poses.append(pose)

        return path_msg


def main(args=None):
    rclpy.init(args=args)
    node = RLPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# ============================================================
# 9) RL MODEL COMPARISON / EVALUATION
# ============================================================

from dataclasses import dataclass


@dataclass
class RLModelConfig:
    name: str
    episodes: int
    obstacles: set | None  # set of (x, y) grid cells with obstacles


def train_rl_model(config: RLModelConfig, grid_size: int = 10):
    """
    Εκπαιδεύει ένα RL μοντέλο (Q-learning) για συγκεκριμένη διαμόρφωση εμποδίων.
    Επιστρέφει (env, agent).
    """
    env, agent = train_q_learning(
        episodes=config.episodes,
        width=grid_size,
        height=grid_size,
        start=(0, 0),
        goal=(grid_size - 1, grid_size - 1),
        obstacles=config.obstacles,
    )
    return env, agent


def evaluate_agent(env: GridWorldEnv, agent: QLearningAgent,
                   n_episodes: int = 100, max_steps: int = 200):
    """
    Αξιολόγηση ενός εκπαιδευμένου agent σε πολλά επεισόδια.

    Μετράμε:
      - success_rate
      - mean_steps_successful
      - mean_return
    """
    successes = 0
    returns = []
    steps_success = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                if state == env.goal:
                    successes += 1
                    steps_success.append(t + 1)
                break

        returns.append(total_reward)

    success_rate = successes / n_episodes if n_episodes > 0 else 0.0
    mean_return = float(np.mean(returns)) if returns else 0.0
    mean_steps_successful = (
        float(np.mean(steps_success)) if steps_success else float("inf")
    )

    return {
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_steps_successful": mean_steps_successful,
        "episodes": n_episodes,
    }


def compare_rl_models():
    """
    Συγκρίνει πολλαπλά RL μοντέλα (διαφορετικά εμπόδια / training episodes)
    και τυπώνει πίνακα με μετρικές.
    """
    grid_size = 10

    # Ορισμός μοντέλων
    configs = [
        RLModelConfig(
            name="no_obstacles",
            episodes=500,
            obstacles=None,
        ),
        RLModelConfig(
            name="center_wall",
            episodes=500,
            obstacles={(3, 3), (3, 4), (3, 5), (4, 5), (5, 5)},
        ),
        RLModelConfig(
            name="diagonal_barrier",
            episodes=700,
            obstacles={(i, i) for i in range(3, 7)},
        ),
    ]

    results = []

    print("=== RL MODEL COMPARISON (GridWorld) ===")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"{'model':20s}  {'succ_rate':>9s}  {'mean_steps_succ':>15s}  {'mean_return':>12s}")

    for cfg in configs:
        print(f"\n[INFO] Training model '{cfg.name}' for {cfg.episodes} episodes...")
        env, agent = train_rl_model(cfg, grid_size=grid_size)
        metrics = evaluate_agent(env, agent, n_episodes=100, max_steps=grid_size * grid_size)
        results.append((cfg, metrics))

        print(f"     -> success_rate       = {metrics['success_rate']:.3f}")
        print(f"     -> mean_steps_success = {metrics['mean_steps_successful']:.2f}")
        print(f"     -> mean_return        = {metrics['mean_return']:.2f}")

    print("\n=== SUMMARY TABLE ===")
    print(f"{'model':20s}  {'succ_rate':>9s}  {'mean_steps_succ':>15s}  {'mean_return':>12s}")
    for cfg, m in results:
        print(
            f"{cfg.name:20s}  "
            f"{m['success_rate']:9.3f}  "
            f"{m['mean_steps_successful']:15.2f}  "
            f"{m['mean_return']:12.2f}"
        )


# ============================================================
# MAIN ENTRY POINT: RUN RL MODEL COMPARISON
# ===========================================================
