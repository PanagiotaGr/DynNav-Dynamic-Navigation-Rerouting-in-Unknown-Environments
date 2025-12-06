"""
Advanced Path Planning Module

Περιέχει:
  7) RL-based Path Planning (GridWorld + Q-learning)
  8) Sampling-based Planners (PRM / RRT)
"""

import random
import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass


# ============================================================
# 7) RL-BASED PATH PLANNING (GRIDWORLD + Q-LEARNING)
# ============================================================

class GridWorldEnv:
    """
    Απλό 2D GridWorld περιβάλλον για path planning με εμπόδια.
    State = (x, y)
    Actions = 0: up, 1: right, 2: down, 3: left
    """

    def __init__(self, width=10, height=10, start=(0, 0), goal=(9, 9), obstacles=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles is not None else set()
        self.action_space = [0, 1, 2, 3]
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state

        if action == 0:      # up
            y = max(0, y - 1)
        elif action == 1:    # right
            x = min(self.width - 1, x + 1)
        elif action == 2:    # down
            y = min(self.height - 1, y + 1)
        elif action == 3:    # left
            x = max(0, x - 1)

        next_state = (x, y)

        # Αν πέσουμε σε εμπόδιο, μικρό penalty και δεν κινείται
        if next_state in self.obstacles:
            reward = -5.0
            next_state = self.state
        else:
            # Βασικό shaping reward: αρνητικό βήμα, θετικό όταν πλησιάζει goal
            old_dist = self._manhattan(self.state, self.goal)
            new_dist = self._manhattan(next_state, self.goal)

            step_cost = -1.0
            progress_reward = (old_dist - new_dist) * 0.5
            reward = step_cost + progress_reward

        done = False
        if next_state == self.goal:
            reward = +50.0
            done = True

        self.state = next_state
        return next_state, reward, done, {}

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


class QLearningAgent:
    """
    Q-learning agent για GridWorld.
    """

    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.2):
        self.q = defaultdict(lambda: np.zeros(len(actions), dtype=float))
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _state_key(self, state):
        # μετατροπή (x, y) → string key
        return f"{state[0]}_{state[1]}"

    def select_action(self, state):
        state_key = self._state_key(state)

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return int(np.argmax(self.q[state_key]))

    def update(self, state, action, reward, next_state, done):
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        best_next = 0.0
        if not done:
            best_next = np.max(self.q[next_state_key])

        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state_key][action]

        self.q[state_key][action] += self.alpha * td_error


def train_q_learning(
    episodes=2000,
    width=10,
    height=10,
    start=(0, 0),
    goal=(9, 9),
    obstacles=None,
):
    """
    Απλό training loop για Q-learning σε GridWorld.
    """
    env = GridWorldEnv(width, height, start, goal, obstacles)
    agent = QLearningAgent(actions=env.action_space)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for t in range(500):  # max steps per episode
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        # Optional: ελαφρύ decay της εpsilon για λιγότερο exploration
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.999

        if (ep + 1) % 100 == 0:
            print(f"[RL] Episode {ep+1}/{episodes}, total_reward={total_reward:.2f}")

    return env, agent


def extract_policy_path(env: GridWorldEnv, agent: QLearningAgent, max_steps=200):
    """
    Χρησιμοποιεί το εκπαιδευμένο Q-table για να βρει path από start → goal.
    """
    state = env.reset()
    path = [state]

    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, _, done, _ = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    return path


# ============================================================
# 8) SAMPLING-BASED PLANNERS: PRM / RRT
# ============================================================

@dataclass
class ObstacleRect:
    """
    Ορθογώνιο εμπόδιο στον 2D χώρο.
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x, y) -> bool:
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)


def collision_free(x, y, obstacles):
    return all(not obs.contains(x, y) for obs in obstacles)


# -----------------------------
# PRM (Probabilistic Roadmap)
# -----------------------------

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def prm_build_roadmap(
    n_samples,
    k_neighbors,
    obstacles,
    bounds=((0.0, 1.0), (0.0, 1.0))
):
    """
    Κατασκευή Probabilistic Roadmap (PRM) σε 2D.
    Επιστρέφει:
      nodes: λίστα σημείων (x, y)
      edges: adjacency list (λεξικό από node index -> λίστα (neighbor_idx, cost))
    """
    nodes = []
    edges = {}

    # 1) Δειγματοληψία σημείων
    while len(nodes) < n_samples:
        x = random.uniform(bounds[0][0], bounds[0][1])
        y = random.uniform(bounds[1][0], bounds[1][1])
        if collision_free(x, y, obstacles):
            nodes.append((x, y))

    # 2) Σύνδεση κάθε κόμβου με k πιο κοντινούς
    for i, p in enumerate(nodes):
        distances = [(j, euclidean(p, nodes[j])) for j in range(len(nodes)) if j != i]
        distances.sort(key=lambda tup: tup[1])
        neighbors = distances[:k_neighbors]

        edges[i] = []
        for j, dist in neighbors:
            # Εδώ μπορούμε να βάλουμε line-collision check (για απλότητα το αγνοούμε
            # ή θεωρούμε ότι η απόσταση είναι μικρή και free)
            edges[i].append((j, dist))

    return nodes, edges


def dijkstra(nodes, edges, start_idx, goal_idx):
    """
    Dijkstra πάνω σε roadmap για shortest path.
    """
    n = len(nodes)
    dist = [math.inf] * n
    prev = [-1] * n
    dist[start_idx] = 0.0

    visited = set()
    while len(visited) < n:
        # πάρε τον κόμβο με την ελάχιστη dist που δεν είναι visited
        current = None
        current_dist = math.inf
        for i in range(n):
            if i not in visited and dist[i] < current_dist:
                current = i
                current_dist = dist[i]

        if current is None:
            break
        if current == goal_idx:
            break

        visited.add(current)

        for neighbor, cost in edges.get(current, []):
            if neighbor in visited:
                continue
            alt = dist[current] + cost
            if alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = current

    # reconstruct path
    path_idx = []
    node = goal_idx
    while node != -1:
        path_idx.append(node)
        node = prev[node]
    path_idx.reverse()

    path = [nodes[i] for i in path_idx]
    return path


# -----------------------------
# RRT (Rapidly-exploring Random Tree)
# -----------------------------

@dataclass
class RRTNode:
    x: float
    y: float
    parent: int


class RRTPlanner:
    """
    Απλός RRT planner σε 2D.
    """

    def __init__(
        self,
        start,
        goal,
        obstacles,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        step_size=0.05,
        goal_sample_rate=0.05,
        max_iters=5000,
        goal_tolerance=0.05
    ):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iters = max_iters
        self.goal_tolerance = goal_tolerance

        self.nodes = [RRTNode(start[0], start[1], parent=-1)]

    def plan(self):
        for it in range(self.max_iters):
            # 1) δείγμα σημείου
            if random.random() < self.goal_sample_rate:
                x_rand, y_rand = self.goal
            else:
                x_rand = random.uniform(self.bounds[0][0], self.bounds[0][1])
                y_rand = random.uniform(self.bounds[1][0], self.bounds[1][1])

            # 2) εύρεση κοντινότερου κόμβου
            nearest_idx = self._nearest_node_index(x_rand, y_rand)
            x_near, y_near = self.nodes[nearest_idx].x, self.nodes[nearest_idx].y

            # 3) κίνηση από near προς rand
            x_new, y_new = self._steer(x_near, y_near, x_rand, y_rand)

            # 4) collision check
            if not collision_free(x_new, y_new, self.obstacles):
                continue

            # 5) πρόσθεση νέου κόμβου στο δέντρο
            new_node = RRTNode(x_new, y_new, parent=nearest_idx)
            self.nodes.append(new_node)

            # 6) έλεγχος αν φτάσαμε κοντά στο goal
            if euclidean((x_new, y_new), self.goal) < self.goal_tolerance:
                print(f"[RRT] Goal reached at iteration {it}")
                return self._reconstruct_path(len(self.nodes) - 1)

        print("[RRT] Goal NOT reached within iteration limit")
        return None

    def _nearest_node_index(self, x, y):
        dists = [
            (i, (node.x - x)**2 + (node.y - y)**2) for i, node in enumerate(self.nodes)
        ]
        dists.sort(key=lambda tup: tup[1])
        return dists[0][0]

    def _steer(self, x_from, y_from, x_to, y_to):
        dx = x_to - x_from
        dy = y_to - y_from
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            return x_from, y_from
        scale = self.step_size / dist
        x_new = x_from + dx * scale
        y_new = y_from + dy * scale
        # clip στα bounds
        x_new = min(max(x_new, self.bounds[0][0]), self.bounds[0][1])
        y_new = min(max(y_new, self.bounds[1][0]), self.bounds[1][1])
        return x_new, y_new

    def _reconstruct_path(self, node_idx):
        path = []
        while node_idx != -1:
            node = self.nodes[node_idx]
            path.append((node.x, node.y))
            node_idx = node.parent
        path.reverse()
        return path


# ============================================================
# DEMO / MAIN (για να το τεστάρεις γρήγορα)
# ============================================================

if __name__ == "__main__":
    # --------- RL DEMO ----------
    print("=== RL-based Path Planning (GridWorld + Q-learning) ===")
    obstacles = {(3, 3), (3, 4), (3, 5), (4, 5), (5, 5)}
    env, agent = train_q_learning(
        episodes=500,
        width=10,
        height=10,
        start=(0, 0),
        goal=(9, 9),
        obstacles=obstacles,
    )
    path = extract_policy_path(env, agent)
    print("RL path:", path)

    # --------- PRM DEMO ----------
    print("\n=== PRM Path Planning ===")
    obs_rects = [ObstacleRect(0.3, 0.3, 0.6, 0.6)]
    nodes, edges = prm_build_roadmap(
        n_samples=200,
        k_neighbors=10,
        obstacles=obs_rects,
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )

    # πρόσθεσε start/goal στο roadmap
    start = (0.05, 0.05)
    goal = (0.95, 0.95)

    nodes.append(start)
    start_idx = len(nodes) - 1
    nodes.append(goal)
    goal_idx = len(nodes) - 1

    # σύνδεση start/goal με κοντινούς κόμβους
    def connect_node(idx, k_neighbors=10):
        p = nodes[idx]
        distances = [(j, euclidean(p, nodes[j])) for j in range(len(nodes)) if j != idx]
        distances.sort(key=lambda tup: tup[1])
        edges[idx] = []
        for j, dist in distances[:k_neighbors]:
            edges[idx].append((j, dist))

    connect_node(start_idx)
    connect_node(goal_idx)

    prm_path = dijkstra(nodes, edges, start_idx, goal_idx)
    print("PRM path length:", len(prm_path))

    # --------- RRT DEMO ----------
    print("\n=== RRT Path Planning ===")
    rrt = RRTPlanner(
        start=start,
        goal=goal,
        obstacles=obs_rects,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        step_size=0.03,
        max_iters=5000,
        goal_tolerance=0.03,
    )
    rrt_path = rrt.plan()
    print("RRT path length:", 0 if rrt_path is None else len(rrt_path))
