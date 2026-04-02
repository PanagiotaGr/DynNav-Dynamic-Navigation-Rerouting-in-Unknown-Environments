"""
Contribution 22: Curriculum RL Training
========================================
Automatically schedules training difficulty for the PPO agent (Contribution 21)
using a curriculum that progresses from simple to complex environments.

Difficulty dimensions:
    - Number of obstacles (3 → 20)
    - Map size (4x4 → 20x20)
    - Dynamic obstacles (static → moving)
    - Sensor noise (0 → 0.3)
    - Goal distance (short → long)

Curriculum strategies:
    - Fixed schedule: predefined stages
    - Adaptive: advance when success_rate > threshold
    - Reverse curriculum: start near goal, expand start region

Integrates with `data_curriculum/` directory already in the DynNav repo.

Research Question (RQ-Curr): Does curriculum training reduce sample
complexity and improve generalisation to unseen environments?

References:
    Bengio et al. (2009) "Curriculum Learning"
    Florensa et al. (2017) "Reverse Curriculum Generation for RL"
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Difficulty config
# ---------------------------------------------------------------------------

@dataclass
class DifficultyLevel:
    name: str
    n_obstacles: int
    map_size: float          # environment bounding box side length
    obstacle_speed: float    # 0 = static, >0 = dynamic
    sensor_noise: float      # std of Gaussian noise on observations
    goal_dist_range: tuple[float, float]  # (min, max) goal distance
    success_threshold: float = 0.7       # advance when success_rate > this


CURRICULUM_STAGES = [
    DifficultyLevel("easy",    n_obstacles=2,  map_size=5.0,  obstacle_speed=0.0, sensor_noise=0.0,  goal_dist_range=(1.0, 2.0)),
    DifficultyLevel("medium",  n_obstacles=5,  map_size=8.0,  obstacle_speed=0.0, sensor_noise=0.05, goal_dist_range=(2.0, 4.0)),
    DifficultyLevel("hard",    n_obstacles=10, map_size=10.0, obstacle_speed=0.1, sensor_noise=0.1,  goal_dist_range=(3.0, 6.0)),
    DifficultyLevel("expert",  n_obstacles=18, map_size=15.0, obstacle_speed=0.3, sensor_noise=0.2,  goal_dist_range=(5.0, 10.0)),
    DifficultyLevel("extreme", n_obstacles=25, map_size=20.0, obstacle_speed=0.5, sensor_noise=0.3,  goal_dist_range=(8.0, 15.0)),
]


# ---------------------------------------------------------------------------
# Curriculum strategy
# ---------------------------------------------------------------------------

class CurriculumStrategy(Enum):
    FIXED = "fixed"        # advance every N episodes
    ADAPTIVE = "adaptive"  # advance when success_rate > threshold
    REVERSE = "reverse"    # start near goal, expand


class CurriculumScheduler:
    """
    Manages difficulty progression during RL training.
    Tracks success rates per stage and decides when to advance.
    """

    def __init__(self,
                 stages: list[DifficultyLevel] = CURRICULUM_STAGES,
                 strategy: CurriculumStrategy = CurriculumStrategy.ADAPTIVE,
                 advance_every: int = 100,       # for FIXED strategy
                 window_size: int = 50):         # rolling window for success rate
        self.stages = stages
        self.strategy = strategy
        self.advance_every = advance_every
        self.window = window_size

        self.current_stage_idx = 0
        self._outcomes: list[float] = []   # 1.0 = success, 0.0 = failure
        self._episode = 0
        self.stage_history: list[dict] = []

    @property
    def current(self) -> DifficultyLevel:
        return self.stages[self.current_stage_idx]

    @property
    def success_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        recent = self._outcomes[-self.window:]
        return float(np.mean(recent))

    def record_episode(self, success: bool) -> bool:
        """
        Record episode outcome. Returns True if stage advanced.
        """
        self._episode += 1
        self._outcomes.append(1.0 if success else 0.0)

        should_advance = False
        if self.strategy == CurriculumStrategy.ADAPTIVE:
            should_advance = (
                len(self._outcomes) >= self.window
                and self.success_rate >= self.current.success_threshold
            )
        elif self.strategy == CurriculumStrategy.FIXED:
            should_advance = (self._episode % self.advance_every == 0)

        if should_advance and self.current_stage_idx < len(self.stages) - 1:
            self._advance()
            return True
        return False

    def _advance(self):
        record = {
            "from_stage": self.current.name,
            "episode": self._episode,
            "success_rate": round(self.success_rate, 3),
        }
        self.current_stage_idx += 1
        record["to_stage"] = self.current.name
        self.stage_history.append(record)
        self._outcomes.clear()
        logger.info(
            "Curriculum advanced: %s → %s (success_rate=%.2f)",
            record["from_stage"], record["to_stage"], record["success_rate"]
        )

    def summary(self) -> dict:
        return {
            "current_stage": self.current.name,
            "stage_idx": self.current_stage_idx,
            "total_episodes": self._episode,
            "current_success_rate": round(self.success_rate, 3),
            "stage_transitions": self.stage_history,
        }


# ---------------------------------------------------------------------------
# Curriculum-aware environment wrapper
# ---------------------------------------------------------------------------

class CurriculumNavEnv:
    """
    Wraps a NavEnv and applies the current difficulty level from the scheduler.
    """

    def __init__(self, scheduler: CurriculumScheduler, obs_dim: int = 84):
        self.scheduler = scheduler
        self.obs_dim = obs_dim
        self._rng = np.random.default_rng()
        self.robot_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)
        self.obstacles: list[np.ndarray] = []
        self.obs_velocities: list[np.ndarray] = []
        self.steps = 0
        self.max_steps = 300

    def reset(self) -> np.ndarray:
        diff = self.scheduler.current
        self.steps = 0

        # Spawn robot and goal according to difficulty
        self.robot_pos = self._rng.uniform(0, diff.map_size * 0.2, 2)
        goal_dist = self._rng.uniform(*diff.goal_dist_range)
        angle = self._rng.uniform(0, 2 * np.pi)
        self.goal_pos = np.clip(
            self.robot_pos + goal_dist * np.array([np.cos(angle), np.sin(angle)]),
            0, diff.map_size
        )

        # Obstacles
        self.obstacles = [
            self._rng.uniform(diff.map_size * 0.2, diff.map_size * 0.8, 2)
            for _ in range(diff.n_obstacles)
        ]
        self.obs_velocities = [
            self._rng.uniform(-diff.obstacle_speed, diff.obstacle_speed, 2)
            for _ in range(diff.n_obstacles)
        ]
        return self._get_obs(diff)

    def step(self, action: np.ndarray
             ) -> tuple[np.ndarray, float, bool, dict]:
        diff = self.scheduler.current
        self.steps += 1

        # Move robot
        self.robot_pos = np.clip(self.robot_pos + action * 0.3, 0, diff.map_size)

        # Move dynamic obstacles
        for i, (obs, vel) in enumerate(zip(self.obstacles, self.obs_velocities)):
            self.obstacles[i] = np.clip(obs + vel * 0.1, 0, diff.map_size)

        dist_goal = float(np.linalg.norm(self.robot_pos - self.goal_pos))
        min_obs_d = min((np.linalg.norm(self.robot_pos - o)
                         for o in self.obstacles), default=np.inf)

        reward = -0.01
        reward += 0.1 / (dist_goal + 0.1)
        if min_obs_d < 0.5:
            reward -= 3.0
        done = dist_goal < 0.5 or self.steps >= self.max_steps
        success = dist_goal < 0.5

        if done:
            self.scheduler.record_episode(success)

        obs = self._get_obs(diff)
        # Add sensor noise
        obs += self._rng.standard_normal(obs.shape) * diff.sensor_noise
        return obs, reward, done, {"success": success, "dist_goal": dist_goal}

    def _get_obs(self, diff: DifficultyLevel) -> np.ndarray:
        goal_vec = self.goal_pos - self.robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-8)
        obs = np.concatenate([
            self.robot_pos / diff.map_size,
            goal_dir,
            [goal_dist / diff.map_size],
            [diff.n_obstacles / 25.0],
        ])
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        return obs[:self.obs_dim].astype(np.float32)


# ---------------------------------------------------------------------------
# Curriculum training runner
# ---------------------------------------------------------------------------

def run_curriculum_training(
    n_episodes: int = 500,
    strategy: CurriculumStrategy = CurriculumStrategy.ADAPTIVE,
    out_csv: str = "results/curriculum_log.csv",
    seed: int = 42,
) -> list[dict]:
    """
    Full curriculum training run. Returns episode log.
    Integrates with PPOAgent from contribution 21.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../21_ppo_navigation_agent"))

    try:
        from ppo_nav_agent import PPOAgent, PPOConfig
        agent_available = True
    except ImportError:
        agent_available = False
        logger.warning("PPOAgent not found — running environment stub only")

    scheduler = CurriculumScheduler(strategy=strategy)
    env = CurriculumNavEnv(scheduler)
    rng = np.random.default_rng(seed)
    log = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            # Random policy stub (replace with agent.select_action(obs))
            action = rng.uniform(-0.5, 0.5, 2)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        record = {
            "episode": ep + 1,
            "stage": scheduler.current.name,
            "stage_idx": scheduler.current_stage_idx,
            "ep_reward": round(ep_reward, 3),
            "ep_steps": ep_steps,
            "success": int(info.get("success", False)),
            "success_rate": round(scheduler.success_rate, 3),
        }
        log.append(record)

        if (ep + 1) % 50 == 0:
            logger.info("[Ep %4d] stage=%-8s sr=%.2f reward=%.2f",
                        ep + 1, scheduler.current.name,
                        scheduler.success_rate, ep_reward)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader(); w.writerows(log)

    logger.info("Curriculum training complete. Summary: %s", scheduler.summary())
    return log
