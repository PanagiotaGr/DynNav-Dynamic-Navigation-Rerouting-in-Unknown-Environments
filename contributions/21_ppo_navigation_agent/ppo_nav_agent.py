"""
Contribution 21: PPO Navigation Agent
======================================
Proximal Policy Optimization (PPO) agent for autonomous navigation in
unknown environments. The agent learns a policy π(a|s) that maps occupancy
observations to velocity commands, with reward shaping from DynNav's
existing risk and safety modules.

Reward structure:
    r = r_progress - λ_risk * r_risk - λ_col * r_collision + r_goal

Research Question (RQ-PPO): Does RL-based navigation with risk-shaped
rewards outperform classical A* in highly dynamic environments?

References:
    Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    Mirowski et al. (2017) "Learning to Navigate in Complex Environments"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    obs_dim: int = 84          # flattened observation (e.g. 8x8 local grid + goal dir)
    action_dim: int = 2        # (linear_vel, angular_vel) or discrete 5 actions
    hidden_dim: int = 128
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    n_epochs: int = 4           # PPO update epochs per rollout
    batch_size: int = 64
    rollout_len: int = 512
    max_grad_norm: float = 0.5
    # Reward shaping
    lambda_risk: float = 0.5
    lambda_collision: float = 5.0
    goal_reward: float = 10.0
    step_penalty: float = -0.01


# ---------------------------------------------------------------------------
# Neural network stubs (replace with PyTorch nn.Module in production)
# ---------------------------------------------------------------------------

class LinearLayer:
    """Single linear layer with optional activation (numpy stub)."""

    def __init__(self, in_d: int, out_d: int, activation: str = "relu", seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((in_d, out_d)) * np.sqrt(2.0 / in_d)
        self.b = np.zeros(out_d)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W + self.b
        if self.activation == "relu":
            return np.maximum(0, out)
        if self.activation == "tanh":
            return np.tanh(out)
        return out  # linear


class ActorNetwork:
    """Policy network: obs → action mean (continuous) or logits (discrete)."""

    def __init__(self, cfg: PPOConfig):
        self.l1 = LinearLayer(cfg.obs_dim, cfg.hidden_dim, "tanh", seed=1)
        self.l2 = LinearLayer(cfg.hidden_dim, cfg.hidden_dim, "tanh", seed=2)
        self.out = LinearLayer(cfg.hidden_dim, cfg.action_dim, "tanh", seed=3)
        self.log_std = np.full(cfg.action_dim, -0.5)   # learnable log std

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.l1.forward(obs)
        x = self.l2.forward(x)
        mean = self.out.forward(x)
        std = np.exp(self.log_std)
        return mean, std

    def sample(self, obs: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, float]:
        mean, std = self.forward(obs)
        action = mean + std * rng.standard_normal(mean.shape)
        log_prob = float(np.sum(
            -0.5 * ((action - mean) / (std + 1e-8)) ** 2
            - np.log(std + 1e-8) - 0.5 * np.log(2 * np.pi)
        ))
        return action, log_prob


class CriticNetwork:
    """Value network: obs → V(s)."""

    def __init__(self, cfg: PPOConfig):
        self.l1 = LinearLayer(cfg.obs_dim, cfg.hidden_dim, "tanh", seed=4)
        self.l2 = LinearLayer(cfg.hidden_dim, cfg.hidden_dim, "tanh", seed=5)
        self.out = LinearLayer(cfg.hidden_dim, 1, "linear", seed=6)

    def forward(self, obs: np.ndarray) -> float:
        x = self.l1.forward(obs)
        x = self.l2.forward(x)
        return float(self.out.forward(x).squeeze())


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def clear(self):
        self.obs.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs); self.actions.append(action)
        self.log_probs.append(log_prob); self.rewards.append(reward)
        self.values.append(value); self.dones.append(done)

    def compute_gae(self, last_value: float,
                    gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation."""
        T = len(self.rewards)
        advantages = np.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values)
        return advantages, returns


# ---------------------------------------------------------------------------
# Navigation environment (stub)
# ---------------------------------------------------------------------------

class NavEnv:
    """
    Minimal navigation environment compatible with PPO.
    Replace with Gazebo/ROS2 bridge for real-world deployment.
    """

    def __init__(self, cfg: PPOConfig, grid_size: int = 16):
        self.cfg = cfg
        self.grid_size = grid_size
        self._rng = np.random.default_rng()
        self.robot_pos = np.array([0.0, 0.0])
        self.goal_pos = np.array([5.0, 5.0])
        self.obstacles: list[np.ndarray] = []
        self.steps = 0
        self.max_steps = 200

    def reset(self) -> np.ndarray:
        self.robot_pos = self._rng.uniform(0, 2, 2)
        self.goal_pos = self._rng.uniform(6, 8, 2)
        n_obs = self._rng.integers(3, 8)
        self.obstacles = [self._rng.uniform(2, 7, 2) for _ in range(n_obs)]
        self.steps = 0
        return self._get_obs()

    def step(self, action: np.ndarray
             ) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        self.robot_pos = np.clip(self.robot_pos + action * 0.3, 0, 10)

        dist_goal = float(np.linalg.norm(self.robot_pos - self.goal_pos))
        min_obs_dist = min(
            (np.linalg.norm(self.robot_pos - o) for o in self.obstacles),
            default=np.inf
        )

        # Reward shaping
        reward = self.cfg.step_penalty
        reward += 0.1 * (1.0 / (dist_goal + 0.1))        # progress
        reward -= self.cfg.lambda_risk * max(0, 1.5 - min_obs_dist)
        if min_obs_dist < 0.4:
            reward -= self.cfg.lambda_collision
        if dist_goal < 0.5:
            reward += self.cfg.goal_reward

        done = dist_goal < 0.5 or self.steps >= self.max_steps
        info = {"dist_goal": dist_goal, "min_obs_dist": min_obs_dist}
        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Flat observation: robot_pos, goal_dir, goal_dist, 8-dir clearances."""
        goal_vec = self.goal_pos - self.robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-8)

        clearances = []
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            d = np.array([np.cos(angle), np.sin(angle)])
            min_c = min(
                (np.linalg.norm(self.robot_pos + d * t - o)
                 for o in self.obstacles for t in [0.5, 1.0, 1.5]),
                default=2.0
            )
            clearances.append(min(min_c, 2.0) / 2.0)

        obs = np.concatenate([
            self.robot_pos / 10.0,
            goal_dir,
            [goal_dist / 10.0],
            clearances,
        ])
        # Pad/truncate to obs_dim
        if len(obs) < self.cfg.obs_dim:
            obs = np.pad(obs, (0, self.cfg.obs_dim - len(obs)))
        return obs[:self.cfg.obs_dim].astype(np.float32)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent for DynNav navigation.
    Uses numpy-based networks; swap for PyTorch for GPU training.
    """

    def __init__(self, cfg: PPOConfig | None = None):
        self.cfg = cfg or PPOConfig()
        self.actor = ActorNetwork(self.cfg)
        self.critic = CriticNetwork(self.cfg)
        self.buffer = RolloutBuffer()
        self._rng = np.random.default_rng(42)
        self.training_log: list[dict] = []

    def select_action(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        action, log_prob = self.actor.sample(obs, self._rng)
        value = self.critic.forward(obs)
        return action, log_prob, value

    def collect_rollout(self, env: NavEnv) -> dict:
        """Collect one rollout of rollout_len steps."""
        self.buffer.clear()
        obs = env.reset()
        ep_rewards, ep_lengths = [], []
        ep_r, ep_len = 0.0, 0

        for _ in range(self.cfg.rollout_len):
            action, log_prob, value = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            self.buffer.add(obs, action, log_prob, reward, value, float(done))
            ep_r += reward
            ep_len += 1
            obs = next_obs
            if done:
                ep_rewards.append(ep_r)
                ep_lengths.append(ep_len)
                ep_r, ep_len = 0.0, 0
                obs = env.reset()

        last_val = self.critic.forward(obs)
        advantages, returns = self.buffer.compute_gae(
            last_val, self.cfg.gamma, self.cfg.gae_lambda
        )
        return {
            "mean_ep_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "n_episodes": len(ep_rewards),
            "advantages_mean": float(advantages.mean()),
            "returns_mean": float(returns.mean()),
        }

    def update(self, advantages: np.ndarray,
               returns: np.ndarray) -> dict:
        """
        PPO clipped surrogate objective update (numpy stub).
        In production: compute actual gradients with PyTorch autograd.
        """
        # Normalise advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Stub: log approximate losses without backprop
        policy_loss = float(-np.mean(adv))
        value_loss = float(np.mean((returns - np.array(self.buffer.values)) ** 2))
        entropy = float(np.mean(self.actor.log_std))
        total_loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
        return {
            "policy_loss": round(policy_loss, 5),
            "value_loss": round(value_loss, 5),
            "entropy": round(entropy, 5),
            "total_loss": round(total_loss, 5),
        }

    def train(self, env: NavEnv, n_updates: int = 50) -> list[dict]:
        """Main training loop."""
        for update in range(n_updates):
            stats = self.collect_rollout(env)
            advantages, returns = self.buffer.compute_gae(
                0.0, self.cfg.gamma, self.cfg.gae_lambda
            )
            loss_stats = self.update(advantages, returns)
            record = {"update": update + 1, **stats, **loss_stats}
            self.training_log.append(record)
            if (update + 1) % 10 == 0:
                logger.info(
                    "Update %3d | ep_r=%.2f | ep_len=%.1f | v_loss=%.4f",
                    update + 1, stats["mean_ep_reward"],
                    stats["mean_ep_length"], loss_stats["value_loss"]
                )
        return self.training_log
