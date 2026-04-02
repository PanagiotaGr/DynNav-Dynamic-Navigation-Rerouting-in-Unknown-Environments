"""
Contribution 13: Latent World Model (Dreamer-v3 style)
=======================================================
Implements a Recurrent State Space Model (RSSM) that learns a compact latent
representation of the environment and performs 'mental rollouts' — imagined
future trajectories in latent space — before committing to any real action.

This is especially powerful combined with Contribution 04
(irreversibility / returnability): the robot simulates whether a sequence of
actions is recoverable *before* executing it.

Research Question (RQ-WM): Do mental rollouts in latent space reduce the
frequency of irreversible failures compared to reactive replanning alone?

References:
    Hafner et al. (2023) "Mastering Diverse Domains through World Models"
    (DreamerV3, arXiv:2301.04104)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RSSMConfig:
    obs_dim: int = 64 * 64          # flattened observation size
    action_dim: int = 2             # (vx, vy) or (linear, angular)
    latent_dim: int = 32            # stochastic latent z
    hidden_dim: int = 64            # deterministic recurrent state h
    horizon: int = 12               # imagination horizon (steps)
    n_rollouts: int = 8             # parallel imagined trajectories
    lr: float = 1e-3
    gamma: float = 0.99             # discount for imagined returns
    irreversibility_penalty: float = 5.0


# ---------------------------------------------------------------------------
# Minimal RSSM (numpy; swap for PyTorch GRU + MLP in production)
# ---------------------------------------------------------------------------

class RSSM:
    """
    Recurrent State Space Model with:
        - Deterministic path:  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        - Stochastic path:     z_t ~ q(z | h_t, o_t)   [posterior]
                               z_t ~ p(z | h_t)          [prior]
        - Reward predictor:    r_hat = g(h_t, z_t)
    """

    def __init__(self, cfg: RSSMConfig):
        self.cfg = cfg
        rng = np.random.default_rng(0)
        D = cfg.hidden_dim

        # Transition: [h, z, a] → h'
        self.W_trans = rng.standard_normal(
            (D + cfg.latent_dim + cfg.action_dim, D)
        ) * 0.01

        # Prior: h → z_mean, z_log_std
        self.W_prior_mean = rng.standard_normal((D, cfg.latent_dim)) * 0.01
        self.W_prior_lsd = rng.standard_normal((D, cfg.latent_dim)) * 0.01

        # Posterior: [h, o] → z_mean, z_log_std
        self.W_post_mean = rng.standard_normal(
            (D + cfg.obs_dim, cfg.latent_dim)
        ) * 0.01
        self.W_post_lsd = rng.standard_normal(
            (D + cfg.obs_dim, cfg.latent_dim)
        ) * 0.01

        # Reward predictor: [h, z] → r
        self.W_reward = rng.standard_normal(
            (D + cfg.latent_dim, 1)
        ) * 0.01

        self._rng = np.random.default_rng(1)

    # ------------------------------------------------------------------
    # Core RSSM steps
    # ------------------------------------------------------------------

    def recurrent_step(self, h: np.ndarray, z: np.ndarray,
                        action: np.ndarray) -> np.ndarray:
        """Deterministic path: GRU-like update (tanh linear stub)."""
        inp = np.concatenate([h, z, action])
        h_next = np.tanh(inp @ self.W_trans)
        return h_next

    def prior(self, h: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """p(z | h)  → (z_sample, z_mean, z_log_std)"""
        mean = h @ self.W_prior_mean
        lsd = np.clip(h @ self.W_prior_lsd, -4, 4)
        z = mean + np.exp(lsd) * self._rng.standard_normal(mean.shape)
        return z, mean, lsd

    def posterior(self, h: np.ndarray,
                  obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """q(z | h, o) → (z_sample, z_mean, z_log_std)"""
        inp = np.concatenate([h, obs])
        mean = inp @ self.W_post_mean
        lsd = np.clip(inp @ self.W_post_lsd, -4, 4)
        z = mean + np.exp(lsd) * self._rng.standard_normal(mean.shape)
        return z, mean, lsd

    def predict_reward(self, h: np.ndarray, z: np.ndarray) -> float:
        """r_hat = g(h, z)"""
        inp = np.concatenate([h, z])
        return float((inp @ self.W_reward).squeeze())

    # ------------------------------------------------------------------
    # Mental rollout
    # ------------------------------------------------------------------

    def imagine_rollout(self, h0: np.ndarray, z0: np.ndarray,
                         action_sequence: list[np.ndarray]
                         ) -> dict[str, list]:
        """
        Simulate one trajectory in latent space.
        Returns predicted rewards and latent states.
        """
        h, z = h0.copy(), z0.copy()
        rewards, states = [], []

        for action in action_sequence:
            h = self.recurrent_step(h, z, action)
            z, _, _ = self.prior(h)
            r = self.predict_reward(h, z)
            rewards.append(r)
            states.append((h.copy(), z.copy()))

        return {"rewards": rewards, "states": states}

    def discounted_return(self, rewards: list[float]) -> float:
        """Compute γ-discounted sum of imagined rewards."""
        G, discount = 0.0, 1.0
        for r in rewards:
            G += discount * r
            discount *= self.cfg.gamma
        return G


# ---------------------------------------------------------------------------
# World-Model-based Planner
# ---------------------------------------------------------------------------

class WorldModelPlanner:
    """
    Uses imagined rollouts to score candidate action sequences
    and select the best plan before execution.
    """

    def __init__(self, rssm: RSSM | None = None,
                 config: RSSMConfig | None = None):
        self.cfg = config or RSSMConfig()
        self.rssm = rssm or RSSM(self.cfg)
        self._h = np.zeros(self.cfg.hidden_dim)
        self._z = np.zeros(self.cfg.latent_dim)

    def update_belief(self, obs: np.ndarray, action: np.ndarray) -> None:
        """Step the recurrent state forward with a real observation."""
        obs_flat = obs.flatten()[: self.cfg.obs_dim]
        self._h = self.rssm.recurrent_step(self._h, self._z, action)
        self._z, _, _ = self.rssm.posterior(self._h, obs_flat)

    def select_best_action_sequence(
        self,
        candidate_sequences: list[list[np.ndarray]],
        irreversibility_checker: callable | None = None,
    ) -> tuple[list[np.ndarray], float]:
        """
        Score multiple candidate action sequences via imagined rollouts.
        Applies irreversibility penalty if checker provided.
        Returns (best_sequence, best_return).
        """
        best_seq, best_return = None, -np.inf

        for seq in candidate_sequences:
            rollout = self.rssm.imagine_rollout(self._h, self._z, seq)
            G = self.rssm.discounted_return(rollout["rewards"])

            # Penalise irreversible trajectories
            if irreversibility_checker is not None:
                final_state = rollout["states"][-1]
                if not irreversibility_checker(final_state):
                    G -= self.cfg.irreversibility_penalty
                    logger.debug("Irreversibility penalty applied to sequence")

            if G > best_return:
                best_return = G
                best_seq = seq

        logger.info("Best imagined return: %.3f", best_return)
        return best_seq, best_return

    def generate_random_sequences(self, n: int,
                                   horizon: int | None = None
                                   ) -> list[list[np.ndarray]]:
        """Sample random action sequences for CEM / random shooting."""
        H = horizon or self.cfg.horizon
        rng = np.random.default_rng()
        return [
            [rng.uniform(-1, 1, self.cfg.action_dim) for _ in range(H)]
            for _ in range(n)
        ]
