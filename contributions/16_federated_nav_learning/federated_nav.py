"""
Contribution 16: Federated Navigation Learning
===============================================
Privacy-preserving multi-robot learning: each robot trains a local navigation
model on its own experience, shares only parameter updates (gradients / model
deltas), and receives a globally aggregated model — without exposing raw
sensor data or trajectories.

Implements FedAvg (McMahan et al. 2017) with optional differential-privacy
noise injection (Gaussian mechanism) and contribution-weighted aggregation.

Research Question (RQ-Fed): Does federated learning of navigation heuristics
across a heterogeneous robot fleet improve generalisation to unseen environments
without sharing private map data?

References:
    McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" (FedAvg)
    Dwork & Roth (2014) "The Algorithmic Foundations of Differential Privacy"
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FedNavConfig:
    n_robots: int = 6
    local_epochs: int = 5
    global_rounds: int = 20
    lr: float = 0.01
    dp_epsilon: Optional[float] = None    # None = no DP noise
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    aggregation: str = "weighted"         # "uniform" | "weighted"
    input_dim: int = 16                   # feature vector size
    output_dim: int = 2                   # (vx, vy) action


# ---------------------------------------------------------------------------
# Minimal linear navigation model (replace with MLP / CNN in prod)
# ---------------------------------------------------------------------------

@dataclass
class NavModel:
    """Simple linear policy: action = W @ features + b"""
    W: np.ndarray
    b: np.ndarray

    @classmethod
    def random_init(cls, in_d: int, out_d: int, seed: int = 0) -> "NavModel":
        rng = np.random.default_rng(seed)
        return cls(
            W=rng.standard_normal((out_d, in_d)) * 0.01,
            b=np.zeros(out_d),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(self.W @ features + self.b)

    def get_params(self) -> dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        self.W = params["W"].copy()
        self.b = params["b"].copy()

    def clone(self) -> "NavModel":
        return NavModel(self.W.copy(), self.b.copy())


# ---------------------------------------------------------------------------
# Robot (local client)
# ---------------------------------------------------------------------------

class FederatedRobotClient:
    """
    Represents one robot that trains locally on its own experience buffer
    and returns model updates to the server.
    """

    def __init__(self, robot_id: int, model: NavModel, cfg: FedNavConfig):
        self.robot_id = robot_id
        self.model = model.clone()
        self.cfg = cfg
        self._experience: list[tuple[np.ndarray, np.ndarray]] = []
        self._rng = np.random.default_rng(robot_id)

    def collect_experience(self, n_steps: int = 50) -> None:
        """Simulate local data collection (stub: random features & targets)."""
        features = self._rng.standard_normal((n_steps, self.cfg.input_dim))
        # Simulated expert actions (e.g. from a local planner)
        targets = np.tanh(self._rng.standard_normal((n_steps, self.cfg.output_dim)))
        self._experience = list(zip(features, targets))
        logger.debug("[Robot %d] Collected %d samples", self.robot_id, n_steps)

    def local_train(self) -> dict[str, np.ndarray]:
        """Run SGD for local_epochs; return updated parameters."""
        if not self._experience:
            raise RuntimeError(f"Robot {self.robot_id}: no experience collected")

        for _ in range(self.cfg.local_epochs):
            for feat, tgt in self._experience:
                pred = self.model.predict(feat)
                err = pred - tgt  # MSE gradient
                d_tanh = (1 - pred ** 2)                          # (out_d,)
                grad_W = np.outer(d_tanh * err, feat)             # (out_d, in_d)
                grad_b = d_tanh * err                             # (out_d,)
                self.model.W -= self.cfg.lr * grad_W
                self.model.b -= self.cfg.lr * grad_b

        params = self.model.get_params()
        if self.cfg.dp_epsilon is not None:
            params = self._add_dp_noise(params)

        return params

    def _add_dp_noise(self, params: dict[str, np.ndarray]
                      ) -> dict[str, np.ndarray]:
        """Gaussian mechanism for (ε, δ)-differential privacy."""
        sigma = (
            self.cfg.dp_clip_norm
            * np.sqrt(2 * np.log(1.25 / self.cfg.dp_delta))
            / self.cfg.dp_epsilon
        )
        rng = np.random.default_rng()
        noisy = {}
        for k, v in params.items():
            # Clip gradient norm
            norm = np.linalg.norm(v)
            clipped = v * min(1.0, self.cfg.dp_clip_norm / (norm + 1e-8))
            noisy[k] = clipped + rng.standard_normal(v.shape) * sigma
        logger.debug("[Robot %d] DP noise σ=%.4f applied", self.robot_id, sigma)
        return noisy

    @property
    def n_samples(self) -> int:
        return len(self._experience)


# ---------------------------------------------------------------------------
# Federated Server
# ---------------------------------------------------------------------------

class FederatedServer:
    """
    Aggregates model updates from robot clients using FedAvg (weighted or
    uniform) and broadcasts the new global model.
    """

    def __init__(self, global_model: NavModel, cfg: FedNavConfig):
        self.global_model = global_model
        self.cfg = cfg
        self.round_history: list[dict] = []

    def aggregate(self, client_updates: list[tuple[dict[str, np.ndarray], int]]
                  ) -> dict[str, np.ndarray]:
        """
        FedAvg aggregation.
        Parameters
        ----------
        client_updates : list of (params_dict, n_samples)
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")

        total_n = sum(n for _, n in client_updates)
        weights = (
            [n / total_n for _, n in client_updates]
            if self.cfg.aggregation == "weighted"
            else [1.0 / len(client_updates)] * len(client_updates)
        )

        agg: dict[str, np.ndarray] = {}
        for (params, _), w in zip(client_updates, weights):
            for k, v in params.items():
                agg[k] = agg.get(k, np.zeros_like(v)) + w * v

        return agg

    def run_round(self, clients: list[FederatedRobotClient]) -> dict:
        """Execute one federated round: collect → train → aggregate → broadcast."""
        updates = []
        for client in clients:
            client.model.set_params(self.global_model.get_params())
            client.collect_experience()
            params = client.local_train()
            updates.append((params, client.n_samples))

        new_params = self.aggregate(updates)
        self.global_model.set_params(new_params)

        # Evaluate on a held-out synthetic validation set
        rng = np.random.default_rng(999)
        val_X = rng.standard_normal((100, self.cfg.input_dim))
        val_Y = np.tanh(rng.standard_normal((100, self.cfg.output_dim)))
        preds = np.stack([self.global_model.predict(x) for x in val_X])
        val_loss = float(np.mean((preds - val_Y) ** 2))

        record = {
            "round": len(self.round_history) + 1,
            "val_mse": round(val_loss, 6),
            "n_clients": len(clients),
            "total_samples": sum(n for _, n in updates),
        }
        self.round_history.append(record)
        logger.info("Fed round %d | val_MSE=%.4f", record["round"], val_loss)
        return record

    def run_training(self, clients: list[FederatedRobotClient]
                     ) -> list[dict]:
        """Run all global rounds."""
        for _ in range(self.cfg.global_rounds):
            self.run_round(clients)
        return self.round_history
