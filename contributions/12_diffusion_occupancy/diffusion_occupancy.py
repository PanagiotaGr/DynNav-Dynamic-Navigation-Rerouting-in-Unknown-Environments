"""
Contribution 12: Diffusion Occupancy Maps
==========================================
Score-based diffusion model for probabilistic occupancy prediction.
Instead of a single deterministic occupancy grid, the model produces a
*distribution* over future occupancy maps by running reverse diffusion
from Gaussian noise conditioned on the current sensor history.

Research Question (RQ-Diff): Does a diffusion-based occupancy model
reduce collision rate and improve risk estimation compared to deterministic
or Kalman-filter-based predictions?

References:
    Song et al. (2020) "Score-Based Generative Modeling through SDEs"
    Luo & Hu (2021) "Diffusion Probabilistic Models for 3D Point Cloud Generation"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiffusionOccupancyConfig:
    grid_h: int = 64
    grid_w: int = 64
    T: int = 50                    # diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    n_samples: int = 10            # particles for Monte-Carlo risk
    history_len: int = 5           # frames of occupancy history as conditioning
    device: str = "cpu"            # "cuda" if available


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int,
                          s: float = 0.008) -> tuple[np.ndarray, np.ndarray]:
    """Cosine beta schedule (Nichol & Dhariwal 2021)."""
    steps = np.arange(T + 1) / T
    alpha_bar = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    betas = np.clip(betas, 0, 0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return betas, alphas_cumprod


# ---------------------------------------------------------------------------
# Lightweight U-Net score network (numpy-only, replace with PyTorch in prod)
# ---------------------------------------------------------------------------

class ScoreNetwork:
    """
    Minimal score-estimator s_θ(x_t, t, cond).
    Production: replace with a proper U-Net (e.g. diffusers UNet2DModel).
    Here we use a single-layer MLP stub for correctness testing.
    """

    def __init__(self, grid_h: int, grid_w: int):
        self.h = grid_h
        self.w = grid_w
        d = grid_h * grid_w
        rng = np.random.default_rng(0)
        self.W1 = rng.standard_normal((d + 1, 128)) * 0.01   # +1 for timestep
        self.W2 = rng.standard_normal((128, d)) * 0.01

    def forward(self, x: np.ndarray, t: int,
                cond: np.ndarray) -> np.ndarray:
        """
        x    : (H, W) noisy occupancy
        t    : diffusion timestep
        cond : (H, W) conditioning (last observed occupancy)
        Returns score estimate (H, W).
        """
        flat = x.flatten()
        inp = np.append(flat, t / 1000.0)
        h = np.tanh(inp @ self.W1)
        score = (h @ self.W2).reshape(self.h, self.w)
        return score


# ---------------------------------------------------------------------------
# Diffusion Occupancy Predictor
# ---------------------------------------------------------------------------

class DiffusionOccupancyPredictor:
    """
    Generates N future occupancy map samples via DDPM reverse diffusion,
    then derives risk metrics from the ensemble.
    """

    def __init__(self, config: DiffusionOccupancyConfig | None = None):
        self.cfg = config or DiffusionOccupancyConfig()
        self.betas, self.alphas_bar = cosine_beta_schedule(self.cfg.T)
        self.alphas = 1.0 - self.betas
        self.score_net = ScoreNetwork(self.cfg.grid_h, self.cfg.grid_w)
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Forward (training) pass  — adds noise
    # ------------------------------------------------------------------

    def q_sample(self, x0: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Diffuse x0 to timestep t.
        Returns (x_t, noise).
        """
        noise = self._rng.standard_normal(x0.shape)
        alpha_bar_t = self.alphas_bar[t]
        x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Reverse (inference) pass  — denoises
    # ------------------------------------------------------------------

    def p_sample_step(self, x_t: np.ndarray, t: int,
                      cond: np.ndarray) -> np.ndarray:
        """One reverse step: x_t → x_{t-1}."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]

        score = self.score_net.forward(x_t, t, cond)
        # DDPM mean (simplified)
        mean = (1 / np.sqrt(alpha_t)) * (
            x_t - (beta_t / np.sqrt(1 - alpha_bar_t)) * score
        )
        if t > 0:
            noise = self._rng.standard_normal(x_t.shape)
            x_prev = mean + np.sqrt(beta_t) * noise
        else:
            x_prev = mean
        return x_prev

    def sample(self, cond: np.ndarray) -> np.ndarray:
        """Full reverse chain: noise → predicted occupancy map."""
        x = self._rng.standard_normal((self.cfg.grid_h, self.cfg.grid_w))
        for t in reversed(range(self.cfg.T)):
            x = self.p_sample_step(x, t, cond)
        return np.clip(x, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Risk metrics from ensemble
    # ------------------------------------------------------------------

    def predict_risk(self, history: list[np.ndarray]
                     ) -> dict[str, np.ndarray]:
        """
        Run N reverse diffusions conditioned on the last observed occupancy.
        Returns per-cell occupancy mean, std, and CVaR risk map.

        Parameters
        ----------
        history : list of (H, W) binary occupancy grids, newest last
        """
        if not history:
            raise ValueError("Need at least one occupancy frame in history")

        cond = history[-1].astype(float)
        samples = np.stack(
            [self.sample(cond) for _ in range(self.cfg.n_samples)], axis=0
        )  # (N, H, W)

        mean_map = samples.mean(axis=0)
        std_map = samples.std(axis=0)

        # CVaR at α=0.95: expected occupancy in worst 5% of samples
        alpha = 0.95
        k = max(1, int((1 - alpha) * self.cfg.n_samples))
        sorted_s = np.sort(samples, axis=0)   # ascending along sample axis
        cvar_map = sorted_s[-k:].mean(axis=0)

        logger.debug(
            "Diffusion risk map | mean=%.3f std=%.3f cvar=%.3f",
            mean_map.mean(), std_map.mean(), cvar_map.mean()
        )
        return {
            "mean": mean_map,
            "std": std_map,
            "cvar_95": cvar_map,
            "samples": samples,
        }

    def risk_weighted_cost(self, path: list[tuple[int, int]],
                           risk_maps: dict[str, np.ndarray],
                           lambda_risk: float = 2.0) -> float:
        """
        Compute the risk-augmented path cost.
        cost = path_length + lambda_risk * sum(cvar along path cells)
        """
        cvar = risk_maps["cvar_95"]
        risk_sum = sum(
            cvar[r, c] for r, c in path
            if 0 <= r < cvar.shape[0] and 0 <= c < cvar.shape[1]
        )
        return float(len(path)) + lambda_risk * risk_sum
