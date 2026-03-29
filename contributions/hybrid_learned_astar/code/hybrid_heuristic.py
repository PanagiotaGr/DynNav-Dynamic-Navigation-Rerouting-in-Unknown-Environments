from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
import torch

from contributions.learned_uncertainty_astar.code.uncertainty_heuristic_net import (
    UncertaintyHeuristicNet,
)
from contributions.learned_uncertainty_astar.code.uncertainty_astar import extract_features


@dataclass
class HybridHeuristicStats:
    queries: int = 0
    fallback_count: int = 0
    smooth_blend_count: int = 0
    total_std: float = 0.0

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.queries if self.queries > 0 else 0.0

    @property
    def mean_std(self) -> float:
        return self.total_std / self.queries if self.queries > 0 else 0.0


class EuclideanHeuristic:
    def __call__(self, node: Tuple[int, int], goal: Tuple[int, int], grid: np.ndarray) -> float:
        return math.dist(node, goal)


class LearnedHeuristicWrapper:
    def __init__(self, model: UncertaintyHeuristicNet, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        node: Tuple[int, int],
        goal: Tuple[int, int],
        grid: np.ndarray,
    ) -> Tuple[float, float]:
        feats = extract_features(node=node, goal=goal, grid=grid)
        x = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std = self.model(x)
        return float(mean.item()), float(std.item())


class HybridHeuristic:
    """
    Hard switch:
        if std <= tau: use learned mean
        else: use admissible heuristic

    Optional smooth blending:
        h = (1 - w) * h_mean + w * h_adm
        where w = clip((std - tau_low)/(tau_high - tau_low), 0, 1)
    """

    def __init__(
        self,
        learned_model: UncertaintyHeuristicNet,
        tau: float = 2.0,
        device: str = "cpu",
        use_smooth_blending: bool = False,
        tau_low: float = 1.5,
        tau_high: float = 2.5,
    ):
        self.learned = LearnedHeuristicWrapper(learned_model, device=device)
        self.admissible = EuclideanHeuristic()
        self.tau = tau
        self.use_smooth_blending = use_smooth_blending
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.stats = HybridHeuristicStats()

    def reset_stats(self) -> None:
        self.stats = HybridHeuristicStats()

    def __call__(
        self,
        node: Tuple[int, int],
        goal: Tuple[int, int],
        grid: np.ndarray,
    ) -> float:
        h_mean, h_std = self.learned.predict(node, goal, grid)
        h_adm = self.admissible(node, goal, grid)

        self.stats.queries += 1
        self.stats.total_std += h_std

        if self.use_smooth_blending:
            if self.tau_high <= self.tau_low:
                raise ValueError("tau_high must be greater than tau_low")

            w = (h_std - self.tau_low) / (self.tau_high - self.tau_low)
            w = float(np.clip(w, 0.0, 1.0))

            if w > 0.0:
                self.stats.smooth_blend_count += 1
            if w >= 1.0:
                self.stats.fallback_count += 1

            return (1.0 - w) * h_mean + w * h_adm

        if h_std <= self.tau:
            return h_mean

        self.stats.fallback_count += 1
        return h_adm
