"""
Contribution 23: 3D Gaussian Splatting Mapper
===============================================
Builds an implicit 3D map from RGB(D) frames using Gaussian Splatting
principles. Each Gaussian represents a local scene element with:
    - position (μ ∈ R³)
    - covariance (Σ ∈ R³ˣ³) — shape and orientation
    - opacity (α ∈ [0,1])
    - colour (c ∈ R³)

The map supports:
    - Incremental update as new frames arrive
    - Uncertainty estimation from Gaussian density
    - Navigation-ready occupancy extraction (project Gaussians onto 2D grid)
    - Frontier detection for exploration planning

Research Question (RQ-3DGS): Does a Gaussian-Splatting-based 3D map
provide better uncertainty estimates for navigation than a 2D occupancy grid?

References:
    Kerbl et al. (2023) "3D Gaussian Splatting for Real-Time Novel View Synthesis"
    Matsuki et al. (2024) "Gaussian Splatting SLAM"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian primitive
# ---------------------------------------------------------------------------

@dataclass
class Gaussian3D:
    mu: np.ndarray           # (3,) position
    cov: np.ndarray          # (3,3) covariance
    alpha: float             # opacity [0,1]
    color: np.ndarray        # (3,) RGB [0,1]
    id: int = -1
    confidence: float = 1.0  # decreases if not reinforced by new frames

    def mahalanobis(self, point: np.ndarray) -> float:
        """Mahalanobis distance from point to this Gaussian's centre."""
        diff = point - self.mu
        try:
            inv_cov = np.linalg.inv(self.cov + np.eye(3) * 1e-6)
            return float(np.sqrt(diff @ inv_cov @ diff))
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(diff))

    def volume(self) -> float:
        """Approximate volume of the 3-sigma ellipsoid."""
        eigvals = np.linalg.eigvalsh(self.cov)
        return float((4 / 3) * np.pi * np.prod(np.sqrt(np.abs(eigvals)) * 3))

    def project_2d(self, axis: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Project onto a 2D plane by dropping one axis."""
        axes = [i for i in range(3) if i != axis]
        mu_2d = self.mu[axes]
        cov_2d = self.cov[np.ix_(axes, axes)]
        return mu_2d, cov_2d


# ---------------------------------------------------------------------------
# Gaussian Splatting Map
# ---------------------------------------------------------------------------

@dataclass
class GSMapConfig:
    merge_threshold: float = 0.3      # Mahalanobis dist for merging
    max_gaussians: int = 5000
    decay_rate: float = 0.995         # confidence decay per frame
    min_confidence: float = 0.1       # prune below this
    grid_resolution: float = 0.1      # metres per cell for 2D projection
    grid_size: tuple[int, int] = (100, 100)
    occupancy_alpha_threshold: float = 0.3


class GaussianSplattingMap:
    """
    Incremental 3D Gaussian Splatting map for robot navigation.
    Supports uncertainty-aware occupancy extraction for DynNav planners.
    """

    def __init__(self, config: GSMapConfig | None = None):
        self.cfg = config or GSMapConfig()
        self.gaussians: list[Gaussian3D] = []
        self._next_id = 0
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Map update
    # ------------------------------------------------------------------

    def add_frame(self, points: np.ndarray,
                  colors: Optional[np.ndarray] = None,
                  pose: Optional[np.ndarray] = None) -> int:
        """
        Integrate a new point cloud frame into the Gaussian map.

        Parameters
        ----------
        points  : (N, 3) array of 3D points in camera frame
        colors  : (N, 3) RGB colours [0,1], optional
        pose    : (4, 4) camera-to-world transform, optional

        Returns number of new Gaussians added.
        """
        if pose is not None:
            R = pose[:3, :3]
            t = pose[:3, 3]
            points = (R @ points.T).T + t

        if colors is None:
            colors = np.ones((len(points), 3)) * 0.5

        self.frame_count += 1
        self._decay_confidence()
        n_added = 0

        for pt, col in zip(points, colors):
            merged = self._try_merge(pt)
            if not merged:
                if len(self.gaussians) < self.cfg.max_gaussians:
                    self._add_gaussian(pt, col)
                    n_added += 1

        self._prune()
        logger.debug("Frame %d: +%d Gaussians, total=%d",
                     self.frame_count, n_added, len(self.gaussians))
        return n_added

    def _add_gaussian(self, point: np.ndarray,
                      color: np.ndarray) -> Gaussian3D:
        g = Gaussian3D(
            mu=point.copy(),
            cov=np.eye(3) * 0.01,
            alpha=0.8,
            color=np.clip(color, 0, 1),
            id=self._next_id,
        )
        self._next_id += 1
        self.gaussians.append(g)
        return g

    def _try_merge(self, point: np.ndarray) -> bool:
        """
        If a nearby Gaussian exists, update it (EKF-style) instead of
        adding a new one. Returns True if merged.
        """
        for g in self.gaussians:
            if g.mahalanobis(point) < self.cfg.merge_threshold:
                # Kalman-style update: shift mean toward new point
                alpha = 0.1
                g.mu = (1 - alpha) * g.mu + alpha * point
                g.cov = (1 - alpha) * g.cov + alpha * np.outer(
                    point - g.mu, point - g.mu
                )
                g.confidence = min(1.0, g.confidence + 0.05)
                return True
        return False

    def _decay_confidence(self):
        for g in self.gaussians:
            g.confidence *= self.cfg.decay_rate

    def _prune(self):
        before = len(self.gaussians)
        self.gaussians = [g for g in self.gaussians
                          if g.confidence >= self.cfg.min_confidence]
        pruned = before - len(self.gaussians)
        if pruned > 0:
            logger.debug("Pruned %d low-confidence Gaussians", pruned)

    # ------------------------------------------------------------------
    # Occupancy extraction (2D projection for DynNav planner)
    # ------------------------------------------------------------------

    def to_occupancy_grid(self, z_min: float = 0.1,
                           z_max: float = 2.0) -> np.ndarray:
        """
        Project Gaussians onto a 2D occupancy grid.
        Returns (H, W) float array in [0, 1].
        """
        H, W = self.cfg.grid_size
        grid = np.zeros((H, W))
        res = self.cfg.grid_resolution

        for g in self.gaussians:
            if not (z_min <= g.mu[2] <= z_max):
                continue
            if g.alpha < self.cfg.occupancy_alpha_threshold:
                continue

            cx = int(g.mu[0] / res) + W // 2
            cy = int(g.mu[1] / res) + H // 2
            if not (0 <= cx < W and 0 <= cy < H):
                continue

            # Spread Gaussian influence over nearby cells
            sigma = max(1, int(np.sqrt(g.cov[0, 0] + g.cov[1, 1]) / res))
            for dx in range(-sigma, sigma + 1):
                for dy in range(-sigma, sigma + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        contribution = g.alpha * np.exp(-0.5 * (dist / (sigma + 1e-8)) ** 2)
                        grid[ny, nx] = min(1.0, grid[ny, nx] + contribution)

        return grid

    def uncertainty_map(self) -> np.ndarray:
        """
        Per-cell uncertainty: regions with few/low-confidence Gaussians
        are high uncertainty. Returns (H, W) in [0, 1].
        """
        occ = self.to_occupancy_grid()
        conf_grid = np.zeros_like(occ)

        H, W = self.cfg.grid_size
        res = self.cfg.grid_resolution

        for g in self.gaussians:
            cx = int(g.mu[0] / res) + W // 2
            cy = int(g.mu[1] / res) + H // 2
            if 0 <= cx < W and 0 <= cy < H:
                conf_grid[cy, cx] = max(conf_grid[cy, cx], g.confidence)

        # Uncertainty = 1 - confidence (unknown areas = high uncertainty)
        uncertainty = 1.0 - np.clip(conf_grid, 0, 1)
        return uncertainty

    def frontier_cells(self, uncertainty_threshold: float = 0.7
                        ) -> list[tuple[int, int]]:
        """
        Frontier detection: cells at the boundary of known/unknown space.
        Returns list of (row, col) frontier cell indices.
        """
        unc = self.uncertainty_map()
        occ = self.to_occupancy_grid()
        H, W = self.cfg.grid_size
        frontiers = []

        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if occ[r, c] > 0.1:
                    continue   # occupied — not a frontier
                neighbours_unc = [
                    unc[r + dr, c + dc]
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                ]
                if any(u > uncertainty_threshold for u in neighbours_unc):
                    frontiers.append((r, c))

        return frontiers

    def stats(self) -> dict:
        if not self.gaussians:
            return {"n_gaussians": 0}
        alphas = [g.alpha for g in self.gaussians]
        confs = [g.confidence for g in self.gaussians]
        return {
            "n_gaussians": len(self.gaussians),
            "mean_alpha": round(float(np.mean(alphas)), 3),
            "mean_confidence": round(float(np.mean(confs)), 3),
            "frames_integrated": self.frame_count,
        }
