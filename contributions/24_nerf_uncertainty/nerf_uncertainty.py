"""
Contribution 24: NeRF-based Uncertainty Maps
============================================
Uses Neural Radiance Field (NeRF) rendering confidence as a proxy for
spatial uncertainty in unexplored regions. Low rendering confidence
(high photometric error) indicates uncertain / unobserved areas.

This uncertainty map feeds directly into:
    - Contribution 12 (Diffusion Occupancy) as prior
    - Contribution 23 (Gaussian Splatting) for frontier weighting
    - The existing information-gain explorer (ig_explorer/)

Research Question (RQ-NeRF): Does NeRF-derived uncertainty provide
better exploration guidance than entropy-based occupancy uncertainty?

References:
    Mildenhall et al. (2020) "NeRF: Representing Scenes as Neural Radiance Fields"
    Ran et al. (2023) "NeSF: Neural Semantic Fields for Generalizable Zero-Shot Semantic Segmentation"
    Zhan et al. (2022) "ActiveRMAP: Radiance Field for Active Mapping"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NeRFConfig:
    pos_encoding_L: int = 6       # positional encoding frequencies
    dir_encoding_L: int = 4
    hidden_dim: int = 64
    n_samples_coarse: int = 32
    n_samples_fine: int = 64
    near: float = 0.1
    far: float = 5.0
    grid_resolution: float = 0.1  # metres per cell for uncertainty extraction
    grid_size: tuple[int, int] = (64, 64)
    n_uncertainty_samples: int = 8  # MC dropout samples for uncertainty


class PositionalEncoder:
    """Fourier positional encoding for NeRF inputs."""

    def __init__(self, L: int):
        self.L = L
        self.out_dim = 3 + 3 * 2 * L   # original + sin/cos per frequency

    def encode(self, x: np.ndarray) -> np.ndarray:
        """x: (..., 3) → (..., out_dim)"""
        parts = [x]
        for k in range(self.L):
            freq = 2.0 ** k * np.pi
            parts.append(np.sin(freq * x))
            parts.append(np.cos(freq * x))
        return np.concatenate(parts, axis=-1)


class TinyNeRF:
    """
    Lightweight NeRF with MC-Dropout for uncertainty estimation.
    Production: replace with PyTorch NeRF (instant-ngp, nerfstudio).
    """

    def __init__(self, cfg: NeRFConfig):
        self.cfg = cfg
        self.pos_enc = PositionalEncoder(cfg.pos_encoding_L)
        self.dir_enc = PositionalEncoder(cfg.dir_encoding_L)

        rng = np.random.default_rng(0)
        in_d = self.pos_enc.out_dim
        dir_d = self.dir_enc.out_dim
        H = cfg.hidden_dim

        # Density network: pos_enc → sigma
        self.W_density1 = rng.standard_normal((in_d, H)) * 0.01
        self.W_density2 = rng.standard_normal((H, H)) * 0.01
        self.W_sigma = rng.standard_normal((H, 1)) * 0.01

        # Colour network: [H + dir_enc] → RGB
        self.W_color1 = rng.standard_normal((H + dir_d, H // 2)) * 0.01
        self.W_color2 = rng.standard_normal((H // 2, 3)) * 0.01

        self._dropout_rate = 0.1  # MC dropout rate

    def query(self, points: np.ndarray, dirs: np.ndarray,
              training: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Query NeRF at 3D points with view directions.
        Returns (rgb, sigma) with shapes (N, 3) and (N,).
        """
        rng = np.random.default_rng()
        pe = self.pos_enc.encode(points)
        de = self.dir_enc.encode(dirs)

        # Density branch
        h = np.maximum(0, pe @ self.W_density1)
        if training or True:  # MC dropout always on for uncertainty
            mask = rng.random(h.shape) > self._dropout_rate
            h = h * mask / (1 - self._dropout_rate)
        h = np.maximum(0, h @ self.W_density2)
        sigma = np.maximum(0, (h @ self.W_sigma).squeeze(-1))

        # Colour branch
        h_dir = np.concatenate([h, de], axis=-1)
        h2 = np.maximum(0, h_dir @ self.W_color1)
        rgb = 1.0 / (1.0 + np.exp(-(h2 @ self.W_color2)))  # sigmoid

        return rgb, sigma

    def render_ray(self, ray_origin: np.ndarray,
                   ray_dir: np.ndarray) -> tuple[float, float, float]:
        """
        Volume render one ray. Returns (depth, colour_mean, uncertainty).
        """
        t_vals = np.linspace(self.cfg.near, self.cfg.far,
                             self.cfg.n_samples_coarse)
        points = ray_origin + t_vals[:, None] * ray_dir[None, :]
        dirs = np.broadcast_to(ray_dir, points.shape).copy()

        # MC dropout uncertainty: multiple forward passes
        sigmas_mc = []
        for _ in range(self.cfg.n_uncertainty_samples):
            _, sigma = self.query(points, dirs, training=True)
            sigmas_mc.append(sigma)

        sigma_mean = np.mean(sigmas_mc, axis=0)
        sigma_std = np.std(sigmas_mc, axis=0)

        # Volume rendering weights
        delta = np.diff(t_vals, append=t_vals[-1] + 0.1)
        alpha = 1.0 - np.exp(-sigma_mean * delta)
        T = np.cumprod(np.concatenate([[1.0], 1.0 - alpha[:-1]]))
        weights = T * alpha

        depth = float(np.sum(weights * t_vals))
        uncertainty = float(np.sum(weights * sigma_std))

        rgb_vals, _ = self.query(points, dirs, training=False)
        colour = float(np.sum(weights[:, None] * rgb_vals, axis=0).mean())

        return depth, colour, uncertainty


class NeRFUncertaintyMapper:
    """
    Builds a 2D uncertainty map by rendering rays downward through the scene
    and aggregating per-point uncertainty estimates.
    """

    def __init__(self, nerf: TinyNeRF | None = None,
                 cfg: NeRFConfig | None = None):
        self.cfg = cfg or NeRFConfig()
        self.nerf = nerf or TinyNeRF(self.cfg)

    def build_uncertainty_map(self, camera_poses: list[np.ndarray],
                               height: float = 1.5) -> np.ndarray:
        """
        Cast rays from each camera pose and project uncertainty onto 2D grid.

        Parameters
        ----------
        camera_poses : list of (4,4) camera-to-world transforms
        height       : camera height above ground

        Returns (H, W) uncertainty map in [0, 1].
        """
        H, W = self.cfg.grid_size
        unc_grid = np.zeros((H, W))
        count_grid = np.zeros((H, W)) + 1e-8

        for pose in camera_poses:
            # Sample rays in a grid pattern from camera
            for u in np.linspace(-0.5, 0.5, 8):
                for v in np.linspace(-0.5, 0.5, 8):
                    ray_dir_cam = np.array([u, v, 1.0])
                    ray_dir_cam /= np.linalg.norm(ray_dir_cam)

                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    ray_origin = t
                    ray_dir = R @ ray_dir_cam

                    depth, _, unc = self.nerf.render_ray(ray_origin, ray_dir)

                    # Project hit point to 2D grid
                    hit = ray_origin + depth * ray_dir
                    cx = int(hit[0] / self.cfg.grid_resolution) + W // 2
                    cy = int(hit[1] / self.cfg.grid_resolution) + H // 2
                    if 0 <= cx < W and 0 <= cy < H:
                        unc_grid[cy, cx] += unc
                        count_grid[cy, cx] += 1.0

        unc_map = unc_grid / count_grid
        # Normalise
        if unc_map.max() > 0:
            unc_map = unc_map / unc_map.max()
        logger.info("NeRF uncertainty map built from %d poses | mean_unc=%.3f",
                    len(camera_poses), float(unc_map.mean()))
        return unc_map

    def uncertainty_to_exploration_weights(
        self, unc_map: np.ndarray,
        occupancy: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert uncertainty map to exploration priority weights.
        High uncertainty + unoccupied → high priority for exploration.
        """
        weights = unc_map.copy()
        if occupancy is not None:
            # Don't explore occupied cells
            weights = weights * (1.0 - np.clip(occupancy, 0, 1))
        return weights / (weights.sum() + 1e-8)
