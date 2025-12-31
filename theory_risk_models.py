"""
theory_risk_models.py

Formalization helpers for risk-aware navigation and meta-decision making.

This module does NOT implement a full planner. Instead, it provides
well-documented building blocks that make the theoretical model explicit:

- state space and paths
- path cost functional J = length + λ · risk
- risk aggregation from an uncertainty grid σ(x)
- meta-decision layer: NORMAL vs SAFE mode based on self-trust

These utilities are intended to be referenced in the "Theory" section
of a thesis or paper, and to connect the mathematical notation to
the actual implementation in the repository.
"""

from dataclasses import dataclass
from typing import Sequence, Callable, Literal, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. State space and paths
# ---------------------------------------------------------------------------

@dataclass
class Path:
    """
    A discrete path γ = (x_0, x_1, ..., x_{N-1}) in R^d.

    Attributes
    ----------
    points : np.ndarray
        Array of shape (N, d) with the sequence of states x_i.
    """
    points: np.ndarray

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    def length(self) -> float:
        """
        Computes the geometric length of the path:

            L(γ) = Σ || x_{i+1} - x_i ||_2

        Returns
        -------
        float
            Total path length.
        """
        if self.num_points < 2:
            return 0.0
        diffs = np.diff(self.points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        return float(seg_lengths.sum())


# ---------------------------------------------------------------------------
# 2. Risk from uncertainty grid σ(x)
# ---------------------------------------------------------------------------

def sample_sigma_from_grid(
    points: np.ndarray,
    sigma_grid: np.ndarray,
    grid_origin: Tuple[float, float],
    grid_resolution: float,
) -> np.ndarray:
    """
    Samples the uncertainty σ(x) from a 2D grid at the given points.

    The grid is assumed to represent σ(x) on a regular lattice:

        σ_grid[i, j]  ≈  σ(x, y)  for
        x = x0 + i * resolution,  y = y0 + j * resolution

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) with positions (x, y).
    sigma_grid : np.ndarray
        2D array with σ values.
    grid_origin : (float, float)
        (x0, y0) world coordinates corresponding to sigma_grid[0, 0].
    grid_resolution : float
        Linear cell size (meters per cell).

    Returns
    -------
    np.ndarray
        Array of shape (N,) with σ(x_i) values (nearest neighbor lookup).
    """
    x0, y0 = grid_origin
    h, w = sigma_grid.shape

    xs = points[:, 0]
    ys = points[:, 1]

    ix = np.floor((xs - x0) / grid_resolution).astype(int)
    iy = np.floor((ys - y0) / grid_resolution).astype(int)

    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)

    sigmas = sigma_grid[iy, ix]
    return sigmas.astype(float)


def path_risk_integral(
    path: Path,
    sigma_values: np.ndarray,
    mode: Literal["sum", "mean", "energy"] = "sum",
) -> float:
    """
    Aggregates risk along a path γ given σ(x_i).

    Conceptually, we approximate an integral along the path:

        R(γ) ≈ Σ σ(x_i) · Δs_i

    where Δs_i is the segment length. For simplicity we provide three modes:

    - "sum": R(γ) = Σ σ(x_i)
    - "mean": R(γ) = (1/N) Σ σ(x_i)
    - "energy": R(γ) = Σ σ(x_i)^2

    Parameters
    ----------
    path : Path
        Discrete path with points x_i.
    sigma_values : np.ndarray
        Array of shape (N,) with σ(x_i).
    mode : {"sum", "mean", "energy"}
        Aggregation mode.

    Returns
    -------
    float
        Aggregated risk R(γ).
    """
    sigma_values = np.asarray(sigma_values, dtype=float)
    assert sigma_values.shape[0] == path.num_points, "sigma_values must match path length"

    if mode == "sum":
        return float(sigma_values.sum())
    elif mode == "mean":
        return float(sigma_values.mean())
    elif mode == "energy":
        return float((sigma_values ** 2).sum())
    else:
        raise ValueError(f"Unknown risk mode: {mode}")


# ---------------------------------------------------------------------------
# 3. Cost functional J = length + λ · risk
# ---------------------------------------------------------------------------

def path_cost(
    path: Path,
    sigma_values: np.ndarray,
    lam: float,
    risk_mode: Literal["sum", "mean", "energy"] = "sum",
) -> float:
    """
    Computes the risk-aware cost functional

        J(γ; λ) = L(γ) + λ · R(γ),

    where L(γ) is the geometric path length and R(γ) is a risk
    functional derived from σ(x) along the path.

    Parameters
    ----------
    path : Path
        The discrete path γ.
    sigma_values : np.ndarray
        Array σ(x_i) along the path.
    lam : float
        Risk weighting parameter λ ≥ 0.
    risk_mode : {"sum", "mean", "energy"}
        How to aggregate σ(x) into R(γ).

    Returns
    -------
    float
        The total cost J(γ; λ).
    """
    L = path.length()
    R = path_risk_integral(path, sigma_values, mode=risk_mode)
    return float(L + lam * R)


# ---------------------------------------------------------------------------
# 4. Meta-decision: NORMAL vs SAFE mode (self-trust layer)
# ---------------------------------------------------------------------------

def choose_meta_policy(
    self_trust_scores: Sequence[float],
    threshold: float,
) -> Literal["NORMAL", "SAFE"]:
    """
    Meta-decision rule on top of a planner, based on self-trust.

    Given a set of self-trust scores S_r ∈ [0, 1] for each robot r, we define:

        if   min_r S_r < τ    → SAFE mode
        else                  → NORMAL mode,

    where τ ∈ (0, 1] is a user-defined safety threshold.

    Parameters
    ----------
    self_trust_scores : Sequence[float]
        Self-trust values for each robot (or each sensing modality).
    threshold : float
        Safe-mode threshold τ.

    Returns
    -------
    {"NORMAL", "SAFE"}
        The active meta-policy.
    """
    min_s = float(min(self_trust_scores)) if self_trust_scores else 1.0
    if min_s < threshold:
        return "SAFE"
    else:
        return "NORMAL"

