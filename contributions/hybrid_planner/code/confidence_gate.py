"""
confidence_gate.py

Confidence gate for the hybrid heuristic:
  - Use learned heuristic (h_mean) when h_std is LOW (confident)
  - Fall back to admissible Euclidean heuristic when h_std is HIGH (uncertain)

This guarantees bounded suboptimality:
  - Euclidean heuristic is admissible → A* with it is optimal
  - When we trust the learned heuristic, we gain search speed

Suboptimality bound
-------------------
If we use the learned heuristic h_L and it is ε-admissible (h_L ≤ (1+ε)·h*),
then the resulting path cost ≤ (1+ε) · optimal.

The gate ensures ε-admissibility by checking:
    h_mean - k * h_std ≥ 0    (lower confidence bound is non-negative)
    h_mean ≤ admissible_h + epsilon_budget
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HeuristicChoice(Enum):
    LEARNED = "learned"
    ADMISSIBLE = "admissible"


@dataclass
class GateConfig:
    """
    Parameters for the confidence gate.

    std_threshold   : if h_std > this, fall back to admissible heuristic
    epsilon_budget  : max allowed excess over admissible (ε-admissibility)
    k               : lower-bound = h_mean - k * h_std must be ≥ 0
    """
    std_threshold: float = 2.0
    epsilon_budget: float = 0.5
    k: float = 1.0


class ConfidenceGate:
    """
    Decides whether to use the learned or admissible heuristic for each node.

    Parameters
    ----------
    config : GateConfig
    """

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()
        self._learned_uses: int = 0
        self._admissible_uses: int = 0

    def select(
        self,
        h_mean: float,
        h_std: float,
        h_admissible: float,
    ) -> tuple[float, HeuristicChoice]:
        """
        Choose the best heuristic value for this node.

        Parameters
        ----------
        h_mean      : learned heuristic mean
        h_std       : learned heuristic std (uncertainty)
        h_admissible: admissible heuristic (e.g. Euclidean distance)

        Returns
        -------
        (h_value, choice)
        """
        cfg = self.config

        # Condition 1: uncertainty too high → fall back
        if h_std > cfg.std_threshold:
            self._admissible_uses += 1
            return h_admissible, HeuristicChoice.ADMISSIBLE

        # Condition 2: lower confidence bound must be non-negative
        lower_bound = h_mean - cfg.k * h_std
        if lower_bound < 0:
            self._admissible_uses += 1
            return h_admissible, HeuristicChoice.ADMISSIBLE

        # Condition 3: learned must not exceed ε-budget above admissible
        if h_mean > h_admissible + cfg.epsilon_budget:
            self._admissible_uses += 1
            return h_admissible, HeuristicChoice.ADMISSIBLE

        # All checks passed → use learned
        self._learned_uses += 1
        return h_mean, HeuristicChoice.LEARNED

    @property
    def learned_fraction(self) -> float:
        total = self._learned_uses + self._admissible_uses
        return self._learned_uses / total if total > 0 else 0.0

    def reset_stats(self) -> None:
        self._learned_uses = 0
        self._admissible_uses = 0
