"""
adaptive_risk_policy.py

Adaptive controller for the belief–risk planner.
It updates the lambda weight based on realized fused risk
(e.g. integrated uncertainty along the planned path).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskContext:
    """
    Optional context features that can modulate the risk target.

    Attributes
    ----------
    step_idx : int
        Index of the current planning step.
    distance_to_goal : float
        Estimated Euclidean distance to goal (in meters or cells).
    map_entropy : Optional[float]
        Global map entropy / uncertainty (if available).
    """
    step_idx: int = 0
    distance_to_goal: float = 0.0
    map_entropy: Optional[float] = None


class AdaptiveRiskPolicy:
    """
    Simple feedback controller for lambda in the belief–risk cost:

        J(lambda) = L_geom + lambda * R_fused

    The policy tries to keep the realized fused risk R_k
    close to a target risk level R_target, while respecting
    [lambda_min, lambda_max].

    This is intentionally model-agnostic, so it can be plugged into
    different planners / environments.
    """

    def __init__(
        self,
        lambda_init: float = 0.5,
        lambda_min: float = 0.0,
        lambda_max: float = 5.0,
        target_risk: float = 1.0,
        eta: float = 0.1,
        adaptive_target_distance: bool = True,
        min_distance_for_relaxation: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        lambda_init : float
            Initial lambda value.
        lambda_min : float
            Lower bound for lambda.
        lambda_max : float
            Upper bound for lambda.
        target_risk : float
            Nominal target fused risk (R_target).
        eta : float
            Learning-rate / step size for lambda update.
        adaptive_target_distance : bool
            If True, the target risk is relaxed when far from the goal.
        min_distance_for_relaxation : float
            Distance threshold under which the pure target_risk is used.
        """
        self.lambda_value = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.base_target_risk = target_risk
        self.eta = eta

        self.adaptive_target_distance = adaptive_target_distance
        self.min_distance_for_relaxation = min_distance_for_relaxation

    def _compute_effective_target(
        self,
        context: Optional[RiskContext] = None,
    ) -> float:
        """
        Compute an effective risk target given the context.

        Example heuristic:
        - Far from goal: allow slightly higher risk to make faster progress.
        - Near goal: enforce the base_target_risk.
        """
        if not self.adaptive_target_distance or context is None:
            return self.base_target_risk

        d = max(context.distance_to_goal, 0.0)

        if d <= self.min_distance_for_relaxation:
            # Very close to goal -> strictly enforce base target.
            return self.base_target_risk

        # Simple relaxation: target risk increases logarithmically with distance.
        # You can tune this according to your environment.
        relaxed_target = self.base_target_risk * (1.0 + 0.2 * (d / (d + 1.0)))
        return relaxed_target

    def update(
        self,
        realized_risk: float,
        context: Optional[RiskContext] = None,
    ) -> float:
        """
        Update lambda based on realized fused risk and context.

        Parameters
        ----------
        realized_risk : float
            Measured fused risk R_k along the last planned path.
        context : Optional[RiskContext]
            Additional context (distance to goal, step index, etc.)

        Returns
        -------
        float
            Updated lambda value (clamped within [lambda_min, lambda_max]).
        """
        target = self._compute_effective_target(context)
        error = realized_risk - target

        # Gradient-descent-like update:
        # if realized_risk > target  -> increase lambda -> more risk-averse
        # if realized_risk < target  -> decrease lambda -> more risk-seeking
        self.lambda_value += self.eta * error

        # Clamp to valid range
        self.lambda_value = max(self.lambda_min, min(self.lambda_max, self.lambda_value))
        return self.lambda_value

    def get_lambda(self) -> float:
        """Return the current lambda value."""
        return self.lambda_value

