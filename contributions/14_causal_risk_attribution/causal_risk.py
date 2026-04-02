"""
Contribution 14: Causal Risk Attribution
=========================================
Uses Structural Causal Models (SCM) to answer *why* a navigation failure
occurred and to perform counterfactual queries: "what would have happened
if sensor X had not failed / if the obstacle had not been there?"

This goes beyond correlation-based anomaly detection (Contribution 08) by
reasoning about *causal mechanisms*, enabling targeted interventions.

Research Question (RQ-Causal): Can counterfactual reasoning reduce repeated
navigation failures by identifying root causes rather than symptoms?

References:
    Pearl (2009) "Causality: Models, Reasoning, and Inference"
    Scholkopf et al. (2021) "Toward Causal Representation Learning"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SCM primitives
# ---------------------------------------------------------------------------

class NodeType(Enum):
    EXOGENOUS = "exogenous"   # external noise variables
    ENDOGENOUS = "endogenous" # causally determined variables


@dataclass
class CausalNode:
    name: str
    node_type: NodeType = NodeType.ENDOGENOUS
    parents: list[str] = field(default_factory=list)
    # structural equation: f(parent_values, noise) → value
    # stored as a callable; defaults to weighted sum + noise
    structural_eq: Optional[callable] = None
    _value: Optional[float] = field(default=None, init=False, repr=False)

    def evaluate(self, parent_values: dict[str, float],
                 noise: float = 0.0) -> float:
        if self.structural_eq is not None:
            v = self.structural_eq(parent_values, noise)
        else:
            # Default: linear combination of parents + noise
            v = sum(parent_values.values()) + noise
        self._value = v
        return v


# ---------------------------------------------------------------------------
# Navigation SCM
# ---------------------------------------------------------------------------

class NavigationSCM:
    """
    Structural Causal Model for a mobile-robot navigation episode.

    Variables (default):
        sensor_noise   (exogenous)
        localization_error  ← sensor_noise
        map_accuracy        ← sensor_noise
        obstacle_detection  ← localization_error, sensor_noise
        path_risk           ← map_accuracy, obstacle_detection
        collision           ← path_risk, localization_error
    """

    def __init__(self):
        self.nodes: dict[str, CausalNode] = {}
        self._build_default_graph()

    def _build_default_graph(self):
        self.add_node(CausalNode(
            "sensor_noise", NodeType.EXOGENOUS,
            structural_eq=lambda p, n: n
        ))
        self.add_node(CausalNode(
            "localization_error", NodeType.ENDOGENOUS,
            parents=["sensor_noise"],
            structural_eq=lambda p, n: 0.6 * p["sensor_noise"] + 0.4 * n
        ))
        self.add_node(CausalNode(
            "map_accuracy", NodeType.ENDOGENOUS,
            parents=["sensor_noise"],
            structural_eq=lambda p, n: 1.0 - 0.5 * abs(p["sensor_noise"]) + 0.1 * n
        ))
        self.add_node(CausalNode(
            "obstacle_detection", NodeType.ENDOGENOUS,
            parents=["localization_error", "sensor_noise"],
            structural_eq=lambda p, n: (
                1.0 - 0.4 * abs(p["localization_error"])
                - 0.3 * abs(p["sensor_noise"]) + 0.05 * n
            )
        ))
        self.add_node(CausalNode(
            "path_risk", NodeType.ENDOGENOUS,
            parents=["map_accuracy", "obstacle_detection"],
            structural_eq=lambda p, n: (
                0.5 * (1 - p["map_accuracy"])
                + 0.5 * (1 - p["obstacle_detection"])
                + 0.05 * n
            )
        ))
        self.add_node(CausalNode(
            "collision", NodeType.ENDOGENOUS,
            parents=["path_risk", "localization_error"],
            structural_eq=lambda p, n: (
                0.7 * p["path_risk"]
                + 0.3 * abs(p["localization_error"])
                + 0.05 * n
            )
        ))

    def add_node(self, node: CausalNode) -> None:
        self.nodes[node.name] = node

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def observational_query(self, noise_samples: dict[str, float]
                             ) -> dict[str, float]:
        """
        Evaluate the SCM under given exogenous noise values.
        Returns the value of every endogenous node.
        """
        values: dict[str, float] = {}
        # Topological evaluation order (hardcoded for default graph)
        order = [
            "sensor_noise", "localization_error", "map_accuracy",
            "obstacle_detection", "path_risk", "collision"
        ]
        for name in order:
            node = self.nodes[name]
            parent_vals = {p: values[p] for p in node.parents}
            noise = noise_samples.get(name, 0.0)
            values[name] = node.evaluate(parent_vals, noise)
        return values

    def counterfactual_query(
        self,
        observed_noise: dict[str, float],
        intervention: dict[str, float],
    ) -> dict[str, float]:
        """
        Counterfactual: given the *same* noise, what would have happened
        if we had set node(s) to specific values (do-calculus intervention)?

        Parameters
        ----------
        observed_noise  : exogenous noise realisation (from the actual episode)
        intervention    : {node_name: forced_value}  — do(X=x)
        """
        values: dict[str, float] = {}
        order = [
            "sensor_noise", "localization_error", "map_accuracy",
            "obstacle_detection", "path_risk", "collision"
        ]
        for name in order:
            if name in intervention:
                values[name] = intervention[name]
                continue
            node = self.nodes[name]
            parent_vals = {p: values[p] for p in node.parents}
            noise = observed_noise.get(name, 0.0)
            values[name] = node.evaluate(parent_vals, noise)
        return values

    def average_causal_effect(
        self,
        treatment_node: str,
        outcome_node: str,
        treatment_value: float,
        control_value: float,
        n_samples: int = 500,
    ) -> float:
        """
        E[Y | do(X=treatment)] - E[Y | do(X=control)]
        Estimated by Monte-Carlo over noise distributions.
        """
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(n_samples):
            noise = {name: rng.standard_normal() for name in self.nodes}
            y_treat = self.counterfactual_query(
                noise, {treatment_node: treatment_value}
            )[outcome_node]
            y_ctrl = self.counterfactual_query(
                noise, {treatment_node: control_value}
            )[outcome_node]
            diffs.append(y_treat - y_ctrl)

        ace = float(np.mean(diffs))
        logger.info(
            "ACE(do(%s=%s) vs do(%s=%s)) on %s = %.4f",
            treatment_node, treatment_value,
            treatment_node, control_value,
            outcome_node, ace,
        )
        return ace

    def root_cause_ranking(
        self,
        observed_noise: dict[str, float],
        outcome_node: str = "collision",
        n_samples: int = 200,
    ) -> list[tuple[str, float]]:
        """
        Rank all upstream nodes by their counterfactual contribution to
        the observed outcome (Shapley-inspired ablation).
        """
        baseline = self.observational_query(observed_noise)[outcome_node]
        scores = {}
        rng = np.random.default_rng(0)

        for name, node in self.nodes.items():
            if name == outcome_node:
                continue
            diffs = []
            for _ in range(n_samples):
                noise = {k: rng.standard_normal() for k in self.nodes}
                noise.update(observed_noise)
                # Intervene: set node to zero (null intervention)
                cf = self.counterfactual_query(noise, {name: 0.0})
                diffs.append(baseline - cf[outcome_node])
            scores[name] = float(np.mean(diffs))

        ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        logger.info("Root cause ranking for %s: %s", outcome_node, ranked)
        return ranked
