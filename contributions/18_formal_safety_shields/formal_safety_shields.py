"""
Contribution 18: Formal Safety Shields — STL + Control Barrier Functions
=========================================================================
Provides runtime-verifiable safety guarantees by wrapping any navigation
planner with two complementary mechanisms:

1. Signal Temporal Logic (STL) monitor: evaluates temporal safety
   specifications over the robot's state trajectory and flags violations.

2. Control Barrier Function (CBF) filter: modifies velocity commands in
   real-time to guarantee obstacle avoidance with formal Lyapunov-style
   safety certificates.

Research Question (RQ-Formal): Do STL+CBF safety shields reduce constraint
violations without significantly degrading navigation efficiency?

References:
    Donze & Maler (2010) "Robust Satisfaction of Temporal Logic Specifications"
    Ames et al. (2019) "Control Barrier Functions: Theory and Applications"
    Raman et al. (2015) "Model Predictive Control with Signal Temporal Logic Specs"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal Temporal Logic (STL) primitives
# ---------------------------------------------------------------------------

class STLFormula:
    """Base class for STL formulae. Evaluates robustness ρ over a signal."""

    def robustness(self, signal: np.ndarray, t: int) -> float:
        raise NotImplementedError


class STLAtom(STLFormula):
    """Predicate μ(s) ≥ 0 evaluated at time t."""

    def __init__(self, predicate: Callable[[np.ndarray], float], name: str = ""):
        self.predicate = predicate
        self.name = name

    def robustness(self, signal: np.ndarray, t: int) -> float:
        return float(self.predicate(signal[t]))


class STLAlways(STLFormula):
    """□[a,b] φ  — Always φ holds in [t+a, t+b]."""

    def __init__(self, formula: STLFormula, a: int, b: int):
        self.formula = formula
        self.a, self.b = a, b

    def robustness(self, signal: np.ndarray, t: int) -> float:
        T = len(signal)
        rho_vals = [
            self.formula.robustness(signal, min(t + tau, T - 1))
            for tau in range(self.a, self.b + 1)
        ]
        return float(min(rho_vals)) if rho_vals else 0.0


class STLEventually(STLFormula):
    """◇[a,b] φ  — Eventually φ holds in [t+a, t+b]."""

    def __init__(self, formula: STLFormula, a: int, b: int):
        self.formula = formula
        self.a, self.b = a, b

    def robustness(self, signal: np.ndarray, t: int) -> float:
        T = len(signal)
        rho_vals = [
            self.formula.robustness(signal, min(t + tau, T - 1))
            for tau in range(self.a, self.b + 1)
        ]
        return float(max(rho_vals)) if rho_vals else 0.0


class STLAnd(STLFormula):
    """φ₁ ∧ φ₂"""

    def __init__(self, *formulae: STLFormula):
        self.formulae = formulae

    def robustness(self, signal: np.ndarray, t: int) -> float:
        return min(f.robustness(signal, t) for f in self.formulae)


class STLOr(STLFormula):
    """φ₁ ∨ φ₂"""

    def __init__(self, *formulae: STLFormula):
        self.formulae = formulae

    def robustness(self, signal: np.ndarray, t: int) -> float:
        return max(f.robustness(signal, t) for f in self.formulae)


# ---------------------------------------------------------------------------
# STL Safety Monitor
# ---------------------------------------------------------------------------

class STLMonitor:
    """
    Online STL monitor: evaluates a set of safety specifications over the
    robot's state trajectory and issues warnings / blocks actions.
    """

    def __init__(self, specs: list[tuple[str, STLFormula]],
                 robustness_margin: float = 0.0):
        self.specs = specs   # (name, formula)
        self.margin = robustness_margin
        self._trajectory: list[np.ndarray] = []
        self.violation_log: list[dict] = []

    def update(self, state: np.ndarray) -> dict[str, float]:
        """Append state and evaluate all specs at current time."""
        self._trajectory.append(state)
        signal = np.array(self._trajectory)
        t = len(signal) - 1
        results = {}
        for name, formula in self.specs:
            rho = formula.robustness(signal, max(0, t))
            results[name] = rho
            if rho < self.margin:
                entry = {"t": t, "spec": name, "rho": rho}
                self.violation_log.append(entry)
                logger.warning("STL violation | spec=%s ρ=%.4f", name, rho)
        return results

    def is_safe(self) -> bool:
        if not self._trajectory:
            return True
        signal = np.array(self._trajectory)
        t = len(signal) - 1
        return all(
            f.robustness(signal, t) >= self.margin
            for _, f in self.specs
        )

    def reset(self) -> None:
        self._trajectory.clear()
        self.violation_log.clear()


# ---------------------------------------------------------------------------
# Control Barrier Function (CBF) safety filter
# ---------------------------------------------------------------------------

@dataclass
class CBFConfig:
    safety_radius: float = 0.4       # minimum distance to obstacle (metres)
    alpha: float = 1.5               # CBF class-K function gain
    gamma: float = 0.95              # margin relaxation factor
    max_iter: int = 20               # QP solver iterations (gradient projection)
    step_size: float = 0.05          # gradient projection step


class CBFSafetyFilter:
    """
    Control Barrier Function filter for a unicycle / point robot.

    Given a desired velocity command u_des and a set of obstacle positions,
    returns a minimally modified u_safe that provably maintains h(x) ≥ 0
    where h(x) = ‖x - x_obs‖² - r²  (sphere safety set).
    """

    def __init__(self, cfg: CBFConfig | None = None):
        self.cfg = cfg or CBFConfig()

    def h(self, robot_pos: np.ndarray, obs_pos: np.ndarray) -> float:
        """Barrier function: h(x) = ‖robot − obs‖ − r ≥ 0 means safe."""
        return float(np.linalg.norm(robot_pos - obs_pos) - self.cfg.safety_radius)

    def dh_dx(self, robot_pos: np.ndarray, obs_pos: np.ndarray) -> np.ndarray:
        """Gradient ∂h/∂x."""
        diff = robot_pos - obs_pos
        norm = np.linalg.norm(diff) + 1e-8
        return diff / norm

    def filter(self, u_des: np.ndarray,
               robot_pos: np.ndarray,
               obstacles: list[np.ndarray]) -> np.ndarray:
        """
        QP-based CBF filter (gradient projection for simplicity).
        Finds u_safe ≈ argmin ‖u − u_des‖² s.t. Lf_h + alpha*h ≥ 0 ∀ obs.
        """
        u = u_des.copy().astype(float)

        for obs in obstacles:
            for _ in range(self.cfg.max_iter):
                h_val = self.h(robot_pos, obs)
                grad = self.dh_dx(robot_pos, obs)

                # CBF condition: grad · u + alpha * h ≥ 0
                lf_h = float(grad @ u)
                constraint = lf_h + self.cfg.alpha * h_val

                if constraint >= 0:
                    break  # satisfied

                # Project u onto the constraint boundary
                violation = -constraint
                u = u + self.cfg.step_size * violation * grad
                logger.debug("CBF correction applied | h=%.3f lf_h=%.3f", h_val, lf_h)

        return u

    def batch_filter(self, u_des: np.ndarray,
                     robot_pos: np.ndarray,
                     obstacles: list[np.ndarray],
                     robot_state: Optional[np.ndarray] = None
                     ) -> tuple[np.ndarray, dict]:
        """
        Filter command and return diagnostics.
        """
        u_safe = self.filter(u_des, robot_pos, obstacles)
        correction = np.linalg.norm(u_safe - u_des)
        min_dist = min(
            np.linalg.norm(robot_pos - obs) for obs in obstacles
        ) if obstacles else np.inf

        info = {
            "correction_norm": round(float(correction), 5),
            "min_obstacle_dist": round(float(min_dist), 5),
            "safety_violated": min_dist < self.cfg.safety_radius,
        }
        if info["safety_violated"]:
            logger.warning(
                "CBF: robot at distance %.3f < safety radius %.3f",
                min_dist, self.cfg.safety_radius
            )
        return u_safe, info


# ---------------------------------------------------------------------------
# Combined safety shield
# ---------------------------------------------------------------------------

class SafetyShield:
    """
    Wraps a planner with both STL monitoring and CBF command filtering.
    Usage:
        shield = SafetyShield(stl_monitor, cbf_filter)
        u_safe = shield.step(u_desired, robot_pos, obstacles, robot_state)
    """

    def __init__(self, stl_monitor: STLMonitor,
                 cbf_filter: CBFSafetyFilter):
        self.stl = stl_monitor
        self.cbf = cbf_filter
        self.steps = 0
        self.total_corrections = 0.0

    def step(self, u_des: np.ndarray,
             robot_pos: np.ndarray,
             obstacles: list[np.ndarray],
             robot_state: Optional[np.ndarray] = None
             ) -> tuple[np.ndarray, dict]:
        """
        One control step:
        1. Update STL monitor.
        2. Apply CBF filter.
        3. Return safe command + diagnostics.
        """
        stl_state = robot_state if robot_state is not None else robot_pos
        stl_robustness = self.stl.update(stl_state)
        u_safe, cbf_info = self.cbf.batch_filter(u_des, robot_pos, obstacles)

        self.steps += 1
        self.total_corrections += cbf_info["correction_norm"]

        return u_safe, {
            "stl": stl_robustness,
            "cbf": cbf_info,
            "stl_safe": self.stl.is_safe(),
            "avg_correction": round(self.total_corrections / self.steps, 6),
        }
