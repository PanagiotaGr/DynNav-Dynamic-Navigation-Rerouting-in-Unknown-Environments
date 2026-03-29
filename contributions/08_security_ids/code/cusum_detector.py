"""
cusum_detector.py

CUSUM (Cumulative Sum) anomaly detector for sensor innovation sequences.

Theory
------
Let z_t be the innovation (residual) from the state estimator at time t.
Under normal conditions z_t ~ N(0, σ²).
Under attack/fault: z_t ~ N(δ, σ²) where δ > 0 is the drift.

CUSUM statistic:
    S_0 = 0
    S_t = max(0, S_{t-1} + z_t - k)

Alarm triggered when S_t > h.

Parameters
    k : slack (typically δ/2, where δ is the expected drift magnitude)
    h : detection threshold

When anomaly detected → increase risk weight or trigger safe mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CUSUMConfig:
    k: float = 0.5      # slack = expected_drift / 2
    h: float = 5.0      # detection threshold
    sigma: float = 1.0  # expected innovation std under normal conditions


@dataclass
class CUSUMState:
    S_pos: float = 0.0   # upper CUSUM (detects positive drift / upward shift)
    S_neg: float = 0.0   # lower CUSUM (detects negative drift / downward shift)
    alarm: bool = False
    n_alarms: int = 0
    step: int = 0


class CUSUMDetector:
    """
    Two-sided CUSUM detector for scalar innovation sequences.

    Two-sided variant detects both upward (sensor over-reading)
    and downward (sensor failure / spoofing) anomalies.

    Parameters
    ----------
    config : CUSUMConfig
    """

    def __init__(self, config: Optional[CUSUMConfig] = None):
        self.cfg = config or CUSUMConfig()
        self.state = CUSUMState()
        self._history: list[float] = []
        self._s_pos_history: list[float] = []
        self._s_neg_history: list[float] = []

    def update(self, innovation: float) -> bool:
        """
        Process one innovation sample.

        Parameters
        ----------
        innovation : float
            Normalised innovation (z_t / sigma) or raw residual.

        Returns
        -------
        True if alarm is currently active.
        """
        z = float(innovation)
        k = self.cfg.k

        self.state.S_pos = max(0.0, self.state.S_pos + z - k)
        self.state.S_neg = max(0.0, self.state.S_neg - z - k)

        alarm = self.state.S_pos > self.cfg.h or self.state.S_neg > self.cfg.h
        if alarm and not self.state.alarm:
            self.state.n_alarms += 1
        self.state.alarm = alarm
        self.state.step += 1

        self._history.append(z)
        self._s_pos_history.append(self.state.S_pos)
        self._s_neg_history.append(self.state.S_neg)

        return alarm

    def reset(self) -> None:
        """Reset CUSUM statistics (e.g., after alarm acknowledged)."""
        self.state.S_pos = 0.0
        self.state.S_neg = 0.0
        self.state.alarm = False

    @property
    def risk_multiplier(self) -> float:
        """
        Risk multiplier for planner integration.
        Returns > 1.0 during anomaly, 1.0 otherwise.

        Uses a smooth function: 1 + tanh((S - h) / h)
        so risk grows gradually as CUSUM exceeds threshold.
        """
        S_max = max(self.state.S_pos, self.state.S_neg)
        if S_max <= self.cfg.h:
            return 1.0
        excess = (S_max - self.cfg.h) / self.cfg.h
        return 1.0 + float(np.tanh(excess))

    @property
    def should_trigger_safe_mode(self) -> bool:
        """True when CUSUM exceeds 2× threshold (severe anomaly)."""
        return (
            self.state.S_pos > 2.0 * self.cfg.h or
            self.state.S_neg > 2.0 * self.cfg.h
        )

    def summary(self) -> dict:
        return {
            "step":        self.state.step,
            "S_pos":       round(self.state.S_pos, 4),
            "S_neg":       round(self.state.S_neg, 4),
            "alarm":       self.state.alarm,
            "n_alarms":    self.state.n_alarms,
            "risk_mult":   round(self.risk_multiplier, 4),
            "safe_mode":   self.should_trigger_safe_mode,
        }


# ---------------------------------------------------------------------------
# Simulation helper: generate innovation sequences under attack
# ---------------------------------------------------------------------------

def simulate_innovations(
    n_steps: int = 200,
    attack_start: int = 100,
    attack_magnitude: float = 1.5,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a synthetic innovation sequence with a step-attack at attack_start.

    Returns
    -------
    innovations : (n_steps,) array
        z_t ~ N(0, sigma)       for t < attack_start
        z_t ~ N(delta, sigma)   for t >= attack_start
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, sigma, size=n_steps)
    z[attack_start:] += attack_magnitude
    return z


def evaluate_cusum(
    innovations: np.ndarray,
    config: Optional[CUSUMConfig] = None,
    attack_start: Optional[int] = None,
) -> dict:
    """
    Run CUSUM on the full innovation sequence and compute detection metrics.

    Returns
    -------
    dict with: detection_step, detection_delay, false_alarms, missed_detection
    """
    detector = CUSUMDetector(config)
    detection_step = None
    false_alarms = 0

    for t, z in enumerate(innovations):
        alarm = detector.update(z)
        if alarm and detection_step is None:
            detection_step = t
            if attack_start is not None and t < attack_start:
                false_alarms += 1

    delay = None
    missed = False
    if attack_start is not None:
        if detection_step is not None and detection_step >= attack_start:
            delay = detection_step - attack_start
        elif detection_step is None:
            missed = True

    return {
        "n_steps":          len(innovations),
        "attack_start":     attack_start,
        "detection_step":   detection_step,
        "detection_delay":  delay,
        "false_alarms":     false_alarms,
        "missed_detection": missed,
        "total_alarms":     detector.state.n_alarms,
        "final_S_pos":      round(detector.state.S_pos, 4),
        "final_S_neg":      round(detector.state.S_neg, 4),
    }
