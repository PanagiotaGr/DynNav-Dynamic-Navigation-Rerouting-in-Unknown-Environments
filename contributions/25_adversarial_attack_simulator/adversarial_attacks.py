"""
Contribution 25: Adversarial Attack Simulator
==============================================
Generates adversarial perturbations on sensor data (LiDAR, camera, odometry)
to evaluate the robustness of DynNav's navigation and intrusion detection
systems (Contribution 08).

Attack types:
    - FGSM (Fast Gradient Sign Method) on neural network inputs
    - PGD (Projected Gradient Descent) — stronger iterative attack
    - LiDAR spoofing: inject phantom obstacles or remove real ones
    - GPS/odometry spoofing: drift robot's perceived position
    - Sensor blinding: zero out sensor regions

Research Question (RQ-Adv): How robust is DynNav's planning pipeline
to adversarial sensor manipulation, and does the IDS (Contribution 08)
detect these attacks?

References:
    Goodfellow et al. (2015) "Explaining and Harnessing Adversarial Examples"
    Cao et al. (2019) "Adversarial Sensor Attack on LiDAR-based Perception"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attack types
# ---------------------------------------------------------------------------

class AttackType(Enum):
    FGSM = "fgsm"
    PGD = "pgd"
    LIDAR_SPOOF_ADD = "lidar_spoof_add"      # inject phantom obstacles
    LIDAR_SPOOF_REMOVE = "lidar_spoof_remove" # remove real obstacles
    ODOM_DRIFT = "odom_drift"                 # gradually shift odometry
    SENSOR_BLIND = "sensor_blind"             # zero out sensor region


@dataclass
class AttackConfig:
    attack_type: AttackType = AttackType.FGSM
    epsilon: float = 0.1          # perturbation budget (L∞ norm)
    pgd_steps: int = 20           # PGD iterations
    pgd_step_size: float = 0.01
    lidar_n_phantoms: int = 5     # phantom obstacles for spoof
    lidar_remove_frac: float = 0.2 # fraction of points to remove
    odom_drift_rate: float = 0.02  # metres drift per step
    blind_region_frac: float = 0.3 # fraction of sensor to blind
    targeted: bool = False         # targeted vs untargeted attack


# ---------------------------------------------------------------------------
# Gradient-based attacks (FGSM / PGD)
# ---------------------------------------------------------------------------

class GradientAttacker:
    """
    FGSM and PGD attacks on neural network observation inputs.
    Works with any differentiable loss function provided as a callable.
    Uses finite-difference approximation (no autograd needed).
    """

    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg

    def fgsm(self, obs: np.ndarray,
             loss_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """
        FGSM: x_adv = x + ε * sign(∇_x L(x))
        Gradient approximated by finite differences.
        """
        grad = self._finite_diff_grad(obs, loss_fn)
        delta = self.cfg.epsilon * np.sign(grad)
        return np.clip(obs + delta, 0.0, 1.0)

    def pgd(self, obs: np.ndarray,
            loss_fn: Callable[[np.ndarray], float],
            x_nat: Optional[np.ndarray] = None) -> np.ndarray:
        """
        PGD: iterative FGSM with projection onto L∞ ball.
        """
        if x_nat is None:
            x_nat = obs.copy()

        x_adv = obs.copy() + np.random.default_rng().uniform(
            -self.cfg.epsilon, self.cfg.epsilon, obs.shape
        )

        for step in range(self.cfg.pgd_steps):
            grad = self._finite_diff_grad(x_adv, loss_fn)
            x_adv = x_adv + self.cfg.pgd_step_size * np.sign(grad)
            # Project onto ε-ball around natural example
            x_adv = np.clip(x_adv, x_nat - self.cfg.epsilon,
                            x_nat + self.cfg.epsilon)
            x_adv = np.clip(x_adv, 0.0, 1.0)

        logger.debug("PGD attack: %d steps, final loss=%.4f",
                     self.cfg.pgd_steps, loss_fn(x_adv))
        return x_adv

    def _finite_diff_grad(self, x: np.ndarray,
                          loss_fn: Callable[[np.ndarray], float],
                          h: float = 1e-4) -> np.ndarray:
        """Numerical gradient via central differences."""
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy(); x_plus.flat[i] += h
            x_minus = x.copy(); x_minus.flat[i] -= h
            grad.flat[i] = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * h)
        return grad


# ---------------------------------------------------------------------------
# LiDAR attacks
# ---------------------------------------------------------------------------

class LiDARAttacker:
    """
    Physics-plausible attacks on LiDAR point clouds.
    """

    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(42)

    def spoof_add(self, point_cloud: np.ndarray,
                  robot_pos: np.ndarray,
                  max_range: float = 5.0) -> np.ndarray:
        """
        Inject phantom obstacle points near the robot.
        point_cloud: (N, 3) array of XYZ points
        """
        phantoms = []
        for _ in range(self.cfg.lidar_n_phantoms):
            angle = self._rng.uniform(0, 2 * np.pi)
            dist = self._rng.uniform(0.5, max_range * 0.6)
            pt = robot_pos + dist * np.array([
                np.cos(angle), np.sin(angle), 0.0
            ])
            # Cluster of points to make it look realistic
            cluster = pt + self._rng.standard_normal((8, 3)) * 0.05
            phantoms.append(cluster)

        phantoms_arr = np.vstack(phantoms) if phantoms else np.zeros((0, 3))
        result = np.vstack([point_cloud, phantoms_arr])
        logger.info("LiDAR spoof: injected %d phantom points", len(phantoms_arr))
        return result

    def spoof_remove(self, point_cloud: np.ndarray,
                     target_region: Optional[np.ndarray] = None
                     ) -> np.ndarray:
        """
        Remove points from a specific direction (simulate sensor jamming).
        """
        n = len(point_cloud)
        if target_region is not None:
            # Remove points near target region
            dists = np.linalg.norm(point_cloud - target_region, axis=1)
            mask = dists > 1.0   # keep points far from target
        else:
            # Random removal
            keep = self._rng.random(n) > self.cfg.lidar_remove_frac
            mask = keep

        removed = n - mask.sum()
        logger.info("LiDAR remove: dropped %d / %d points", removed, n)
        return point_cloud[mask]

    def sensor_blind(self, point_cloud: np.ndarray,
                     blind_angle_deg: float = 90.0) -> np.ndarray:
        """Blind a sector of the LiDAR scan."""
        angles = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])
        blind_rad = np.deg2rad(blind_angle_deg / 2)
        # Blind the forward sector
        mask = np.abs(angles) > blind_rad
        logger.info("LiDAR blind: kept %d / %d points", mask.sum(), len(mask))
        return point_cloud[mask]


# ---------------------------------------------------------------------------
# Odometry / GPS spoofer
# ---------------------------------------------------------------------------

class OdometrySpoofer:
    """
    Injects gradual drift into odometry estimates.
    Simulates GPS spoofing or wheel-encoder manipulation.
    """

    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg
        self._cumulative_drift = np.zeros(3)   # (x, y, yaw)
        self._rng = np.random.default_rng(0)
        self._active = False

    def activate(self):
        self._active = True
        logger.warning("Odometry spoofing ACTIVATED")

    def deactivate(self):
        self._active = False
        self._cumulative_drift[:] = 0

    def corrupt(self, odom: np.ndarray) -> np.ndarray:
        """
        odom: (3,) array [x, y, yaw]
        Returns corrupted odometry.
        """
        if not self._active:
            return odom.copy()

        drift_step = self._rng.standard_normal(3) * self.cfg.odom_drift_rate
        self._cumulative_drift += drift_step
        corrupted = odom + self._cumulative_drift
        logger.debug("Odom drift: cumulative=%.3f m",
                     np.linalg.norm(self._cumulative_drift[:2]))
        return corrupted

    @property
    def total_drift_m(self) -> float:
        return float(np.linalg.norm(self._cumulative_drift[:2]))


# ---------------------------------------------------------------------------
# Robustness evaluator
# ---------------------------------------------------------------------------

class RobustnessEvaluator:
    """
    Evaluates a navigation system's robustness against a suite of attacks.
    """

    def __init__(self, attack_cfg: AttackConfig | None = None):
        self.cfg = attack_cfg or AttackConfig()
        self.grad_attacker = GradientAttacker(self.cfg)
        self.lidar_attacker = LiDARAttacker(self.cfg)
        self.odom_spoofer = OdometrySpoofer(self.cfg)

    def evaluate(self,
                 obs_samples: list[np.ndarray],
                 loss_fn: Callable[[np.ndarray], float],
                 point_clouds: Optional[list[np.ndarray]] = None,
                 ) -> dict:
        """
        Run all attacks on provided samples.
        Returns robustness metrics dict.
        """
        results = {}

        # FGSM
        fgsm_losses = []
        for obs in obs_samples:
            adv = self.grad_attacker.fgsm(obs, loss_fn)
            fgsm_losses.append(loss_fn(adv) - loss_fn(obs))
        results["fgsm_mean_loss_increase"] = round(float(np.mean(fgsm_losses)), 5)

        # PGD
        pgd_losses = []
        for obs in obs_samples[:min(5, len(obs_samples))]:   # PGD is slow
            adv = self.grad_attacker.pgd(obs, loss_fn)
            pgd_losses.append(loss_fn(adv) - loss_fn(obs))
        results["pgd_mean_loss_increase"] = round(float(np.mean(pgd_losses)), 5)

        # LiDAR spoof
        if point_clouds:
            pc = point_clouds[0]
            spoofed = self.lidar_attacker.spoof_add(pc, np.zeros(3))
            results["lidar_phantom_points_added"] = len(spoofed) - len(pc)
            removed = self.lidar_attacker.spoof_remove(pc)
            results["lidar_points_removed_frac"] = round(
                1 - len(removed) / len(pc), 3)

        # Odom drift
        self.odom_spoofer.activate()
        odom = np.array([5.0, 3.0, 0.5])
        for _ in range(100):
            odom = self.odom_spoofer.corrupt(odom)
        results["odom_drift_after_100_steps_m"] = round(
            self.odom_spoofer.total_drift_m, 3)
        self.odom_spoofer.deactivate()

        logger.info("Robustness evaluation complete: %s", results)
        return results
