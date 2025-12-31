"""
self_trust_manager.py

Online Self-Trust estimation + Safe Mode triggering
για multi-robot navigation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SelfTrustState:
    robot_id: int
    S: float = 1.0
    safe_mode: bool = False


class SelfTrustManager:
    def __init__(
        self,
        alpha_calib: float = 0.6,   # πόσο μετρά η κακή calibration
        alpha_drift: float = 0.4,   # πόσο μετρά το drift mismatch
        safe_threshold: float = 0.5
    ):
        self.alpha_calib = alpha_calib
        self.alpha_drift = alpha_drift
        self.safe_threshold = safe_threshold
        self.states = {}

    def register_robot(self, robot_id: int):
        self.states[robot_id] = SelfTrustState(robot_id=robot_id)

    def update(
        self,
        robot_id: int,
        calibration_error: float,
        expected_drift: float,
        observed_drift: float
    ):
        """
        Parameters
        ----------
        calibration_error : float
            π.χ. NLL / ECE / Z-score variance error ∈ [0,1]
        expected_drift : float
            predicted drift από model
        observed_drift : float
            πραγματικό drift (UKF / ground truth / pose error)
        """

        if robot_id not in self.states:
            self.register_robot(robot_id)

        st = self.states[robot_id]

        drift_mismatch = max(0.0, (observed_drift - expected_drift))
        drift_mismatch = min(drift_mismatch, 1.0)

        S = 1.0 - (
            self.alpha_calib * calibration_error +
            self.alpha_drift * drift_mismatch
        )

        S = float(np.clip(S, 0.0, 1.0))

        st.S = S
        st.safe_mode = S < self.safe_threshold

        return st.S, st.safe_mode

    def get_state(self, robot_id: int):
        return self.states.get(robot_id, None)
