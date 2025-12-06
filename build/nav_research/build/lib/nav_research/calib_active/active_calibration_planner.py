# active_calibration_planner.py
import numpy as np
from .observability_metrics import observability_score

class ActiveCalibrationPlanner:
    def __init__(self):
        # candidate motion primitives
        self.motion_primitives = [
            np.array([0.3, 0, 0]),     # forward
            np.array([0, 0.3, 0]),     # sideways right
            np.array([0, -0.3, 0]),    # sideways left
            np.array([0, 0, 0.3]),     # rotate
        ]

    def choose_best_motion(self, H_candidates):
        """
        H_candidates: list of Jacobians for each motion primitive
        """
        scores = [observability_score(H) for H in H_candidates]
        best_idx = int(np.argmax(scores))
        return self.motion_primitives[best_idx], scores[best_idx]
