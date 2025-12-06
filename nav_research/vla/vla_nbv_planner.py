import numpy as np
from scipy.ndimage import distance_transform_edt
from nav_research.vla.vla_intents import VLAIntents
from nav_research.vla.vla_cost_fusion import VLACostFusion


class VLA_NBV_Planner:
    """
    Computes a Next-Best-View (NBV) goal based on natural-language instructions.
    
    Parameters
    ----------
    coverage : np.ndarray
        Grid with values in [0,1] where 1 = fully covered, 0 = uncovered.
    uncertainty : np.ndarray
        VO or SLAM uncertainty measure per grid cell.
    features : np.ndarray
        Local feature density map (e.g., ORB keypoints).
    drift : np.ndarray
        Drift / error accumulation per region.
    obstacles : np.ndarray
        Binary grid: 1 = obstacle, 0 = free.
    """

    def __init__(self, coverage, uncertainty, features, drift, obstacles):
        self.coverage = coverage
        self.uncertainty = uncertainty
        self.features = features
        self.drift = drift
        self.obstacle_dist = distance_transform_edt(1 - obstacles)

    def _normalize(self, arr):
        """Normalize array to [0,1] range. If flat, return zeros."""
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    def pick_nbv(self, command):
        """
        Returns:
            (goal_x, goal_y), costmap
        """
        # STEP 1 — interpret language
        intent = VLAIntents.classify(command)

        # STEP 2 — load weights for this intent
        weights = VLACostFusion.fuse(intent)

        # STEP 3 — normalize input maps
        cov = self._normalize(1 - self.coverage)     # want low coverage → explore
        unc = self._normalize(self.uncertainty)
        feat = self._normalize(self.features)
        drift = self._normalize(self.drift)
        obs = self._normalize(self.obstacle_dist)

        # STEP 4 — weighted costmap
        cost = (
            weights["coverage"] * cov +
            weights["uncertainty"] * unc +
            weights["feature_density"] * (1 - feat) +
            weights["drift_risk"] * drift +
            weights["obstacle_proximity"] * obs
        )

        # STEP 5 — pick highest scoring cell
        y, x = np.unravel_index(np.argmax(cost), cost.shape)
        return (x, y), cost
