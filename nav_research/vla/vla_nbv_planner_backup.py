import numpy as np
from scipy.ndimage import distance_transform_edt
from nav_research.vla.vla_intents import VLAIntents
from nav_research.vla.vla_cost_fusion import VLACostFusion

class VLA_NBV_Planner:
    def __init__(self, coverage, uncertainty, features, drift, obstacles):
        self.coverage = coverage
        self.uncertainty = uncertainty
        self.features = features
        self.drift = drift
        self.obstacle_dist = distance_transform_edt(1 - obstacles)

    def _norm(self, x):
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    def pick_nbv(self, command):
        intent = VLAIntents.classify(command)
        weights = VLACostFusion.fuse(intent)

        cov = self._norm(1 - self.coverage)
        unc = self._norm(self.uncertainty)
        feat = self._norm(self.features)
        drift = self._norm(self.drift)
        obs = self._norm(self.obstacle_dist)

        cost = (weights["coverage"] * cov +
                weights["uncertainty"] * unc +
                weights["feature_density"] * (1 - feat) +
                weights["drift_risk"] * drift +
                weights["obstacle_proximity"] * obs)

        y, x = np.unravel_index(np.argmax(cost), cost.shape)
        return (x, y), cost
