echo 'class VLACostFusion:
    DEFAULT_WEIGHTS = {
        "uncertainty": 0.2,
        "coverage": 0.2,
        "obstacle_proximity": 0.2,
        "feature_density": 0.2,
        "drift_risk": 0.2
    }

    INTENT_WEIGHTS = {
        "HIGH_UNCERTAINTY": {"uncertainty": 1.0},
        "LOW_COVERAGE": {"coverage": 1.0},
        "OBSTACLE_AWARE": {"obstacle_proximity": 1.0},
        "DRIFT_STABILIZATION": {"drift_risk": 1.0}
    }

    @staticmethod
    def fuse(intent):
        weights = VLACostFusion.DEFAULT_WEIGHTS.copy()
        if intent in VLACostFusion.INTENT_WEIGHTS:
            for k, v in VLACostFusion.INTENT_WEIGHTS[intent].items():
                weights[k] = v
        return weights' > nav_research/vla/vla_cost_fusion.py
