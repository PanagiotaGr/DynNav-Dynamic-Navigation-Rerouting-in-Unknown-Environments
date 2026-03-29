import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import pytest
from vla_cost_fusion import VLACostFusion
from vla_intents import VLAIntents


class TestVLACostFusion:
    def test_default_weights_sum_to_one(self):
        weights = VLACostFusion.DEFAULT_WEIGHTS
        total = sum(weights.values())
        assert total == pytest.approx(1.0)

    def test_fuse_unknown_intent_returns_defaults(self):
        weights = VLACostFusion.fuse("UNKNOWN_INTENT")
        assert weights == VLACostFusion.DEFAULT_WEIGHTS

    def test_fuse_high_uncertainty_sets_uncertainty_to_one(self):
        weights = VLACostFusion.fuse("HIGH_UNCERTAINTY")
        assert weights["uncertainty"] == pytest.approx(1.0)

    def test_fuse_low_coverage_sets_coverage_to_one(self):
        weights = VLACostFusion.fuse("LOW_COVERAGE")
        assert weights["coverage"] == pytest.approx(1.0)

    def test_fuse_obstacle_aware(self):
        weights = VLACostFusion.fuse("OBSTACLE_AWARE")
        assert weights["obstacle_proximity"] == pytest.approx(1.0)

    def test_fuse_drift_stabilization(self):
        weights = VLACostFusion.fuse("DRIFT_STABILIZATION")
        assert weights["drift_risk"] == pytest.approx(1.0)

    def test_fuse_does_not_modify_class_defaults(self):
        VLACostFusion.fuse("HIGH_UNCERTAINTY")
        assert VLACostFusion.DEFAULT_WEIGHTS["uncertainty"] == pytest.approx(0.2)


class TestVLAIntents:
    def test_uncertainty_keyword(self):
        assert VLAIntents.classify("area is very risky") == "HIGH_UNCERTAINTY"

    def test_uncertainty_word(self):
        assert VLAIntents.classify("high uncertainty zone") == "HIGH_UNCERTAINTY"

    def test_coverage_keyword(self):
        assert VLAIntents.classify("uncovered region ahead") == "LOW_COVERAGE"

    def test_obstacle_keyword(self):
        assert VLAIntents.classify("there is a wall") == "OBSTACLE_AWARE"

    def test_drift_keyword(self):
        assert VLAIntents.classify("robot is drifting") == "DRIFT_STABILIZATION"

    def test_default_on_unknown(self):
        assert VLAIntents.classify("navigate to goal") == "DEFAULT"

    def test_case_insensitive(self):
        assert VLAIntents.classify("RISKY PATH") == "HIGH_UNCERTAINTY"
        assert VLAIntents.classify("WALL AHEAD") == "OBSTACLE_AWARE"
