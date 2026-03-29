import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '03_belief_risk_planning', 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '10_human_language_ethics', 'code'))

import pytest
from belief_risk_planner import SimpleRiskPlanner
from risk_budget import RiskBudgetTracker
from risk_cost_utils import path_risk_sum, path_length_l2


class TestSimpleRiskPlanner:
    def test_cost_no_risk(self):
        planner = SimpleRiskPlanner(lambda_robot=1.0)
        assert planner.evaluate_path_cost(5.0, 0.0) == pytest.approx(5.0)

    def test_cost_with_risk(self):
        planner = SimpleRiskPlanner(lambda_robot=2.0)
        assert planner.evaluate_path_cost(3.0, 1.0) == pytest.approx(5.0)

    def test_select_best_path_chooses_lowest_cost(self):
        planner = SimpleRiskPlanner(lambda_robot=1.0)
        candidates = [
            {"name": "A", "length": 10.0, "risk": 1.0},  # cost=11
            {"name": "B", "length": 5.0,  "risk": 1.0},  # cost=6  <- best
            {"name": "C", "length": 8.0,  "risk": 0.5},  # cost=8.5
        ]
        best, cost = planner.select_best_path(candidates)
        assert best["name"] == "B"
        assert cost == pytest.approx(6.0)

    def test_select_best_path_single_candidate(self):
        planner = SimpleRiskPlanner(lambda_robot=1.0)
        candidates = [{"name": "only", "length": 3.0, "risk": 2.0}]
        best, cost = planner.select_best_path(candidates)
        assert best["name"] == "only"
        assert cost == pytest.approx(5.0)

    def test_high_lambda_penalizes_risk(self):
        planner = SimpleRiskPlanner(lambda_robot=10.0)
        candidates = [
            {"name": "short_risky",  "length": 1.0, "risk": 1.0},  # cost=11
            {"name": "long_safe",    "length": 8.0, "risk": 0.0},  # cost=8  <- best
        ]
        best, _ = planner.select_best_path(candidates)
        assert best["name"] == "long_safe"


class TestRiskBudgetTracker:
    def test_initial_state(self):
        tracker = RiskBudgetTracker(B_total=10.0)
        assert tracker.B_total == pytest.approx(10.0)
        assert tracker.B_remaining == pytest.approx(10.0)
        assert tracker.spent == pytest.approx(0.0)

    def test_consume_reduces_remaining(self):
        tracker = RiskBudgetTracker(B_total=10.0)
        tracker.consume(3.0)
        assert tracker.B_remaining == pytest.approx(7.0)
        assert tracker.spent == pytest.approx(3.0)

    def test_multiple_consumes(self):
        tracker = RiskBudgetTracker(B_total=10.0)
        tracker.consume(2.0)
        tracker.consume(3.0)
        assert tracker.B_remaining == pytest.approx(5.0)
        assert tracker.spent == pytest.approx(5.0)

    def test_will_violate_true(self):
        tracker = RiskBudgetTracker(B_total=5.0)
        tracker.consume(4.0)
        assert tracker.will_violate(2.0) is True

    def test_will_violate_false(self):
        tracker = RiskBudgetTracker(B_total=5.0)
        tracker.consume(2.0)
        assert tracker.will_violate(2.0) is False

    def test_reset(self):
        tracker = RiskBudgetTracker(B_total=10.0)
        tracker.consume(5.0)
        tracker.reset(20.0)
        assert tracker.B_total == pytest.approx(20.0)
        assert tracker.B_remaining == pytest.approx(20.0)
        assert tracker.spent == pytest.approx(0.0)


class TestRiskCostUtils:
    def test_path_risk_sum_empty(self):
        import numpy as np
        grid = np.zeros((5, 5))
        assert path_risk_sum([], grid) == pytest.approx(0.0)

    def test_path_risk_sum_nonzero(self):
        import numpy as np
        grid = np.ones((5, 5))
        path = [(0, 0), (1, 0), (2, 0)]
        assert path_risk_sum(path, grid) == pytest.approx(3.0)

    def test_path_length_l2_straight(self):
        path = [(0, 0), (3, 0), (6, 0)]
        assert path_length_l2(path) == pytest.approx(6.0)

    def test_path_length_l2_diagonal(self):
        import math
        path = [(0, 0), (1, 1)]
        assert path_length_l2(path) == pytest.approx(math.sqrt(2))

    def test_path_length_l2_single_point(self):
        assert path_length_l2([(0, 0)]) == pytest.approx(0.0)

    def test_path_length_l2_empty(self):
        assert path_length_l2([]) == pytest.approx(0.0)
