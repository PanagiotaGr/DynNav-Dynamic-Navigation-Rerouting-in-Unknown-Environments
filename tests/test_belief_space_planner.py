import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '03_belief_risk_planning', 'code'))

import pytest
import numpy as np
from belief_space_planner import (
    BeliefState,
    LinearMotionModel,
    belief_astar,
    compare_belief_vs_deterministic,
)


# ---------------------------------------------------------------------------
# BeliefState
# ---------------------------------------------------------------------------

class TestBeliefState:
    def test_uncertainty_is_trace(self):
        cov = np.diag([2.0, 3.0])
        b = BeliefState(mean=np.array([0.0, 0.0]), cov=cov)
        assert b.uncertainty == pytest.approx(5.0)

    def test_pos_rounds_mean(self):
        b = BeliefState(mean=np.array([3.6, 2.4]), cov=np.eye(2) * 0.01)
        assert b.pos == (4, 2)

    def test_copy_is_independent(self):
        b = BeliefState(mean=np.array([1.0, 2.0]), cov=np.eye(2))
        c = b.copy()
        c.mean[0] = 99.0
        assert b.mean[0] == pytest.approx(1.0)

    def test_identity_cov_uncertainty(self):
        b = BeliefState(mean=np.zeros(2), cov=np.eye(2))
        assert b.uncertainty == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# LinearMotionModel
# ---------------------------------------------------------------------------

class TestLinearMotionModel:
    def test_mean_updates_correctly(self):
        Q = np.diag([0.01, 0.01])
        model = LinearMotionModel(Q=Q)
        b = BeliefState(mean=np.array([3.0, 4.0]), cov=np.zeros((2, 2)))
        action = np.array([1.0, 0.0])
        new_b = model.propagate(b, action)
        assert new_b.mean[0] == pytest.approx(4.0)
        assert new_b.mean[1] == pytest.approx(4.0)

    def test_covariance_grows(self):
        model = LinearMotionModel()
        b = BeliefState(mean=np.zeros(2), cov=np.eye(2) * 0.01)
        u0 = b.uncertainty
        b2 = model.propagate(b, np.array([1.0, 0.0]))
        assert b2.uncertainty > u0

    def test_default_noise(self):
        model = LinearMotionModel()
        np.testing.assert_allclose(model.Q, np.diag([0.01, 0.01]))

    def test_propagate_preserves_shape(self):
        model = LinearMotionModel()
        b = BeliefState(mean=np.zeros(2), cov=np.eye(2))
        new_b = model.propagate(b, np.array([0.0, 1.0]))
        assert new_b.cov.shape == (2, 2)


# ---------------------------------------------------------------------------
# belief_astar
# ---------------------------------------------------------------------------

class TestBeliefAstar:
    def _open_grid(self, H=10, W=10):
        return np.zeros((H, W), dtype=np.int32)

    def test_finds_path_open_grid(self):
        grid = self._open_grid()
        result = belief_astar(grid, (0, 0), (9, 9), lambda_uncert=0.0)
        assert result["found"]
        assert result["path"] is not None
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (9, 9)

    def test_deterministic_lambda_zero_equals_euclidean_length(self):
        grid = self._open_grid(6, 6)
        result = belief_astar(grid, (0, 0), (5, 5), lambda_uncert=0.0)
        assert result["found"]
        # optimal on open grid is 10 steps
        assert result["path"] is not None and len(result["path"]) - 1 >= 10

    def test_uncertainty_increases_along_path(self):
        grid = self._open_grid()
        result = belief_astar(grid, (0, 0), (9, 9), lambda_uncert=1.0)
        assert result["found"]
        uncertainties = [b.uncertainty for b in result["belief_path"]]
        # Each step adds noise → uncertainty should be non-decreasing
        for i in range(len(uncertainties) - 1):
            assert uncertainties[i + 1] >= uncertainties[i] - 1e-9

    def test_lambda_uncertainty_affects_cost(self):
        grid = self._open_grid()
        r0 = belief_astar(grid, (0, 0), (9, 9), lambda_uncert=0.0)
        r2 = belief_astar(grid, (0, 0), (9, 9), lambda_uncert=2.0)
        assert r0["found"] and r2["found"]
        # Higher lambda → higher cost (uncertainty penalty)
        assert r2["cost"] >= r0["cost"] - 1e-9

    def test_not_found_when_blocked(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0:5, 2] = 1   # full wall
        result = belief_astar(grid, (0, 0), (4, 0), lambda_uncert=0.0)
        assert not result["found"]

    def test_expansions_positive(self):
        grid = self._open_grid()
        result = belief_astar(grid, (0, 0), (9, 9))
        assert result["expansions"] > 0


# ---------------------------------------------------------------------------
# compare_belief_vs_deterministic
# ---------------------------------------------------------------------------

class TestCompareBeliefVsDeterministic:
    def test_returns_list(self):
        grid = np.zeros((8, 8), dtype=np.int32)
        results = compare_belief_vs_deterministic(grid, (0, 0), (7, 7))
        assert isinstance(results, list)
        assert len(results) == 4   # default lambda_values=[0,0.5,1,2]

    def test_lambda_zero_is_deterministic(self):
        grid = np.zeros((8, 8), dtype=np.int32)
        results = compare_belief_vs_deterministic(grid, (0, 0), (7, 7), lambda_values=[0.0])
        assert results[0]["lambda"] == 0.0
        assert results[0]["found"]

    def test_all_found_on_open_grid(self):
        grid = np.zeros((8, 8), dtype=np.int32)
        results = compare_belief_vs_deterministic(grid, (0, 0), (7, 7))
        for r in results:
            assert r["found"]
