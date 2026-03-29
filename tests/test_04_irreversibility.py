import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '04_irreversibility_returnability', 'code'))

import pytest
import numpy as np
from irreversibility_map import (
    IrreversibilityConfig,
    _normalize01,
    estimate_deadend_score,
)


class TestNormalize01:
    def test_all_zeros(self):
        a = np.zeros((3, 3))
        result = _normalize01(a)
        assert np.allclose(result, 0.0)

    def test_all_same_value(self):
        a = np.full((3, 3), 5.0)
        result = _normalize01(a)
        assert np.allclose(result, 0.0)

    def test_min_max_range(self):
        a = np.array([0.0, 5.0, 10.0])
        result = _normalize01(a)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_output_range(self):
        rng = np.random.default_rng(42)
        a = rng.random((10, 10)) * 100
        result = _normalize01(a)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0


class TestEstimateDeadendScore:
    def test_fully_free_grid(self):
        free = np.ones((10, 10), dtype=bool)
        score = estimate_deadend_score(free, radius=2)
        assert score.shape == (10, 10)
        assert float(score.max()) <= 1.0
        assert float(score.min()) >= 0.0

    def test_fully_blocked_grid(self):
        free = np.zeros((10, 10), dtype=bool)
        score = estimate_deadend_score(free, radius=2)
        assert score.shape == (10, 10)

    def test_score_higher_near_walls(self):
        free = np.ones((20, 20), dtype=bool)
        free[:, 0] = False
        free[:, -1] = False
        free[0, :] = False
        free[-1, :] = False
        score = estimate_deadend_score(free, radius=2)
        center_score = float(score[10, 10])
        corner_score = float(score[1, 1])
        assert corner_score >= center_score


class TestIrreversibilityConfig:
    def test_default_weights_sum_to_one(self):
        cfg = IrreversibilityConfig()
        total = cfg.w_uncert + cfg.w_sparsity + cfg.w_deadend
        assert total == pytest.approx(1.0)

    def test_custom_config(self):
        cfg = IrreversibilityConfig(w_uncert=0.5, w_sparsity=0.3, w_deadend=0.2)
        assert cfg.w_uncert == pytest.approx(0.5)
        assert cfg.w_deadend == pytest.approx(0.2)
