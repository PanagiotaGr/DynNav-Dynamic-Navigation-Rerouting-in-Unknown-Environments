import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '07_nbv_exploration', 'code'))

import pytest
import math
from info_gain_planner import cell_entropy


class TestCellEntropy:
    def test_max_entropy_at_half(self):
        h = cell_entropy(0.5)
        assert h == pytest.approx(1.0, abs=1e-6)

    def test_near_zero_uncertainty(self):
        h = cell_entropy(0.001)
        assert h < 0.02

    def test_near_one_uncertainty(self):
        h = cell_entropy(0.999)
        assert h < 0.02

    def test_entropy_is_symmetric(self):
        h1 = cell_entropy(0.3)
        h2 = cell_entropy(0.7)
        assert h1 == pytest.approx(h2, abs=1e-9)

    def test_entropy_always_positive(self):
        for p in [0.1, 0.2, 0.5, 0.8, 0.9]:
            assert cell_entropy(p) > 0.0

    def test_entropy_bounded_by_one(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert cell_entropy(p) <= 1.0 + 1e-9
