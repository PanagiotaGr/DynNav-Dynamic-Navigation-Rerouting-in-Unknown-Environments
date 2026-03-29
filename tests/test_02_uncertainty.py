import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '02_uncertainty_calibration', 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '07_nbv_exploration', 'code'))

import pytest
import numpy as np
import pandas as pd
import tempfile
from drift_dataset_builder import compute_vo_drift, extract_local_entropy, extract_local_uncertainty


class TestComputeVoDrift:
    def _make_df(self, xs, ys):
        return pd.DataFrame({"x": xs, "y": ys})

    def test_single_row_drift_zero(self):
        df = self._make_df([0.0], [0.0])
        drifts = compute_vo_drift(df)
        assert len(drifts) == 1
        assert drifts[0] == pytest.approx(0.0)

    def test_straight_line_low_drift(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 0.0, 0.0, 0.0, 0.0]
        df = self._make_df(xs, ys)
        drifts = compute_vo_drift(df)
        assert len(drifts) == 5
        for d in drifts[2:]:
            assert d == pytest.approx(0.0, abs=1e-9)

    def test_sudden_direction_change_increases_drift(self):
        xs = [0.0, 1.0, 2.0, 2.0, 2.0]
        ys = [0.0, 0.0, 0.0, 1.0, 2.0]
        df = self._make_df(xs, ys)
        drifts = compute_vo_drift(df)
        assert drifts[3] > 0.0

    def test_output_length_matches_input(self):
        n = 10
        df = self._make_df(list(range(n)), [0.0] * n)
        drifts = compute_vo_drift(df)
        assert len(drifts) == n


class TestExtractLocalEntropy:
    def test_valid_position_returns_value(self):
        H = np.full((10, 10), 0.5)
        val = extract_local_entropy(H, x=3.0, y=4.0)
        assert val == pytest.approx(0.5)

    def test_out_of_bounds_returns_zero(self):
        H = np.ones((5, 5))
        val = extract_local_entropy(H, x=100.0, y=100.0)
        assert val == pytest.approx(0.0)

    def test_rounding_to_nearest_cell(self):
        H = np.zeros((10, 10))
        H[3, 4] = 0.9
        val = extract_local_entropy(H, x=4.4, y=3.4)
        assert val == pytest.approx(0.9)


class TestExtractLocalUncertainty:
    def test_valid_position(self):
        U = np.full((10, 10), 0.3)
        val = extract_local_uncertainty(U, x=2.0, y=5.0)
        assert val == pytest.approx(0.3)

    def test_out_of_bounds_returns_zero(self):
        U = np.ones((5, 5))
        val = extract_local_uncertainty(U, x=-1.0, y=-1.0)
        assert val == pytest.approx(0.0)
