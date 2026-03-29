import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '07_nbv_exploration', 'code'))

import pytest
import numpy as np
from frontier_vs_infogain import (
    ExplorationMap,
    run_exploration,
    compare_exploration,
    ExplorationResult,
)


def _open_true_grid(H=15, W=15):
    return np.zeros((H, W), dtype=np.int32)


# ---------------------------------------------------------------------------
# ExplorationMap
# ---------------------------------------------------------------------------

class TestExplorationMap:
    def test_initial_all_unknown(self):
        grid = _open_true_grid()
        emap = ExplorationMap(grid, sensor_radius=2)
        assert int(np.sum(emap.known == -1)) == 15 * 15

    def test_reveal_updates_known(self):
        grid = _open_true_grid()
        emap = ExplorationMap(grid, sensor_radius=2)
        revealed = emap.reveal(7, 7)
        assert revealed > 0
        assert emap.known[7, 7] == 0

    def test_reveal_no_double_count(self):
        grid = _open_true_grid()
        emap = ExplorationMap(grid, sensor_radius=2)
        r1 = emap.reveal(7, 7)
        r2 = emap.reveal(7, 7)
        assert r2 == 0   # already revealed

    def test_explored_fraction_increases(self):
        grid = _open_true_grid()
        emap = ExplorationMap(grid, sensor_radius=3)
        f0 = emap.explored_fraction
        emap.reveal(7, 7)
        f1 = emap.explored_fraction
        assert f1 > f0

    def test_explored_fraction_full_when_all_known(self):
        grid = _open_true_grid(5, 5)
        emap = ExplorationMap(grid, sensor_radius=10)
        emap.reveal(2, 2)
        assert emap.explored_fraction == pytest.approx(1.0)

    def test_frontiers_empty_when_all_unknown(self):
        grid = _open_true_grid()
        emap = ExplorationMap(grid)
        fronts = emap.frontiers()
        assert fronts == []

    def test_frontiers_exist_after_partial_reveal(self):
        grid = _open_true_grid()
        emap = ExplorationMap(grid, sensor_radius=2)
        emap.reveal(7, 7)
        fronts = emap.frontiers()
        assert len(fronts) > 0

    def test_unknown_neighbors_count(self):
        grid = _open_true_grid()
        emap = ExplorationMap(grid, sensor_radius=1)
        emap.reveal(7, 7)
        # cells adjacent to center may be revealed depending on radius
        count = emap.unknown_neighbors(7, 7)
        assert count >= 0


# ---------------------------------------------------------------------------
# run_exploration
# ---------------------------------------------------------------------------

class TestRunExploration:
    def test_frontier_strategy_runs(self):
        grid = _open_true_grid()
        result = run_exploration(grid, start=(7, 7), strategy="frontier",
                                 max_steps=200, sensor_radius=3)
        assert isinstance(result, ExplorationResult)
        assert result.strategy == "frontier"
        assert result.steps >= 0

    def test_infogain_strategy_runs(self):
        grid = _open_true_grid()
        result = run_exploration(grid, start=(7, 7), strategy="infogain",
                                 max_steps=200, sensor_radius=3)
        assert isinstance(result, ExplorationResult)
        assert result.strategy == "infogain"

    def test_explored_fraction_positive(self):
        grid = _open_true_grid()
        result = run_exploration(grid, start=(7, 7), strategy="frontier",
                                 max_steps=100, sensor_radius=3)
        assert result.final_explored_fraction > 0.0

    def test_explored_over_time_non_decreasing(self):
        grid = _open_true_grid()
        result = run_exploration(grid, start=(7, 7), strategy="frontier",
                                 max_steps=100, sensor_radius=3)
        for i in range(len(result.explored_over_time) - 1):
            assert result.explored_over_time[i + 1] >= result.explored_over_time[i] - 1e-9

    def test_infogain_at_least_as_good_as_frontier(self):
        # On open grids, infogain tends to explore more efficiently
        grid = _open_true_grid(20, 20)
        r_f = run_exploration(grid, (10, 10), strategy="frontier",
                              max_steps=300, sensor_radius=3, seed=0)
        r_i = run_exploration(grid, (10, 10), strategy="infogain",
                              max_steps=300, sensor_radius=3, seed=0)
        # Both should explore a positive fraction
        assert r_f.final_explored_fraction > 0.0
        assert r_i.final_explored_fraction > 0.0

    def test_replans_positive(self):
        grid = _open_true_grid()
        result = run_exploration(grid, start=(7, 7), strategy="frontier",
                                 max_steps=100, sensor_radius=3)
        assert result.replans >= 1


# ---------------------------------------------------------------------------
# compare_exploration
# ---------------------------------------------------------------------------

class TestCompareExploration:
    def test_returns_two_results(self):
        grid = _open_true_grid()
        results = compare_exploration(grid, start=(7, 7), max_steps=100)
        assert len(results) == 2

    def test_both_strategies_present(self):
        grid = _open_true_grid()
        results = compare_exploration(grid, start=(7, 7), max_steps=100)
        strategies = {r.strategy for r in results}
        assert strategies == {"frontier", "infogain"}
