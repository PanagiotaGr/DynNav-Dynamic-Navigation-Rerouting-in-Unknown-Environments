import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', 'ablation_study', 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', 'benchmarking', 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', 'realtime_replanning', 'code'))

import pytest
import numpy as np
from ablation_runner import (
    AblationConfig,
    AblationResult,
    _ablated_astar,
    print_ablation_table,
)


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------

class TestAblationConfig:
    def test_baseline_name(self):
        cfg = AblationConfig(use_risk=False, use_uncertainty=False, use_learned=False)
        assert cfg.name == "baseline"

    def test_risk_only_name(self):
        cfg = AblationConfig(use_risk=True, use_uncertainty=False, use_learned=False)
        assert "risk" in cfg.name

    def test_all_enabled_name(self):
        cfg = AblationConfig(use_risk=True, use_uncertainty=True, use_learned=True)
        for part in ("risk", "learned", "uncert"):
            assert part in cfg.name

    def test_all_combinations_count(self):
        combos = AblationConfig.all_combinations()
        assert len(combos) == 8   # 2^3

    def test_all_combinations_unique(self):
        combos = AblationConfig.all_combinations()
        names = [c.name for c in combos]
        assert len(names) == len(set(names))

    def test_all_combinations_cover_all_flags(self):
        combos = AblationConfig.all_combinations()
        risk_vals = {c.use_risk for c in combos}
        unc_vals = {c.use_uncertainty for c in combos}
        lrn_vals = {c.use_learned for c in combos}
        assert risk_vals == {True, False}
        assert unc_vals == {True, False}
        assert lrn_vals == {True, False}


# ---------------------------------------------------------------------------
# AblationResult suboptimality
# ---------------------------------------------------------------------------

class TestAblationResult:
    def test_suboptimality_ratio(self):
        r = AblationResult(
            config=AblationConfig(False, False, False),
            seed=0, found=True,
            path_cost=12.0, optimal_cost=10.0,
            node_expansions=50, safety_violations=0,
        )
        assert r.suboptimality == pytest.approx(1.2)

    def test_suboptimality_inf_when_not_found(self):
        r = AblationResult(
            config=AblationConfig(False, False, False),
            seed=0, found=False,
            path_cost=0.0, optimal_cost=10.0,
            node_expansions=0, safety_violations=0,
        )
        assert r.suboptimality == float("inf")

    def test_suboptimality_one_on_optimal(self):
        r = AblationResult(
            config=AblationConfig(False, False, False),
            seed=0, found=True,
            path_cost=10.0, optimal_cost=10.0,
            node_expansions=50, safety_violations=0,
        )
        assert r.suboptimality == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _ablated_astar
# ---------------------------------------------------------------------------

class TestAblatedAstar:
    def _open_grid(self):
        return np.zeros((10, 10), dtype=np.int32)

    def test_baseline_finds_path(self):
        grid = self._open_grid()
        path, exp = _ablated_astar(grid, (0, 0), (9, 9), False, False, False)
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    def test_with_risk_finds_path(self):
        grid = self._open_grid()
        path, exp = _ablated_astar(grid, (0, 0), (9, 9), True, False, False)
        assert path is not None

    def test_expansions_positive(self):
        grid = self._open_grid()
        _, exp = _ablated_astar(grid, (0, 0), (9, 9), False, False, False)
        assert exp > 0

    def test_blocked_goal_returns_none(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0:5, 3] = 1   # complete wall
        path, _ = _ablated_astar(grid, (0, 0), (4, 0), False, False, False)
        assert path is None

    def test_path_has_no_obstacles(self):
        grid = self._open_grid()
        grid[5, 0:8] = 1
        path, _ = _ablated_astar(grid, (0, 0), (9, 9), False, False, False)
        if path:
            for x, y in path:
                assert grid[y, x] == 0


# ---------------------------------------------------------------------------
# print_ablation_table (smoke test)
# ---------------------------------------------------------------------------

class TestPrintAblationTable:
    def test_smoke(self, capsys):
        results = [
            AblationResult(
                config=AblationConfig(False, False, False),
                seed=0, found=True,
                path_cost=10.0, optimal_cost=10.0,
                node_expansions=30, safety_violations=0,
            ),
            AblationResult(
                config=AblationConfig(True, False, False),
                seed=0, found=True,
                path_cost=11.0, optimal_cost=10.0,
                node_expansions=28, safety_violations=0,
            ),
        ]
        print_ablation_table(results)   # should not raise
        captured = capsys.readouterr()
        assert "baseline" in captured.out
