import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', 'benchmarking', 'code'))

import pytest
import numpy as np
from environments import (
    make_static_env, make_dynamic_env, make_partial_map_env,
    Environment, EnvType,
)
from metrics import BenchmarkResult, count_safety_violations, timed_plan


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

class TestMakeStaticEnv:
    def test_returns_environment(self):
        env = make_static_env(H=20, W=20, seed=0)
        assert isinstance(env, Environment)
        assert env.env_type == EnvType.STATIC

    def test_grid_shape(self):
        env = make_static_env(H=15, W=20, seed=1)
        assert env.grid.shape == (15, 20)

    def test_start_goal_free(self):
        env = make_static_env(seed=0)
        sx, sy = env.start
        gx, gy = env.goal
        assert env.grid[sy, sx] == 0
        assert env.grid[gy, gx] == 0

    def test_reproducible_with_same_seed(self):
        e1 = make_static_env(H=20, W=20, seed=42)
        e2 = make_static_env(H=20, W=20, seed=42)
        np.testing.assert_array_equal(e1.grid, e2.grid)
        assert e1.start == e2.start
        assert e1.goal == e2.goal

    def test_different_seeds_differ(self):
        e1 = make_static_env(H=20, W=20, seed=0)
        e2 = make_static_env(H=20, W=20, seed=99)
        # Very likely to differ
        assert not np.array_equal(e1.grid, e2.grid) or e1.start != e2.start


class TestMakePartialMapEnv:
    def test_initially_mostly_unknown(self):
        env = make_partial_map_env(H=20, W=20, sensor_radius=2, seed=0)
        n_known = int(np.sum(env.known_grid != -1))
        n_total = 20 * 20
        # with sensor_radius=2, only a patch revealed
        assert n_known < n_total

    def test_start_area_revealed(self):
        env = make_partial_map_env(H=20, W=20, sensor_radius=3, seed=0)
        sx, sy = env.start
        assert env.known_grid[sy, sx] != -1

    def test_update_known_map_reveals_cells(self):
        env = make_partial_map_env(H=20, W=20, sensor_radius=2, seed=0)
        before = int(np.sum(env.known_grid != -1))
        # move to a new cell
        new_pos = (10, 10)
        env.known_grid[10, 10] = -1   # ensure it's unknown
        env.update_known_map(new_pos)
        after = int(np.sum(env.known_grid != -1))
        assert after >= before

    def test_sensor_radius_stored(self):
        env = make_partial_map_env(sensor_radius=5, seed=0)
        assert env.sensor_radius == 5


class TestEnvironmentProperties:
    def test_hw_properties(self):
        env = make_static_env(H=12, W=15, seed=0)
        assert env.H == 12
        assert env.W == 15

    def test_known_grid_equals_grid_for_static(self):
        env = make_static_env(H=10, W=10, seed=0)
        np.testing.assert_array_equal(env.grid, env.known_grid)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestCountSafetyViolations:
    def test_no_violations_open_grid(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        path = [(2, 2)]   # center of open grid
        assert count_safety_violations(path, grid) == 0

    def test_adjacent_to_obstacle(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, 3] = 1   # obstacle at (3, 2)
        path = [(2, 2)]  # adjacent to obstacle
        assert count_safety_violations(path, grid) == 1

    def test_empty_path(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        assert count_safety_violations([], grid) == 0

    def test_multiple_violations(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[1, 2] = 1
        grid[3, 2] = 1
        path = [(2, 1), (2, 2), (2, 3)]
        violations = count_safety_violations(path, grid)
        assert violations >= 1

    def test_path_not_adjacent(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[0, 0] = 1
        path = [(5, 5), (6, 5), (7, 5)]
        assert count_safety_violations(path, grid) == 0


class TestBenchmarkResult:
    def test_as_dict_keys(self):
        r = BenchmarkResult(
            method="astar", env_type="static", found=True,
            path_cost=10.0, optimal_path_cost=10.0, suboptimality=1.0,
            safety_violations=0, computation_time_ms=5.0,
            node_expansions=50, seed=0,
        )
        d = r.as_dict()
        assert "method" in d
        assert "found" in d
        assert "suboptimality" in d
        assert "node_expansions" in d

    def test_as_dict_found_is_int(self):
        r = BenchmarkResult(
            method="x", env_type="static", found=True,
            path_cost=5.0, optimal_path_cost=5.0, suboptimality=1.0,
            safety_violations=0, computation_time_ms=1.0,
            node_expansions=10, seed=0,
        )
        assert r.as_dict()["found"] == 1


class TestTimedPlan:
    def test_returns_result_and_time(self):
        def dummy():
            return 42
        result, ms = timed_plan(dummy)
        assert result == 42
        assert ms >= 0.0

    def test_time_positive(self):
        import time
        def slow():
            time.sleep(0.01)
            return "done"
        _, ms = timed_plan(slow)
        assert ms >= 5.0   # at least 5ms
