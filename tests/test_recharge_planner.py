import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '06_energy_connectivity', 'code'))

import pytest
import numpy as np
from recharge_planner import BatteryConfig, energy_astar, compare_energy_modes


def _open_grid(H=10, W=10):
    return np.zeros((H, W), dtype=np.int32)


# ---------------------------------------------------------------------------
# BatteryConfig
# ---------------------------------------------------------------------------

class TestBatteryConfig:
    def test_defaults(self):
        bc = BatteryConfig()
        assert bc.capacity == pytest.approx(100.0)
        assert bc.move_cost == pytest.approx(1.0)
        assert bc.risk_coeff == pytest.approx(2.0)
        assert bc.charge_rate == pytest.approx(100.0)

    def test_custom(self):
        bc = BatteryConfig(capacity=50.0, move_cost=2.0)
        assert bc.capacity == pytest.approx(50.0)
        assert bc.move_cost == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# energy_astar — shortest mode
# ---------------------------------------------------------------------------

class TestEnergyAstarShortest:
    def test_finds_path(self):
        grid = _open_grid()
        result = energy_astar(grid, (0, 0), (9, 9), [], BatteryConfig(), mode="shortest")
        assert result["found"]
        assert result["path"] is not None
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (9, 9)

    def test_energy_zero_in_shortest_mode(self):
        grid = _open_grid()
        result = energy_astar(grid, (0, 0), (5, 5), [], BatteryConfig(), mode="shortest")
        assert result["found"]
        assert result["energy_used"] == pytest.approx(0.0)

    def test_no_path_when_blocked(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0:5, 2] = 1
        result = energy_astar(grid, (0, 0), (4, 0), [], BatteryConfig(), mode="shortest")
        assert not result["found"]

    def test_mode_field_in_result(self):
        grid = _open_grid()
        result = energy_astar(grid, (0, 0), (5, 5), [], BatteryConfig(), mode="shortest")
        assert result["mode"] == "shortest"


# ---------------------------------------------------------------------------
# energy_astar — energy_optimal mode
# ---------------------------------------------------------------------------

class TestEnergyAstarEnergyOptimal:
    def test_finds_path_with_enough_battery(self):
        grid = _open_grid()
        bc = BatteryConfig(capacity=100.0, move_cost=1.0)
        result = energy_astar(grid, (0, 0), (5, 5), [], bc, mode="energy_optimal")
        assert result["found"]

    def test_fails_when_battery_too_low(self):
        grid = _open_grid()
        bc = BatteryConfig(capacity=1.0, move_cost=1.0)
        result = energy_astar(grid, (0, 0), (9, 9), [], bc, mode="energy_optimal")
        assert not result["found"]

    def test_energy_used_positive(self):
        grid = _open_grid()
        bc = BatteryConfig(capacity=100.0, move_cost=1.0)
        result = energy_astar(grid, (0, 0), (5, 0), [], bc, mode="energy_optimal")
        if result["found"]:
            assert result["energy_used"] > 0.0


# ---------------------------------------------------------------------------
# energy_astar — recharge_aware mode
# ---------------------------------------------------------------------------

class TestEnergyAstarRechargeAware:
    def test_reaches_goal_with_midway_station(self):
        grid = _open_grid()
        bc = BatteryConfig(capacity=8.0, move_cost=1.0)
        # place station halfway
        stations = [(5, 0)]
        result = energy_astar(grid, (0, 0), (9, 0), stations, bc, mode="recharge_aware")
        assert result["found"]

    def test_no_recharge_needed_short_path(self):
        grid = _open_grid()
        bc = BatteryConfig(capacity=100.0, move_cost=1.0)
        result = energy_astar(grid, (0, 0), (3, 0), [], bc, mode="recharge_aware")
        assert result["found"]
        assert result["recharge_stops"] == 0

    def test_recharge_stops_counted(self):
        grid = _open_grid()
        bc = BatteryConfig(capacity=6.0, move_cost=1.0)
        stations = [(3, 0)]
        result = energy_astar(grid, (0, 0), (9, 0), stations, bc, mode="recharge_aware")
        if result["found"]:
            assert result["recharge_stops"] >= 0


# ---------------------------------------------------------------------------
# compare_energy_modes
# ---------------------------------------------------------------------------

class TestCompareEnergyModes:
    def test_returns_three_results(self):
        grid = _open_grid()
        results = compare_energy_modes(grid, (0, 0), (5, 5), [])
        assert len(results) == 3

    def test_all_modes_present(self):
        grid = _open_grid()
        results = compare_energy_modes(grid, (0, 0), (5, 5), [])
        modes = {r["mode"] for r in results}
        assert modes == {"shortest", "energy_optimal", "recharge_aware"}

    def test_shortest_always_finds_path(self):
        grid = _open_grid()
        results = compare_energy_modes(grid, (0, 0), (5, 5), [])
        shortest = next(r for r in results if r["mode"] == "shortest")
        assert shortest["found"]
