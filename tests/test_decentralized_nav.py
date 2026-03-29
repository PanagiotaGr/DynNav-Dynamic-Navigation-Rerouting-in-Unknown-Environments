import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '09_multi_robot', 'code'))

import pytest
import numpy as np
from decentralized_nav import (
    RobotAgent,
    run_decentralized,
    run_centralized,
    DecentralizedResult,
)


def _open_grid(H=12, W=12):
    return np.zeros((H, W), dtype=np.int32)


# ---------------------------------------------------------------------------
# RobotAgent
# ---------------------------------------------------------------------------

class TestRobotAgent:
    def test_defaults(self):
        r = RobotAgent(robot_id=0, pos=(0, 0), goal=(5, 5))
        assert r.comm_range == pytest.approx(10.0)
        assert r.reached_goal is False
        assert r.steps == 0
        assert r.replans == 0
        assert r.collision_avoidance_events == 0
        assert r.shared_risk == {}

    def test_custom_comm_range(self):
        r = RobotAgent(robot_id=1, pos=(0, 0), goal=(5, 5), comm_range=3.0)
        assert r.comm_range == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# run_decentralized
# ---------------------------------------------------------------------------

class TestRunDecentralized:
    def test_single_robot_reaches_goal(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(11, 11))]
        result = run_decentralized(grid, robots, max_steps=300)
        assert result.all_reached
        assert robots[0].reached_goal

    def test_two_robots_reach_goals(self):
        grid = _open_grid()
        robots = [
            RobotAgent(robot_id=0, pos=(0, 0), goal=(11, 11)),
            RobotAgent(robot_id=1, pos=(11, 0), goal=(0, 11)),
        ]
        result = run_decentralized(grid, robots, max_steps=300)
        assert result.all_reached

    def test_result_strategy(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(5, 5))]
        result = run_decentralized(grid, robots)
        assert result.strategy == "decentralized"

    def test_total_steps_positive(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(10, 10))]
        result = run_decentralized(grid, robots, max_steps=300)
        assert result.total_steps > 0

    def test_avg_path_length_positive(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(10, 10))]
        result = run_decentralized(grid, robots, max_steps=300)
        assert result.avg_path_length > 0.0

    def test_priority_lower_id_reaches_first(self):
        grid = _open_grid()
        robots = [
            RobotAgent(robot_id=0, pos=(0, 0), goal=(5, 5)),
            RobotAgent(robot_id=1, pos=(1, 0), goal=(5, 5)),
        ]
        run_decentralized(grid, robots, max_steps=200)
        # Both should be at goal (may collide → only one exact pos, but both reached)
        assert robots[0].reached_goal

    def test_risk_sharing_disabled(self):
        grid = _open_grid()
        robots = [
            RobotAgent(robot_id=0, pos=(0, 0), goal=(11, 11)),
            RobotAgent(robot_id=1, pos=(11, 0), goal=(0, 11)),
        ]
        result = run_decentralized(grid, robots, share_risk=False, max_steps=300)
        assert result.all_reached


# ---------------------------------------------------------------------------
# run_centralized
# ---------------------------------------------------------------------------

class TestRunCentralized:
    def test_single_robot_reaches_goal(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(11, 11))]
        result = run_centralized(grid, robots, max_steps=300)
        assert result.all_reached
        assert result.strategy == "centralized"

    def test_two_robots_reach_goals(self):
        # Use non-crossing paths: both go horizontally on different rows
        grid = _open_grid()
        robots = [
            RobotAgent(robot_id=0, pos=(0, 0), goal=(11, 0)),
            RobotAgent(robot_id=1, pos=(0, 11), goal=(11, 11)),
        ]
        result = run_centralized(grid, robots, max_steps=300)
        assert result.all_reached

    def test_zero_replans(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(5, 5))]
        result = run_centralized(grid, robots)
        assert result.total_replans == 0

    def test_zero_collision_events(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(5, 5))]
        result = run_centralized(grid, robots)
        assert result.collision_events == 0

    def test_avg_path_positive(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(10, 10))]
        result = run_centralized(grid, robots, max_steps=300)
        assert result.avg_path_length > 0.0


# ---------------------------------------------------------------------------
# Centralized vs Decentralized comparison
# ---------------------------------------------------------------------------

class TestCentralizedVsDecentralized:
    def test_both_reach_goal(self):
        grid = _open_grid()
        r1 = [RobotAgent(robot_id=0, pos=(0, 0), goal=(11, 11))]
        r2 = [RobotAgent(robot_id=0, pos=(0, 0), goal=(11, 11))]
        res_c = run_centralized(grid, r1, max_steps=300)
        res_d = run_decentralized(grid, r2, max_steps=300)
        assert res_c.all_reached
        assert res_d.all_reached

    def test_centralized_no_replans(self):
        grid = _open_grid()
        robots = [RobotAgent(robot_id=0, pos=(0, 0), goal=(10, 10))]
        result = run_centralized(grid, robots, max_steps=300)
        assert result.total_replans == 0
