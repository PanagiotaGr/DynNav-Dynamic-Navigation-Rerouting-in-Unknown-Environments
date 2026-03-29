import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '09_multi_robot', 'code'))

import pytest
import numpy as np
from multi_robot_risk_allocation import (
    RobotState,
    CellTarget,
    build_cost_matrix,
    assign_tasks_hungarian,
    assign_with_risk_profiles,
)
from multi_robot_uncertainty import Robot, fuse_uncertainty


class TestBuildCostMatrix:
    def test_shape(self):
        robots = [RobotState(id=0, x=0.0, y=0.0), RobotState(id=1, x=5.0, y=0.0)]
        targets = [CellTarget(id=0, x=1.0, y=0.0, risk=0.1),
                   CellTarget(id=1, x=4.0, y=0.0, risk=0.9),
                   CellTarget(id=2, x=2.0, y=2.0, risk=0.5)]
        cost = build_cost_matrix(robots, targets)
        assert cost.shape == (2, 3)

    def test_all_positive(self):
        robots = [RobotState(id=0, x=0.0, y=0.0)]
        targets = [CellTarget(id=0, x=3.0, y=4.0, risk=0.5)]
        cost = build_cost_matrix(robots, targets)
        assert float(cost[0, 0]) > 0.0

    def test_zero_risk_lower_cost(self):
        robots = [RobotState(id=0, x=0.0, y=0.0)]
        t_safe = CellTarget(id=0, x=1.0, y=0.0, risk=0.0)
        t_risky = CellTarget(id=1, x=1.0, y=0.0, risk=1.0)
        cost = build_cost_matrix(robots, [t_safe, t_risky], w_dist=1.0, w_risk=1.0)
        assert float(cost[0, 0]) < float(cost[0, 1])


class TestAssignTasksHungarian:
    def test_empty_robots(self):
        targets = [CellTarget(id=0, x=1.0, y=1.0, risk=0.5)]
        assignments, total = assign_tasks_hungarian([], targets)
        assert assignments == []
        assert total == pytest.approx(0.0)

    def test_empty_targets(self):
        robots = [RobotState(id=0, x=0.0, y=0.0)]
        assignments, total = assign_tasks_hungarian(robots, [])
        assert assignments == []
        assert total == pytest.approx(0.0)

    def test_one_robot_one_target(self):
        robots = [RobotState(id=0, x=0.0, y=0.0)]
        targets = [CellTarget(id=0, x=3.0, y=4.0, risk=0.2)]
        assignments, total = assign_tasks_hungarian(robots, targets)
        assert len(assignments) == 1
        assert assignments[0][0] == 0
        assert assignments[0][1] == 0
        assert total > 0.0

    def test_each_robot_gets_one_target(self):
        robots = [RobotState(id=i, x=float(i), y=0.0) for i in range(3)]
        targets = [CellTarget(id=i, x=float(i), y=0.0, risk=0.1) for i in range(3)]
        assignments, _ = assign_tasks_hungarian(robots, targets)
        robot_ids = [a[0] for a in assignments]
        target_ids = [a[1] for a in assignments]
        assert len(set(robot_ids)) == 3
        assert len(set(target_ids)) == 3

    def test_prefers_closer_target(self):
        robots = [RobotState(id=0, x=0.0, y=0.0)]
        targets = [
            CellTarget(id=0, x=1.0, y=0.0, risk=0.0),   # close
            CellTarget(id=1, x=100.0, y=0.0, risk=0.0),  # far
        ]
        assignments, _ = assign_tasks_hungarian(robots, targets, w_dist=1.0, w_risk=0.0)
        assert assignments[0][1] == 0


class TestAssignWithRiskProfiles:
    def test_conservative_avoids_risky_target(self):
        robots = [RobotState(id=0, x=0.0, y=0.0)]
        targets = [
            CellTarget(id=0, x=1.0, y=0.0, risk=0.9),
            CellTarget(id=1, x=1.0, y=0.0, risk=0.0),
        ]
        risk_profile = {0: 10.0}  # very conservative
        assignments, _ = assign_with_risk_profiles(robots, targets, risk_profile)
        assert assignments[0][1] == 1  # picks low-risk target

    def test_empty_inputs(self):
        assignments, total = assign_with_risk_profiles([], [], {})
        assert assignments == []
        assert total == pytest.approx(0.0)


class TestFuseUncertainty:
    def test_single_robot(self):
        robots = [Robot(name="r0", drift_estimate=0.5, uncertainty_estimate=0.1)]
        drift, var = fuse_uncertainty(robots)
        assert drift == pytest.approx(0.5)
        assert var == pytest.approx(0.1)

    def test_two_identical_robots(self):
        robots = [
            Robot(name="r0", drift_estimate=0.4, uncertainty_estimate=0.2),
            Robot(name="r1", drift_estimate=0.4, uncertainty_estimate=0.2),
        ]
        drift, var = fuse_uncertainty(robots)
        assert drift == pytest.approx(0.4)
        assert var > 0.0

    def test_fused_drift_is_mean(self):
        robots = [
            Robot(name="r0", drift_estimate=0.0, uncertainty_estimate=1.0),
            Robot(name="r1", drift_estimate=1.0, uncertainty_estimate=1.0),
        ]
        drift, _ = fuse_uncertainty(robots)
        assert drift == pytest.approx(0.5)

    def test_harmonic_mean_conservative(self):
        robots = [
            Robot(name="r0", drift_estimate=0.0, uncertainty_estimate=0.1),
            Robot(name="r1", drift_estimate=0.0, uncertainty_estimate=10.0),
        ]
        _, var = fuse_uncertainty(robots)
        assert var < 10.0
