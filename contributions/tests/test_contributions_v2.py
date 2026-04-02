"""
Tests for DynNav contributions 19-26
=====================================
Run from repo root:
    pytest contributions/tests/test_contributions_v2.py -v
"""
import sys, os, glob
import numpy as np
import pytest

BASE = os.path.dirname(__file__)
for pattern in [f"../{i:02d}_*" for i in range(19, 27)]:
    for path in glob.glob(os.path.join(BASE, pattern)):
        sys.path.insert(0, path)


# ============================================================
# 19 — LLM Mission Planner
# ============================================================
class TestLLMMissionPlanner:
    def setup_method(self):
        from llm_mission_planner import LLMMissionPlanner, LLMPlannerConfig
        self.planner = LLMMissionPlanner(LLMPlannerConfig())

    def test_keyword_fallback_finds_zones(self):
        m = self.planner._keyword_fallback("go to the kitchen then the corridor")
        assert any(w.label == "kitchen" for w in m.waypoints)
        assert any(w.label == "corridor" for w in m.waypoints)

    def test_waypoints_ordered(self):
        m = self.planner._keyword_fallback("kitchen then living_room then exit")
        priorities = [w.priority for w in m.waypoints]
        assert priorities == sorted(priorities)

    def test_parse_valid_json(self):
        from llm_mission_planner import LLMMissionPlanner
        raw = '{"confidence":0.9,"waypoints":[{"label":"kitchen","priority":1,"action":"navigate","duration_s":0}]}'
        m = self.planner._parse_llm_response("go to kitchen", raw)
        assert m is not None
        assert m.waypoints[0].label == "kitchen"

    def test_resolve_metric(self):
        zone_map = {"kitchen": (2.0, 3.0), "corridor": (5.0, 1.0)}
        m = self.planner._keyword_fallback("kitchen and corridor")
        m = self.planner.resolve_to_metric(m, zone_map)
        for wp in m.waypoints:
            assert wp.metric_xy is not None

    def test_mission_to_dict(self):
        m = self.planner._keyword_fallback("go to exit")
        d = m.to_dict()
        assert "waypoints" in d and "confidence" in d

    def test_edit_distance(self):
        from llm_mission_planner import _edit_distance
        assert _edit_distance("kitchen", "kitchen") == 0
        assert _edit_distance("kitchen", "kichen") == 1


# ============================================================
# 20 — Multimodal Failure Explainer
# ============================================================
class TestFailureExplainer:
    def setup_method(self):
        from multimodal_failure_explainer import (
            MultimodalFailureExplainer, FailureEvent, FailureType
        )
        self.explainer = MultimodalFailureExplainer(use_vlm=False, use_causal=True)
        self.event = FailureEvent(
            failure_type=FailureType.COLLISION,
            timestamp=12.5,
            robot_pos=(3.2, 4.1),
            robot_vel=(0.3, 0.0),
            sensor_readings={"min_obstacle_dist": 0.15},
        )

    def test_explain_returns_report(self):
        report = self.explainer.explain(self.event)
        assert report.failure_type == "collision"
        assert len(report.corrective_actions) > 0

    def test_markdown_output(self):
        report = self.explainer.explain(self.event)
        md = report.to_markdown()
        assert "# Failure Report" in md
        assert "Root Causes" in md

    def test_dict_output(self):
        report = self.explainer.explain(self.event)
        d = report.to_dict()
        assert "root_causes" in d and "corrective_actions" in d

    def test_stl_summary_no_data(self):
        from multimodal_failure_explainer import FailureEvent, FailureType
        ev = FailureEvent(FailureType.TIMEOUT, 5.0, (1.0, 1.0), (0.0, 0.0))
        report = self.explainer.explain(ev)
        assert "No STL data" in report.stl_summary


# ============================================================
# 21 — PPO Navigation Agent
# ============================================================
class TestPPOAgent:
    def setup_method(self):
        from ppo_nav_agent import PPOAgent, PPOConfig, NavEnv
        self.cfg = PPOConfig(obs_dim=14, hidden_dim=32, rollout_len=20)
        self.agent = PPOAgent(self.cfg)
        self.env = NavEnv(self.cfg)

    def test_env_reset_shape(self):
        obs = self.env.reset()
        assert obs.shape == (self.cfg.obs_dim,)

    def test_env_step(self):
        self.env.reset()
        action = np.zeros(2)
        obs, reward, done, info = self.env.step(action)
        assert obs.shape == (self.cfg.obs_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_select_action(self):
        obs = self.env.reset()
        action, log_prob, value = self.agent.select_action(obs)
        assert action.shape == (2,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_rollout_buffer_gae(self):
        from ppo_nav_agent import RolloutBuffer
        buf = RolloutBuffer()
        for _ in range(10):
            buf.add(np.zeros(4), np.zeros(2), -0.5, 1.0, 0.5, 0.0)
        adv, ret = buf.compute_gae(0.0, 0.99, 0.95)
        assert adv.shape == (10,)
        assert ret.shape == (10,)

    def test_short_train(self):
        log = self.agent.train(self.env, n_updates=3)
        assert len(log) == 3
        assert "mean_ep_reward" in log[0]


# ============================================================
# 22 — Curriculum RL
# ============================================================
class TestCurriculumRL:
    def setup_method(self):
        from curriculum_rl import CurriculumScheduler, CurriculumStrategy, CurriculumNavEnv
        self.scheduler = CurriculumScheduler(
            strategy=CurriculumStrategy.ADAPTIVE, window_size=10
        )
        self.env = CurriculumNavEnv(self.scheduler, obs_dim=14)

    def test_initial_stage(self):
        assert self.scheduler.current_stage_idx == 0
        assert self.scheduler.current.name == "easy"

    def test_success_rate_updates(self):
        for _ in range(5):
            self.scheduler.record_episode(True)
        assert self.scheduler.success_rate > 0.0

    def test_stage_advances_on_success(self):
        for _ in range(20):
            self.scheduler.record_episode(True)
        assert self.scheduler.current_stage_idx >= 1

    def test_env_reset_obs_shape(self):
        obs = self.env.reset()
        assert obs.shape == (14,)

    def test_env_step(self):
        self.env.reset()
        obs, r, done, info = self.env.step(np.zeros(2))
        assert obs.shape == (14,)
        assert "success" in info


# ============================================================
# 23 — Gaussian Splatting Mapper
# ============================================================
class TestGaussianSplattingMap:
    def setup_method(self):
        from gaussian_splatting_map import GaussianSplattingMap, GSMapConfig
        cfg = GSMapConfig(grid_size=(32, 32), max_gaussians=200)
        self.gsmap = GaussianSplattingMap(cfg)

    def test_add_frame(self):
        pts = np.random.rand(50, 3) * 2 - 1
        n = self.gsmap.add_frame(pts)
        assert n > 0
        assert len(self.gsmap.gaussians) > 0

    def test_occupancy_grid_shape(self):
        pts = np.random.rand(30, 3)
        pts[:, 2] = 0.5   # z in valid range
        self.gsmap.add_frame(pts)
        grid = self.gsmap.to_occupancy_grid()
        assert grid.shape == (32, 32)
        assert grid.min() >= 0.0 and grid.max() <= 1.0

    def test_uncertainty_map(self):
        pts = np.random.rand(20, 3) * 0.5
        self.gsmap.add_frame(pts)
        unc = self.gsmap.uncertainty_map()
        assert unc.shape == (32, 32)

    def test_frontier_detection(self):
        pts = np.random.rand(30, 3) * 0.3
        pts[:, 2] = 0.5
        self.gsmap.add_frame(pts)
        frontiers = self.gsmap.frontier_cells()
        assert isinstance(frontiers, list)

    def test_stats(self):
        pts = np.random.rand(10, 3)
        self.gsmap.add_frame(pts)
        s = self.gsmap.stats()
        assert "n_gaussians" in s and s["n_gaussians"] > 0


# ============================================================
# 24 — NeRF Uncertainty Maps
# ============================================================
class TestNeRFUncertainty:
    def setup_method(self):
        from nerf_uncertainty import TinyNeRF, NeRFConfig, NeRFUncertaintyMapper
        cfg = NeRFConfig(n_samples_coarse=8, n_uncertainty_samples=3,
                         grid_size=(16, 16), hidden_dim=16)
        self.nerf = TinyNeRF(cfg)
        self.mapper = NeRFUncertaintyMapper(self.nerf, cfg)

    def test_positional_encoding(self):
        from nerf_uncertainty import PositionalEncoder
        enc = PositionalEncoder(L=4)
        x = np.random.rand(5, 3)
        out = enc.encode(x)
        assert out.shape[0] == 5
        assert out.shape[1] == enc.out_dim

    def test_nerf_query_shapes(self):
        pts = np.random.rand(8, 3)
        dirs = np.random.rand(8, 3)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        rgb, sigma = self.nerf.query(pts, dirs)
        assert rgb.shape == (8, 3)
        assert sigma.shape == (8,)

    def test_render_ray(self):
        origin = np.array([0.0, 0.0, 1.5])
        direction = np.array([0.0, 0.0, -1.0])
        depth, colour, unc = self.nerf.render_ray(origin, direction)
        assert isinstance(depth, float) and depth >= 0
        assert isinstance(unc, float) and unc >= 0

    def test_uncertainty_map_shape(self):
        poses = [np.eye(4) for _ in range(3)]
        unc = self.mapper.build_uncertainty_map(poses)
        assert unc.shape == (16, 16)
        assert unc.min() >= 0.0 and unc.max() <= 1.0

    def test_exploration_weights_sum_one(self):
        unc = np.random.rand(16, 16)
        w = self.mapper.uncertainty_to_exploration_weights(unc)
        assert abs(w.sum() - 1.0) < 1e-5


# ============================================================
# 25 — Adversarial Attack Simulator
# ============================================================
class TestAdversarialAttacks:
    def setup_method(self):
        from adversarial_attacks import (
            AttackConfig, AttackType, GradientAttacker,
            LiDARAttacker, OdometrySpoofer, RobustnessEvaluator
        )
        self.cfg = AttackConfig(epsilon=0.05, pgd_steps=5)
        self.grad = GradientAttacker(self.cfg)
        self.lidar = LiDARAttacker(self.cfg)
        self.odom = OdometrySpoofer(self.cfg)
        self.evaluator = RobustnessEvaluator(self.cfg)

    def test_fgsm_changes_obs(self):
        obs = np.random.rand(10).astype(float)
        loss_fn = lambda x: float(np.sum(x))
        adv = self.grad.fgsm(obs, loss_fn)
        assert not np.allclose(obs, adv)

    def test_fgsm_respects_epsilon(self):
        obs = np.random.rand(10).astype(float)
        loss_fn = lambda x: float(np.sum(x))
        adv = self.grad.fgsm(obs, loss_fn)
        assert np.max(np.abs(adv - obs)) <= self.cfg.epsilon + 1e-6

    def test_pgd_stronger_than_fgsm(self):
        obs = np.ones(8) * 0.5
        loss_fn = lambda x: -float(np.sum(x))  # minimise sum
        fgsm_adv = self.grad.fgsm(obs, loss_fn)
        pgd_adv = self.grad.pgd(obs, loss_fn)
        assert loss_fn(pgd_adv) <= loss_fn(fgsm_adv) + 0.1

    def test_lidar_spoof_adds_points(self):
        pc = np.random.rand(100, 3)
        spoofed = self.lidar.spoof_add(pc, np.zeros(3))
        assert len(spoofed) > len(pc)

    def test_lidar_remove_reduces_points(self):
        pc = np.random.rand(100, 3)
        removed = self.lidar.spoof_remove(pc)
        assert len(removed) < len(pc)

    def test_odom_drift_accumulates(self):
        self.odom.activate()
        odom = np.array([0.0, 0.0, 0.0])
        for _ in range(50):
            odom = self.odom.corrupt(odom)
        assert self.odom.total_drift_m > 0


# ============================================================
# 26 — Swarm Consensus
# ============================================================
class TestSwarmConsensus:
    def setup_method(self):
        from swarm_consensus import SwarmCoordinator, SwarmRobot, BFTConsensus
        self.coordinator = SwarmCoordinator(n_robots=5, n_byzantine=1)
        self.grid = np.zeros((20, 20))
        self.grid[8:12, 8:12] = 1.0   # obstacle block

    def test_honest_robot_proposal(self):
        from swarm_consensus import SwarmRobot
        robot = SwarmRobot(robot_id=99, faulty=False)
        prop = robot.compute_proposal(self.grid, (0, 0), (18, 18))
        assert prop is not None
        assert len(prop.path) > 0

    def test_byzantine_robot_corrupt(self):
        from swarm_consensus import SwarmRobot
        robot = SwarmRobot(robot_id=0, faulty=True, fault_type="constant_bad")
        prop = robot.compute_proposal(self.grid, (0, 0), (18, 18))
        assert prop.cost > 1000

    def test_consensus_selects_reasonable_path(self):
        result = self.coordinator.plan(self.grid, (0, 0), (18, 18))
        assert result.agreed_cost < 9999
        assert len(result.agreed_path) > 0

    def test_byz_detection(self):
        result = self.coordinator.plan(self.grid, (0, 0), (18, 18))
        assert result.n_byzantine_detected >= 0

    def test_summary(self):
        self.coordinator.plan(self.grid, (0, 0), (18, 18))
        s = self.coordinator.summary()
        assert s["n_robots"] == 5
        assert s["n_faulty"] == 1
