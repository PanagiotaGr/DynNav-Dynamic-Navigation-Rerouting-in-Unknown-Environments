"""
Tests for DynNav new contributions 11-18
=========================================
Run from the repo root:
    pytest contributions/tests/test_new_contributions.py -v
"""
import sys, os
import numpy as np
import pytest

# Add each contribution to path
BASE = os.path.dirname(__file__)
for d in range(11, 19):
    dirs = [
        os.path.join(BASE, f"../{d:02d}_*"),
    ]
    import glob
    for pattern in dirs:
        for path in glob.glob(pattern):
            sys.path.insert(0, path)


# ============================================================
# 11 — VLM Planner
# ============================================================
class TestVLMPlanner:
    def test_encode_frame(self):
        from vlm_planner import VLMNavigationPlanner
        planner = VLMNavigationPlanner()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        encoded = planner._encode_frame(frame)
        assert isinstance(encoded, str) and len(encoded) > 100

    def test_parse_valid_response(self):
        from vlm_planner import VLMNavigationPlanner
        planner = VLMNavigationPlanner()
        raw = '{"region": "corridor", "goal": "go forward", "confidence": 0.8}'
        goal = planner._parse_response(raw)
        assert goal is not None
        assert goal.region_label == "corridor"
        assert goal.confidence == pytest.approx(0.8)

    def test_parse_invalid_response(self):
        from vlm_planner import VLMNavigationPlanner
        planner = VLMNavigationPlanner()
        assert planner._parse_response("not json") is None

    def test_pixel_to_metric(self):
        from vlm_planner import VLMNavigationPlanner
        planner = VLMNavigationPlanner(depth_scale=1.0)
        depth = np.ones((480, 640)) * 2.0
        x, y = planner._pixel_to_metric((320, 240), depth, (0.0, 0.0, 0.0))
        assert abs(x) < 5.0 and abs(y) < 5.0

    def test_session_summary_empty(self):
        from vlm_planner import VLMNavigationPlanner
        planner = VLMNavigationPlanner()
        s = planner.session_summary()
        assert s["total"] == 0


# ============================================================
# 12 — Diffusion Occupancy
# ============================================================
class TestDiffusionOccupancy:
    def setup_method(self):
        from diffusion_occupancy import DiffusionOccupancyPredictor, DiffusionOccupancyConfig
        cfg = DiffusionOccupancyConfig(grid_h=16, grid_w=16, T=5, n_samples=3)
        self.predictor = DiffusionOccupancyPredictor(cfg)

    def test_q_sample_shape(self):
        x0 = np.random.rand(16, 16)
        xt, noise = self.predictor.q_sample(x0, t=2)
        assert xt.shape == (16, 16)
        assert noise.shape == (16, 16)

    def test_sample_range(self):
        cond = np.random.rand(16, 16)
        s = self.predictor.sample(cond)
        assert s.shape == (16, 16)
        assert s.min() >= 0.0 and s.max() <= 1.0

    def test_predict_risk_keys(self):
        occ = (np.random.rand(16, 16) > 0.8).astype(float)
        risk = self.predictor.predict_risk([occ])
        assert "mean" in risk and "std" in risk and "cvar_95" in risk

    def test_risk_weighted_cost(self):
        occ = np.zeros((16, 16))
        risk = self.predictor.predict_risk([occ])
        path = [(i, i) for i in range(5)]
        cost = self.predictor.risk_weighted_cost(path, risk)
        assert cost >= 0


# ============================================================
# 13 — Latent World Model
# ============================================================
class TestLatentWorldModel:
    def setup_method(self):
        from latent_world_model import RSSM, RSSMConfig, WorldModelPlanner
        cfg = RSSMConfig(obs_dim=16, action_dim=2, latent_dim=8, hidden_dim=16, horizon=5)
        self.rssm = RSSM(cfg)
        self.planner = WorldModelPlanner(self.rssm, cfg)

    def test_recurrent_step_shape(self):
        h = np.zeros(16); z = np.zeros(8); a = np.zeros(2)
        h2 = self.rssm.recurrent_step(h, z, a)
        assert h2.shape == (16,)

    def test_prior_shape(self):
        h = np.zeros(16)
        z, mean, lsd = self.rssm.prior(h)
        assert z.shape == mean.shape == lsd.shape == (8,)

    def test_imagine_rollout_length(self):
        h0 = np.zeros(16); z0 = np.zeros(8)
        actions = [np.zeros(2) for _ in range(5)]
        rollout = self.rssm.imagine_rollout(h0, z0, actions)
        assert len(rollout["rewards"]) == 5

    def test_select_best_sequence(self):
        seqs = self.planner.generate_random_sequences(4, horizon=3)
        best, G = self.planner.select_best_action_sequence(seqs)
        assert best is not None
        assert isinstance(G, float)


# ============================================================
# 14 — Causal Risk Attribution
# ============================================================
class TestCausalRisk:
    def setup_method(self):
        from causal_risk import NavigationSCM
        self.scm = NavigationSCM()

    def test_observational_query_keys(self):
        noise = {n: 0.0 for n in self.scm.nodes}
        vals = self.scm.observational_query(noise)
        assert "collision" in vals and "path_risk" in vals

    def test_counterfactual_intervention(self):
        noise = {n: 0.1 for n in self.scm.nodes}
        baseline = self.scm.observational_query(noise)["collision"]
        cf = self.scm.counterfactual_query(noise, {"sensor_noise": 0.0})["collision"]
        # Setting sensor noise to 0 should generally reduce collision
        assert isinstance(cf, float)

    def test_ace_returns_float(self):
        ace = self.scm.average_causal_effect(
            "sensor_noise", "collision", 0.5, 0.0, n_samples=50
        )
        assert isinstance(ace, float)

    def test_root_cause_ranking(self):
        noise = {n: 0.2 for n in self.scm.nodes}
        ranking = self.scm.root_cause_ranking(noise, n_samples=20)
        assert len(ranking) > 0
        assert ranking[0][0] != "collision"


# ============================================================
# 15 — Neuromorphic Sensing
# ============================================================
class TestNeuromorphicSensing:
    def setup_method(self):
        from neuromorphic_sensing import DVSSimulator, DVSSimulatorConfig, SNNObstacleDetector, event_to_time_surface
        cfg = DVSSimulatorConfig(height=60, width=80)
        self.sim = DVSSimulator(cfg)
        self.detector = SNNObstacleDetector(grid_n=3, grid_m=4)
        self.h, self.w = 60, 80

    def test_events_generated(self):
        f1 = np.random.rand(self.h, self.w)
        f2 = np.clip(f1 + 0.3, 0, 1)
        self.sim.process_frame(f1)
        events = self.sim.process_frame(f2)
        assert isinstance(events, list)

    def test_time_surface_shape(self):
        from neuromorphic_sensing import event_to_time_surface, DVSEvent
        events = [DVSEvent(10, 20, 1000.0, 1), DVSEvent(30, 40, 2000.0, -1)]
        surf = event_to_time_surface(events, self.h, self.w)
        assert surf.shape == (self.h, self.w, 2)

    def test_snn_output_shape(self):
        from neuromorphic_sensing import event_to_time_surface
        surf = np.random.rand(self.h, self.w, 2).astype(np.float32)
        prob_map = self.detector.detect(surf, n_steps=3)
        assert prob_map.shape == (3, 4)
        assert prob_map.min() >= 0.0 and prob_map.max() <= 1.0


# ============================================================
# 16 — Federated Learning
# ============================================================
class TestFederatedNav:
    def setup_method(self):
        from federated_nav import FedNavConfig, NavModel, FederatedRobotClient, FederatedServer
        self.cfg = FedNavConfig(n_robots=3, local_epochs=2, global_rounds=2, input_dim=8, output_dim=2)
        self.global_model = NavModel.random_init(8, 2)
        self.clients = [
            FederatedRobotClient(i, self.global_model, self.cfg) for i in range(3)
        ]
        self.server = FederatedServer(self.global_model, self.cfg)

    def test_local_train_returns_params(self):
        client = self.clients[0]
        client.collect_experience(n_steps=10)
        params = client.local_train()
        assert "W" in params and "b" in params

    def test_aggregate_reduces_loss(self):
        history = self.server.run_training(self.clients)
        assert len(history) == 2
        assert all("val_mse" in r for r in history)

    def test_dp_noise(self):
        from federated_nav import FedNavConfig, NavModel, FederatedRobotClient
        cfg = FedNavConfig(dp_epsilon=1.0, local_epochs=1, input_dim=8, output_dim=2)
        client = FederatedRobotClient(0, NavModel.random_init(8, 2), cfg)
        client.collect_experience(5)
        params = client.local_train()
        assert params["W"].shape == (2, 8)


# ============================================================
# 17 — Topological Semantic Maps
# ============================================================
class TestTopoSemanticMap:
    def setup_method(self):
        from topo_semantic_map import TopologicalSemanticMap
        self.m = TopologicalSemanticMap()
        self.n0 = self.m.add_node("kitchen", (0.0, 0.0))
        self.n1 = self.m.add_node("corridor", (2.0, 0.0))
        self.n2 = self.m.add_node("living_room", (4.0, 0.0))
        self.m.add_edge(self.n0, self.n1, weight=2.0)
        self.m.add_edge(self.n1, self.n2, weight=2.0)

    def test_plan_finds_path(self):
        path, cost = self.m.plan(self.n0, self.n2)
        assert path == [self.n0, self.n1, self.n2]
        assert cost == pytest.approx(4.0)

    def test_invalidate_edge_reroutes(self):
        self.m.invalidate_edge(self.n1, self.n2)
        path, cost = self.m.plan(self.n0, self.n2)
        assert cost == np.inf  # no alternative path

    def test_ground_query(self):
        emb = self.m.embed_label_stub("kitchen")
        results = self.m.ground_query(emb, top_k=2)
        assert len(results) == 2
        # kitchen should be top-1
        assert results[0][1] == "kitchen"

    def test_serialise_round_trip(self):
        from topo_semantic_map import TopologicalSemanticMap
        d = self.m.to_dict()
        m2 = TopologicalSemanticMap.from_dict(d)
        assert len(m2.nodes) == len(self.m.nodes)


# ============================================================
# 18 — Formal Safety Shields
# ============================================================
class TestFormalSafetyShields:
    def setup_method(self):
        from formal_safety_shields import (
            STLAtom, STLAlways, STLMonitor,
            CBFSafetyFilter, CBFConfig, SafetyShield
        )
        self.atom = STLAtom(lambda s: s[0] - 0.1, "x_positive")
        self.always = STLAlways(self.atom, 0, 2)
        self.monitor = STLMonitor([("x_pos", self.always)])
        self.cbf = CBFSafetyFilter(CBFConfig(safety_radius=0.3))
        self.shield = SafetyShield(self.monitor, self.cbf)

    def test_stl_robustness_positive(self):
        signal = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        rho = self.always.robustness(signal, 0)
        assert rho > 0

    def test_cbf_reduces_collision_risk(self):
        pos = np.array([0.3, 0.0])
        obs = [np.array([0.5, 0.0])]  # very close obstacle
        u_des = np.array([0.5, 0.0])  # heading straight into it
        u_safe = self.cbf.filter(u_des, pos, obs)
        # Safe command should be different from desired
        assert not np.allclose(u_safe, u_des)

    def test_shield_step_returns_info(self):
        pos = np.array([1.0, 1.0])
        obs = [np.array([3.0, 3.0])]
        u_des = np.array([0.1, 0.1])
        u_safe, info = self.shield.step(u_des, pos, obs, pos)
        assert "stl" in info and "cbf" in info

    def test_stl_monitor_violation_log(self):
        from formal_safety_shields import STLMonitor, STLAtom, STLAlways
        atom = STLAtom(lambda s: s[0] - 1.0, "threshold")
        mon = STLMonitor([("thresh", STLAlways(atom, 0, 0))], robustness_margin=0.0)
        mon.update(np.array([-1.0]))  # violation
        assert len(mon.violation_log) > 0
