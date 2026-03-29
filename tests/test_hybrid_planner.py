import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', 'hybrid_planner', 'code'))

import pytest
import numpy as np
from confidence_gate import ConfidenceGate, GateConfig, HeuristicChoice


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------

class TestGateConfig:
    def test_defaults(self):
        cfg = GateConfig()
        assert cfg.std_threshold == 2.0
        assert cfg.epsilon_budget == 0.5
        assert cfg.k == 1.0

    def test_custom(self):
        cfg = GateConfig(std_threshold=1.0, epsilon_budget=0.2, k=2.0)
        assert cfg.std_threshold == 1.0
        assert cfg.epsilon_budget == 0.2
        assert cfg.k == 2.0


# ---------------------------------------------------------------------------
# ConfidenceGate.select
# ---------------------------------------------------------------------------

class TestConfidenceGateSelect:
    def test_uses_learned_when_all_ok(self):
        gate = ConfidenceGate(GateConfig(std_threshold=2.0, epsilon_budget=1.0, k=1.0))
        val, choice = gate.select(h_mean=5.0, h_std=0.5, h_admissible=5.0)
        assert choice == HeuristicChoice.LEARNED
        assert val == pytest.approx(5.0)

    def test_falls_back_on_high_std(self):
        gate = ConfidenceGate(GateConfig(std_threshold=1.0, epsilon_budget=1.0, k=1.0))
        val, choice = gate.select(h_mean=5.0, h_std=1.5, h_admissible=5.0)
        assert choice == HeuristicChoice.ADMISSIBLE
        assert val == pytest.approx(5.0)

    def test_falls_back_on_negative_lower_bound(self):
        gate = ConfidenceGate(GateConfig(std_threshold=5.0, epsilon_budget=1.0, k=3.0))
        # lower_bound = 1.0 - 3.0 * 1.0 = -2.0 < 0
        val, choice = gate.select(h_mean=1.0, h_std=1.0, h_admissible=5.0)
        assert choice == HeuristicChoice.ADMISSIBLE

    def test_falls_back_on_epsilon_violation(self):
        gate = ConfidenceGate(GateConfig(std_threshold=5.0, epsilon_budget=0.1, k=0.0))
        # h_mean=8.0 > h_admissible=5.0 + 0.1 → fallback
        val, choice = gate.select(h_mean=8.0, h_std=0.1, h_admissible=5.0)
        assert choice == HeuristicChoice.ADMISSIBLE
        assert val == pytest.approx(5.0)

    def test_exact_epsilon_boundary(self):
        gate = ConfidenceGate(GateConfig(std_threshold=5.0, epsilon_budget=0.5, k=0.0))
        # h_mean = admissible + epsilon exactly → allowed
        val, choice = gate.select(h_mean=5.5, h_std=0.1, h_admissible=5.0)
        assert choice == HeuristicChoice.LEARNED

    def test_zero_std_prefers_learned(self):
        gate = ConfidenceGate()
        val, choice = gate.select(h_mean=4.0, h_std=0.0, h_admissible=4.0)
        assert choice == HeuristicChoice.LEARNED


# ---------------------------------------------------------------------------
# ConfidenceGate statistics
# ---------------------------------------------------------------------------

class TestConfidenceGateStats:
    def test_learned_fraction_zero_initially(self):
        gate = ConfidenceGate()
        assert gate.learned_fraction == pytest.approx(0.0)

    def test_learned_fraction_all_learned(self):
        gate = ConfidenceGate(GateConfig(std_threshold=10.0, epsilon_budget=10.0, k=0.0))
        for _ in range(5):
            gate.select(h_mean=3.0, h_std=0.0, h_admissible=3.0)
        assert gate.learned_fraction == pytest.approx(1.0)

    def test_learned_fraction_all_admissible(self):
        gate = ConfidenceGate(GateConfig(std_threshold=0.0))
        for _ in range(5):
            gate.select(h_mean=5.0, h_std=0.1, h_admissible=5.0)
        assert gate.learned_fraction == pytest.approx(0.0)

    def test_reset_clears_stats(self):
        gate = ConfidenceGate(GateConfig(std_threshold=10.0, epsilon_budget=10.0, k=0.0))
        for _ in range(4):
            gate.select(h_mean=3.0, h_std=0.0, h_admissible=3.0)
        gate.reset_stats()
        assert gate.learned_fraction == pytest.approx(0.0)

    def test_mixed_fraction(self):
        gate = ConfidenceGate(GateConfig(std_threshold=1.0, epsilon_budget=0.5, k=1.0))
        # force learned: std=0.0, mean within budget
        gate.select(h_mean=3.0, h_std=0.0, h_admissible=3.0)
        gate.select(h_mean=3.0, h_std=0.0, h_admissible=3.0)
        # force admissible: std too high
        gate.select(h_mean=3.0, h_std=2.0, h_admissible=3.0)
        assert gate.learned_fraction == pytest.approx(2 / 3)
