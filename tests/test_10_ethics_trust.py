import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '10_human_language_ethics', 'code'))

import pytest
from ethical_layer import EthicalRiskPolicy, EthicalDecision
from trust_layer import TrustManager, TrustState


class TestEthicalRiskPolicy:
    def setup_method(self):
        self.policy = EthicalRiskPolicy(base_scale=1.0, max_scale=2.0)

    def test_no_factors_returns_base_scale(self):
        decision = self.policy.evaluate_from_language({"factors": []})
        assert decision.ethical_risk_scale == pytest.approx(1.0)

    def test_children_factor_increases_scale(self):
        decision = self.policy.evaluate_from_language({"factors": ["children nearby"]})
        assert decision.ethical_risk_scale > 1.0

    def test_elderly_factor_increases_scale(self):
        decision = self.policy.evaluate_from_language({"factors": ["elderly pedestrian"]})
        assert decision.ethical_risk_scale > 1.0

    def test_crowding_factor_increases_scale(self):
        decision = self.policy.evaluate_from_language({"factors": ["crowding detected"]})
        assert decision.ethical_risk_scale > 1.0

    def test_multiple_factors_capped_at_max(self):
        decision = self.policy.evaluate_from_language({
            "factors": ["children", "elderly", "crowding", "many people"]
        })
        assert decision.ethical_risk_scale <= self.policy.max_scale + 1e-9

    def test_returns_ethical_decision_type(self):
        decision = self.policy.evaluate_from_language({"factors": []})
        assert isinstance(decision, EthicalDecision)

    def test_factors_preserved_in_decision(self):
        factors = ["children in area"]
        decision = self.policy.evaluate_from_language({"factors": factors})
        assert decision.ethical_factors == factors


class TestTrustManager:
    def setup_method(self):
        self.manager = TrustManager(alpha_down=0.15, alpha_up=0.05)

    def test_initial_trust_value(self):
        trust = TrustState()
        assert 0.0 <= trust.value <= 1.0

    def test_trust_decreases_on_failure(self):
        trust = TrustState(value=0.8)
        self_heal = {"reasons": ["drift_detected"], "safe_mode": False}
        updated = self.manager.update(trust, self_heal, None)
        assert updated.value < 0.8

    def test_trust_decreases_in_safe_mode(self):
        trust = TrustState(value=0.8)
        self_heal = {"reasons": [], "safe_mode": True}
        updated = self.manager.update(trust, self_heal, None)
        assert updated.value < 0.8

    def test_trust_increases_on_good_language(self):
        trust = TrustState(value=0.5)
        self_heal = {"reasons": [], "safe_mode": False}
        lang = {"factors": ["obstacle ahead"], "risk_scale": 1.5}
        updated = self.manager.update(trust, self_heal, lang)
        assert updated.value > 0.5

    def test_trust_stays_in_bounds(self):
        trust = TrustState(value=0.0)
        self_heal = {"reasons": ["failure", "drift"], "safe_mode": True}
        updated = self.manager.update(trust, self_heal, None)
        assert updated.value >= 0.0
        assert updated.value <= 1.0

    def test_trust_weighted_risk_low_trust(self):
        trust = TrustState(value=0.0)
        weighted = self.manager.compute_trust_weighted_risk(1.0, trust)
        assert weighted > 1.0

    def test_trust_weighted_risk_full_trust(self):
        trust = TrustState(value=1.0)
        weighted = self.manager.compute_trust_weighted_risk(1.0, trust)
        assert weighted == pytest.approx(1.0)

    def test_trust_weighted_risk_proportional(self):
        trust_low = TrustState(value=0.2)
        trust_high = TrustState(value=0.9)
        r_low = self.manager.compute_trust_weighted_risk(1.0, trust_low)
        r_high = self.manager.compute_trust_weighted_risk(1.0, trust_high)
        assert r_low > r_high
