import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '08_security_ids', 'code'))

import pytest
import numpy as np
from cusum_detector import (
    CUSUMConfig, CUSUMDetector, CUSUMState,
    simulate_innovations, evaluate_cusum,
)


# ---------------------------------------------------------------------------
# CUSUMDetector basic update
# ---------------------------------------------------------------------------

class TestCUSUMDetectorUpdate:
    def test_no_alarm_on_zero_innovations(self):
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=5.0))
        for _ in range(50):
            alarm = det.update(0.0)
        assert not alarm
        assert not det.state.alarm

    def test_alarm_on_large_positive_drift(self):
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=5.0))
        alarms = [det.update(2.0) for _ in range(20)]
        assert any(alarms)

    def test_alarm_on_large_negative_drift(self):
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=5.0))
        alarms = [det.update(-2.0) for _ in range(20)]
        assert any(alarms)

    def test_s_pos_never_negative(self):
        det = CUSUMDetector()
        for val in [-1.0, -2.0, 0.0, 0.1]:
            det.update(val)
        assert det.state.S_pos >= 0.0

    def test_s_neg_never_negative(self):
        det = CUSUMDetector()
        for val in [1.0, 2.0, 0.0, -0.1]:
            det.update(val)
        assert det.state.S_neg >= 0.0

    def test_step_counter_increments(self):
        det = CUSUMDetector()
        for _ in range(10):
            det.update(0.0)
        assert det.state.step == 10

    def test_n_alarms_counts_transitions(self):
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=2.0))
        # flood with big values to trigger alarm
        for _ in range(10):
            det.update(2.0)
        n1 = det.state.n_alarms
        # reset and trigger again
        det.reset()
        for _ in range(10):
            det.update(2.0)
        assert det.state.n_alarms >= 1


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestCUSUMReset:
    def test_reset_clears_statistics(self):
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=2.0))
        for _ in range(20):
            det.update(2.0)
        assert det.state.alarm
        det.reset()
        assert not det.state.alarm
        assert det.state.S_pos == 0.0
        assert det.state.S_neg == 0.0

    def test_reset_preserves_step_count(self):
        det = CUSUMDetector()
        for _ in range(5):
            det.update(0.0)
        det.reset()
        assert det.state.step == 5


# ---------------------------------------------------------------------------
# risk_multiplier and safe_mode
# ---------------------------------------------------------------------------

class TestCUSUMProperties:
    def test_risk_multiplier_one_when_no_alarm(self):
        det = CUSUMDetector()
        assert det.risk_multiplier == pytest.approx(1.0)

    def test_risk_multiplier_above_one_during_alarm(self):
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=3.0))
        for _ in range(30):
            det.update(1.5)
        if det.state.alarm:
            assert det.risk_multiplier > 1.0

    def test_safe_mode_false_normally(self):
        det = CUSUMDetector()
        det.update(0.0)
        assert not det.should_trigger_safe_mode

    def test_safe_mode_triggers_on_extreme_drift(self):
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=3.0))
        for _ in range(50):
            det.update(3.0)
        assert det.should_trigger_safe_mode

    def test_summary_keys(self):
        det = CUSUMDetector()
        s = det.summary()
        for key in ("step", "S_pos", "S_neg", "alarm", "n_alarms", "risk_mult", "safe_mode"):
            assert key in s


# ---------------------------------------------------------------------------
# simulate_innovations
# ---------------------------------------------------------------------------

class TestSimulateInnovations:
    def test_length(self):
        z = simulate_innovations(n_steps=100)
        assert len(z) == 100

    def test_attack_increases_mean(self):
        z = simulate_innovations(n_steps=200, attack_start=100, attack_magnitude=3.0, seed=0)
        assert float(np.mean(z[100:])) > float(np.mean(z[:100]))

    def test_normal_section_near_zero_mean(self):
        z = simulate_innovations(n_steps=1000, attack_start=900, seed=42)
        assert abs(float(np.mean(z[:900]))) < 0.5


# ---------------------------------------------------------------------------
# evaluate_cusum
# ---------------------------------------------------------------------------

class TestEvaluateCusum:
    def test_detects_attack(self):
        z = simulate_innovations(n_steps=200, attack_start=100, attack_magnitude=2.0, seed=0)
        result = evaluate_cusum(z, attack_start=100)
        # Should detect the attack (delay might be a few steps)
        assert result["detection_step"] is not None
        assert not result["missed_detection"]

    def test_detection_after_attack_start(self):
        z = simulate_innovations(n_steps=200, attack_start=100, attack_magnitude=2.0, seed=0)
        result = evaluate_cusum(z, attack_start=100)
        if result["detection_step"] is not None:
            assert result["detection_step"] >= 100 or result["false_alarms"] > 0

    def test_result_keys(self):
        z = np.zeros(50)
        result = evaluate_cusum(z)
        for k in ("detection_step", "false_alarms", "missed_detection", "total_alarms"):
            assert k in result
