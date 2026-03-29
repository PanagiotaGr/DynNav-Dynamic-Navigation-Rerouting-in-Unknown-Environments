import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '09_multi_robot', 'code'))

import pytest
from self_trust_manager import SelfTrustManager, SelfTrustState


class TestSelfTrustManager:
    def setup_method(self):
        self.manager = SelfTrustManager(
            alpha_calib=0.6,
            alpha_drift=0.4,
            safe_threshold=0.5
        )

    def test_register_robot(self):
        self.manager.register_robot(0)
        state = self.manager.get_state(0)
        assert state is not None
        assert state.robot_id == 0
        assert state.S == pytest.approx(1.0)
        assert state.safe_mode is False

    def test_update_perfect_conditions(self):
        self.manager.register_robot(1)
        S, safe = self.manager.update(1, calibration_error=0.0, expected_drift=0.0, observed_drift=0.0)
        assert S == pytest.approx(1.0)
        assert safe is False

    def test_update_high_calibration_error(self):
        self.manager.register_robot(2)
        S, safe = self.manager.update(2, calibration_error=1.0, expected_drift=0.0, observed_drift=0.0)
        assert S < 0.5
        assert safe is True

    def test_update_high_drift_mismatch(self):
        self.manager.register_robot(3)
        S, safe = self.manager.update(3, calibration_error=0.0, expected_drift=0.0, observed_drift=1.0)
        assert S < 0.7

    def test_trust_clamped_to_zero(self):
        self.manager.register_robot(4)
        S, _ = self.manager.update(4, calibration_error=1.0, expected_drift=0.0, observed_drift=1.0)
        assert S >= 0.0

    def test_trust_clamped_to_one(self):
        self.manager.register_robot(5)
        S, _ = self.manager.update(5, calibration_error=0.0, expected_drift=1.0, observed_drift=0.5)
        assert S <= 1.0

    def test_auto_register_on_update(self):
        S, safe = self.manager.update(99, calibration_error=0.0, expected_drift=0.0, observed_drift=0.0)
        assert S == pytest.approx(1.0)

    def test_safe_mode_triggered_below_threshold(self):
        self.manager.safe_threshold = 0.8
        self.manager.register_robot(6)
        S, safe = self.manager.update(6, calibration_error=0.3, expected_drift=0.0, observed_drift=0.1)
        if S < 0.8:
            assert safe is True

    def test_multiple_robots_independent(self):
        self.manager.register_robot(10)
        self.manager.register_robot(11)
        S10, _ = self.manager.update(10, calibration_error=1.0, expected_drift=0.0, observed_drift=1.0)
        S11, _ = self.manager.update(11, calibration_error=0.0, expected_drift=0.0, observed_drift=0.0)
        assert S10 < S11
