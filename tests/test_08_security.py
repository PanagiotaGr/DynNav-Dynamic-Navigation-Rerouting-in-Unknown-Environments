import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '08_security_ids', 'code'))

import pytest
import numpy as np
from security_monitor import mahalanobis_squared, _chi2_isf_wilson_hilferty


class TestMahalanobisSquared:
    def test_zero_vector(self):
        nu = np.zeros(3)
        S = np.eye(3)
        assert mahalanobis_squared(nu, S) == pytest.approx(0.0)

    def test_identity_covariance(self):
        nu = np.array([1.0, 0.0, 0.0])
        S = np.eye(3)
        assert mahalanobis_squared(nu, S) == pytest.approx(1.0)

    def test_known_result(self):
        nu = np.array([3.0, 4.0])
        S = np.eye(2)
        assert mahalanobis_squared(nu, S) == pytest.approx(25.0)

    def test_scaled_covariance(self):
        nu = np.array([2.0])
        S = np.array([[4.0]])
        assert mahalanobis_squared(nu, S) == pytest.approx(1.0)

    def test_wrong_shape_raises(self):
        nu = np.array([1.0, 2.0])
        S = np.eye(3)
        with pytest.raises(ValueError):
            mahalanobis_squared(nu, S)


class TestChi2ISF:
    def test_returns_positive(self):
        result = _chi2_isf_wilson_hilferty(alpha=0.05, dof=3)
        assert result > 0.0

    def test_higher_dof_higher_threshold(self):
        t1 = _chi2_isf_wilson_hilferty(0.05, dof=1)
        t5 = _chi2_isf_wilson_hilferty(0.05, dof=5)
        assert t5 > t1

    def test_stricter_alpha_higher_threshold(self):
        t_strict = _chi2_isf_wilson_hilferty(0.001, dof=3)
        t_loose = _chi2_isf_wilson_hilferty(0.10, dof=3)
        assert t_strict > t_loose

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            _chi2_isf_wilson_hilferty(alpha=0.0, dof=3)
        with pytest.raises(ValueError):
            _chi2_isf_wilson_hilferty(alpha=1.0, dof=3)

    def test_invalid_dof_raises(self):
        with pytest.raises(ValueError):
            _chi2_isf_wilson_hilferty(alpha=0.05, dof=0)
