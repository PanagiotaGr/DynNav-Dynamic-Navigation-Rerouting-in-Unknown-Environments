import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '06_energy_connectivity', 'code'))

import pytest
import numpy as np
from energy_model import EnergyModel, EnergyParams
from connectivity_map import ConnectivityMap


class TestEnergyModel:
    def test_step_cost_no_risk(self):
        params = EnergyParams(move_cost=1.0, risk_energy_coeff=0.0)
        model = EnergyModel(params)
        assert model.step_energy_cost(risk=0.5) == pytest.approx(1.0)

    def test_step_cost_with_risk(self):
        params = EnergyParams(move_cost=1.0, risk_energy_coeff=2.0)
        model = EnergyModel(params)
        assert model.step_energy_cost(risk=1.0) == pytest.approx(3.0)

    def test_step_cost_zero_move(self):
        params = EnergyParams(move_cost=0.0, risk_energy_coeff=1.0)
        model = EnergyModel(params)
        assert model.step_energy_cost(risk=0.5) == pytest.approx(0.5)

    def test_default_params(self):
        params = EnergyParams()
        model = EnergyModel(params)
        cost = model.step_energy_cost(risk=0.0)
        assert cost == pytest.approx(1.0)

    def test_initial_energy_budget(self):
        params = EnergyParams(e0=100.0)
        assert params.e0 == pytest.approx(100.0)


class TestConnectivityMap:
    def setup_method(self):
        self.shape = (20, 20)
        self.cmap = ConnectivityMap(
            grid_shape=self.shape,
            ap_xy=(10.0, 10.0),
            rng_seed=0,
        )
        self.occupancy = np.zeros(self.shape)

    def test_build_returns_correct_shapes(self):
        snr, C, P_loss = self.cmap.build(self.occupancy, add_shadowing=False)
        assert snr.shape == self.shape
        assert C.shape == self.shape
        assert P_loss.shape == self.shape

    def test_connectivity_in_range(self):
        _, C, _ = self.cmap.build(self.occupancy, add_shadowing=False)
        assert float(C.min()) >= 0.0
        assert float(C.max()) <= 1.0

    def test_packet_loss_in_range(self):
        _, _, P_loss = self.cmap.build(self.occupancy, add_shadowing=False)
        assert float(P_loss.min()) >= 0.0
        assert float(P_loss.max()) <= 1.0

    def test_obstacles_reduce_connectivity(self):
        occ_clear = np.zeros(self.shape)
        occ_blocked = np.ones(self.shape)
        _, C_clear, _ = self.cmap.build(occ_clear, add_shadowing=False)
        _, C_blocked, _ = self.cmap.build(occ_blocked, add_shadowing=False)
        assert float(C_clear.mean()) > float(C_blocked.mean())

    def test_closer_to_ap_has_better_connectivity(self):
        _, C, _ = self.cmap.build(self.occupancy, add_shadowing=False)
        ap_row, ap_col = 10, 10
        near = float(C[ap_row, ap_col])
        far = float(C[0, 0])
        assert near >= far
