# energy_model.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class EnergyParams:
    e0: float = 100.0              # initial energy budget
    move_cost: float = 1.0         # baseline energy per move
    turn_cost: float = 0.0         # optional (unused in 4-neigh grid)
    risk_energy_coeff: float = 0.0 # optional coupling: risky areas consume more energy
    compute_cost_per_expand: float = 0.0  # optional "compute load" modeling


class EnergyModel:
    def __init__(self, params: EnergyParams):
        self.p = params

    def step_energy_cost(self, risk: float) -> float:
        """
        Energy consumed for one action step.
        risk: scalar risk (0..1 typically) for the cell/transition
        """
        return self.p.move_cost + self.p.risk_energy_coeff * float(risk)
