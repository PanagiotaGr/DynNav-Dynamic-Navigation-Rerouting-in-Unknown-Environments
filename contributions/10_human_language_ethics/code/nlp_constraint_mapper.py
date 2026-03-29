"""
nlp_constraint_mapper.py

Maps natural language instructions to planning cost modifiers.

Examples
--------
    "avoid narrow spaces"     → increase obstacle-proximity penalty
    "prefer safe paths"       → increase risk weight λ
    "move quickly"            → decrease path-length weight (accept more risk)
    "stay on wide corridors"  → penalise cells with low free-neighbor count
    "avoid crowded areas"     → increase social/pedestrian density penalty

Architecture
------------
1. TextClassifier: rule-based keyword mapping → cost modifier dict
2. CostModifier: applies modifiers to a base cost function
3. DemonstrationRunner: shows path changes before/after modifier

The design is intentionally rule-based (no LLM required) so it runs
entirely offline and is reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Cost modifier representation
# ---------------------------------------------------------------------------

@dataclass
class CostModifiers:
    """
    Multiplicative and additive modifiers for the planner cost function.

    obstacle_proximity_mult : multiply proximity penalty by this factor
    risk_weight_mult        : multiply λ_risk by this factor
    path_length_mult        : multiply step cost by this factor
    narrow_space_penalty    : added cost when free_neighbors < threshold
    narrow_threshold        : threshold for narrow space detection (0..4)
    speed_factor            : reciprocal step cost (> 1 → prefer shorter paths)
    description             : human-readable summary of active modifiers
    """
    obstacle_proximity_mult: float = 1.0
    risk_weight_mult: float = 1.0
    path_length_mult: float = 1.0
    narrow_space_penalty: float = 0.0
    narrow_threshold: int = 2
    speed_factor: float = 1.0
    description: list[str] = field(default_factory=list)

    def apply_step_cost(self, base_cost: float, free_neighbors: int) -> float:
        cost = base_cost * self.path_length_mult / self.speed_factor
        if free_neighbors < self.narrow_threshold:
            cost += self.narrow_space_penalty
        return cost

    def apply_risk_weight(self, base_lambda: float) -> float:
        return base_lambda * self.risk_weight_mult

    def apply_proximity_penalty(self, base_penalty: float) -> float:
        return base_penalty * self.obstacle_proximity_mult


# ---------------------------------------------------------------------------
# Rule-based text classifier
# ---------------------------------------------------------------------------

class NLPConstraintMapper:
    """
    Maps a natural language instruction to a CostModifiers instance.

    Matching is case-insensitive and checks for keyword presence.
    Multiple instructions can be chained; modifiers compose multiplicatively.

    Parameters
    ----------
    verbose : print matched rules to stdout
    """

    RULES: list[tuple[list[str], dict]] = [
        # Patterns                          → modifier deltas
        (["avoid narrow", "narrow space", "wide corridor", "corridor"],
         {"obstacle_proximity_mult": 1.5, "narrow_space_penalty": 2.0, "narrow_threshold": 2}),

        (["prefer safe", "safe path", "cautious", "be careful"],
         {"risk_weight_mult": 2.0, "obstacle_proximity_mult": 1.3}),

        (["move quickly", "fast", "shortest", "minimum distance", "hurry"],
         {"path_length_mult": 0.8, "risk_weight_mult": 0.5, "speed_factor": 1.5}),

        (["avoid crowd", "avoid people", "social distance", "less crowded"],
         {"obstacle_proximity_mult": 1.2, "narrow_space_penalty": 1.0}),

        (["avoid dark", "prefer lit", "well-lit"],
         {"obstacle_proximity_mult": 1.3, "risk_weight_mult": 1.5}),

        (["energy", "battery", "save power", "efficient"],
         {"path_length_mult": 1.2, "risk_weight_mult": 0.8}),

        (["risky", "adventurous", "explore"],
         {"risk_weight_mult": 0.3, "obstacle_proximity_mult": 0.5}),
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def parse(self, instruction: str) -> CostModifiers:
        """
        Parse a natural language instruction and return CostModifiers.

        Parameters
        ----------
        instruction : str
            Natural language instruction (e.g., "avoid narrow spaces").

        Returns
        -------
        CostModifiers with all matched rules applied.
        """
        text = instruction.lower()
        mods = CostModifiers()

        for patterns, deltas in self.RULES:
            if any(p in text for p in patterns):
                matched = [p for p in patterns if p in text][0]
                if self.verbose:
                    print(f"[NLP] Matched: '{matched}' → {deltas}")
                mods.description.append(f"'{matched}': {deltas}")

                for key, val in deltas.items():
                    if key == "narrow_threshold":
                        mods.narrow_threshold = val
                    elif key == "narrow_space_penalty":
                        mods.narrow_space_penalty += val
                    else:
                        current = getattr(mods, key)
                        setattr(mods, key, current * val if "mult" in key else val)

        if not mods.description:
            mods.description.append("No rules matched — using default cost.")

        return mods

    def parse_multi(self, instructions: list[str]) -> CostModifiers:
        """Parse multiple instructions and combine their modifiers."""
        combined = CostModifiers()
        for instr in instructions:
            m = self.parse(instr)
            combined.obstacle_proximity_mult *= m.obstacle_proximity_mult
            combined.risk_weight_mult *= m.risk_weight_mult
            combined.path_length_mult *= m.path_length_mult
            combined.narrow_space_penalty += m.narrow_space_penalty
            combined.narrow_threshold = min(combined.narrow_threshold, m.narrow_threshold)
            combined.speed_factor *= m.speed_factor
            combined.description.extend(m.description)
        return combined
