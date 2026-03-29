import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '10_human_language_ethics', 'code'))

import pytest
from nlp_constraint_mapper import NLPConstraintMapper, CostModifiers


# ---------------------------------------------------------------------------
# CostModifiers
# ---------------------------------------------------------------------------

class TestCostModifiers:
    def test_defaults(self):
        m = CostModifiers()
        assert m.obstacle_proximity_mult == pytest.approx(1.0)
        assert m.risk_weight_mult == pytest.approx(1.0)
        assert m.path_length_mult == pytest.approx(1.0)
        assert m.narrow_space_penalty == pytest.approx(0.0)
        assert m.speed_factor == pytest.approx(1.0)

    def test_apply_step_cost_no_penalty(self):
        m = CostModifiers()
        cost = m.apply_step_cost(base_cost=1.0, free_neighbors=4)
        assert cost == pytest.approx(1.0)

    def test_apply_step_cost_narrow_penalty(self):
        m = CostModifiers(narrow_space_penalty=2.0, narrow_threshold=3)
        cost = m.apply_step_cost(base_cost=1.0, free_neighbors=1)
        assert cost == pytest.approx(3.0)

    def test_apply_risk_weight(self):
        m = CostModifiers(risk_weight_mult=2.0)
        assert m.apply_risk_weight(1.0) == pytest.approx(2.0)

    def test_apply_proximity_penalty(self):
        m = CostModifiers(obstacle_proximity_mult=1.5)
        assert m.apply_proximity_penalty(2.0) == pytest.approx(3.0)

    def test_speed_factor_reduces_step_cost(self):
        m = CostModifiers(speed_factor=2.0)
        cost = m.apply_step_cost(1.0, free_neighbors=4)
        assert cost == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# NLPConstraintMapper.parse
# ---------------------------------------------------------------------------

class TestNLPConstraintMapperParse:
    def setup_method(self):
        self.mapper = NLPConstraintMapper(verbose=False)

    def test_no_match_returns_defaults(self):
        m = self.mapper.parse("go to the kitchen")
        assert m.obstacle_proximity_mult == pytest.approx(1.0)
        assert m.risk_weight_mult == pytest.approx(1.0)

    def test_no_match_description(self):
        m = self.mapper.parse("something random")
        assert any("No rules matched" in d for d in m.description)

    def test_avoid_narrow_increases_proximity(self):
        m = self.mapper.parse("avoid narrow spaces")
        assert m.obstacle_proximity_mult > 1.0

    def test_prefer_safe_increases_risk_weight(self):
        m = self.mapper.parse("prefer safe paths")
        assert m.risk_weight_mult > 1.0

    def test_move_quickly_reduces_risk_weight(self):
        m = self.mapper.parse("move quickly to the goal")
        assert m.risk_weight_mult < 1.0

    def test_case_insensitive(self):
        m1 = self.mapper.parse("MOVE QUICKLY")
        m2 = self.mapper.parse("move quickly")
        assert m1.risk_weight_mult == pytest.approx(m2.risk_weight_mult)

    def test_match_description_non_empty(self):
        m = self.mapper.parse("cautious navigation")
        assert len(m.description) >= 1
        assert "No rules matched" not in m.description[0]

    def test_energy_increases_path_length_mult(self):
        m = self.mapper.parse("save battery energy")
        assert m.path_length_mult > 1.0

    def test_risky_reduces_proximity(self):
        m = self.mapper.parse("I want to be adventurous")
        assert m.obstacle_proximity_mult < 1.0


# ---------------------------------------------------------------------------
# NLPConstraintMapper.parse_multi
# ---------------------------------------------------------------------------

class TestNLPConstraintMapperParseMulti:
    def setup_method(self):
        self.mapper = NLPConstraintMapper()

    def test_combined_two_instructions(self):
        m = self.mapper.parse_multi(["prefer safe paths", "move quickly"])
        # safe increases risk_weight (×2), quickly decreases (×0.5) → net ~1.0
        assert m.risk_weight_mult > 0.0

    def test_empty_list_returns_defaults(self):
        m = self.mapper.parse_multi([])
        assert m.obstacle_proximity_mult == pytest.approx(1.0)
        assert m.risk_weight_mult == pytest.approx(1.0)

    def test_duplicate_instructions_amplify(self):
        m1 = self.mapper.parse("prefer safe paths")
        m2 = self.mapper.parse_multi(["prefer safe paths", "prefer safe paths"])
        # Should be more than single application
        assert m2.risk_weight_mult >= m1.risk_weight_mult

    def test_combined_description_merged(self):
        m = self.mapper.parse_multi(["prefer safe paths", "move quickly"])
        assert len(m.description) >= 2
