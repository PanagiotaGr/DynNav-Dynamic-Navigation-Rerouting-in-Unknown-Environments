"""
human_preference_wrapper.py

Wrapper γύρω από έναν υπάρχοντα risk-aware planner, ώστε:
- να παίρνει ανθρώπινη προτίμηση ως string
- να προσαρμόζει το lambda (risk weight)
- να υπολογίζει cost με βάση HumanRiskPolicy
"""

from typing import Any

from user_preferences import parse_human_preference, HumanPreference
from human_risk_policy import HumanRiskPolicy, HumanRiskConfig


class HumanPreferencePlannerWrapper:
    """
    Τυλίγει έναν υπάρχοντα planner που χρησιμοποιεί risk weight (lambda).

    Υποθέτουμε ότι ο underlying planner έχει μέθοδο:
        plan(state, observation, lambda_weight=..., **kwargs)

    Αν το δικό σου API είναι λίγο διαφορετικό, απλά άλλαξε το call
    στο _call_underlying_planner.
    """

    def __init__(
        self,
        underlying_planner: Any,
        human_pref_text: str,
        human_influence_scale: float = 1.0,
    ):
        self.planner = underlying_planner
        self.human_pref_text = human_pref_text
        self.human_pref: HumanPreference = parse_human_preference(human_pref_text)
        cfg = HumanRiskConfig(human_influence_scale=human_influence_scale)
        self.human_risk_policy = HumanRiskPolicy(config=cfg)

    def set_human_preference(self, text: str):
        """Αλλαγή προτίμησης on-the-fly (π.χ. από UI)."""
        self.human_pref_text = text
        self.human_pref = parse_human_preference(text)

    def plan(
        self,
        state,
        observation,
        lambda_robot: float,
        **kwargs,
    ):
        """
        Συνδυάζει:
        - lambda_robot (από self-trust / OOD / drift / κτλ.)
        - human_pref.risk_preference

        και καλεί τον underlying planner με lambda_effective.
        """
        lambda_eff = self.human_risk_policy.compute_lambda_effective(
            lambda_robot=lambda_robot,
            human_pref=self.human_pref,
        )

        # Εδώ προσαρμόζεις στο API του planner σου:
        # Αν ο planner δεν λέγεται plan ή δεν παίρνει lambda_weight,
        # άλλαξε το ανάλογα.
        return self._call_underlying_planner(
            state=state,
            observation=observation,
            lambda_effective=lambda_eff,
            **kwargs,
        )

    def _call_underlying_planner(
        self,
        state,
        observation,
        lambda_effective: float,
        **kwargs,
    ):
        """
        Προσαρμογή στο API του δικού σου planner.

        Παράδειγμα:
            self.planner.plan(state, observation, lambda_weight=lambda_effective, **kwargs)
        """
        if hasattr(self.planner, "plan"):
            # Συνήθες pattern
            return self.planner.plan(
                state=state,
                observation=observation,
                lambda_weight=lambda_effective,
                **kwargs,
            )
        else:
            # Αν ο planner έχει άλλη υπογραφή, προσαρμόζεις εδώ.
            raise NotImplementedError(
                "Προσαρμοσε το _call_underlying_planner στο API του planner σου."
            )
