"""
human_aware_real_planner.py

Generic wrapper για οποιονδήποτε risk-aware planner στο repo.

Λογική:
- Παίρνει έναν "κανονικό" planner (π.χ. PRMPlanner).
- Παίρνει ανθρώπινη προτίμηση σε φυσική γλώσσα.
- Συνδυάζει:
      lambda_robot (από το σύστημα)
      + HumanPreference (risk_preference)
  για να φτιάξει:
      lambda_effective

- Προσπαθεί:
  1) Να γράψει lambda_effective σε attribute του planner (self.lambda_weight).
  2) Να καλέσει plan(...) με keyword για το λ (lambda_weight / risk_weight / lambda_risk).
  3) Αν όλα αυτά δεν ταιριάξουν στο signature, καλεί plan() χωρίς λ
     αλλά o planner έχει ήδη self.lambda_weight αν υπάρχει.
"""

from typing import Any, Tuple

from user_preferences import parse_human_preference, HumanPreference
from human_risk_policy import HumanRiskPolicy, HumanRiskConfig


class HumanAwarePlannerWrapper:
    """
    Τυλίγει έναν υπάρχοντα planner που έχει μέθοδο plan(...).

    Παράδειγμα χρήσης:

        from modules.graph_planning.prm_planner import PRMPlanner

        base_planner = PRMPlanner(...)
        wrapper = HumanAwarePlannerWrapper(
            underlying_planner=base_planner,
            human_pref_text="Προτίμηση: πιο ασφαλής διαδρομή, ακόμη κι αν είναι αργή",
            human_influence_scale=1.0,
        )

        result, lambda_eff = wrapper.plan_with_human_lambda(
            lambda_robot=1.0,
            start=start,
            goal=goal,
            # ... άλλα kwargs για τον planner ...
        )
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

        cfg = HumanRiskConfig(
            human_influence_scale=human_influence_scale,
            low_feature_penalty=5.0,
            dark_area_penalty=5.0,
        )
        self.human_risk_policy = HumanRiskPolicy(config=cfg)

    # --- API για αλλαγή προτίμησης on-the-fly ---

    def set_human_preference(self, text: str):
        """Αλλαγή της ανθρώπινης προτίμησης (π.χ. από UI)."""
        self.human_pref_text = text
        self.human_pref = parse_human_preference(text)

    # --- Συνδυασμός robot λ + human preference ---

    def compute_lambda_effective(self, lambda_robot: float) -> float:
        """
        Συνδυάζει:
            lambda_robot (σύστημα)
        και
            human_pref.risk_preference (άνθρωπος)
        σε lambda_effective.
        """
        return self.human_risk_policy.compute_lambda_effective(
            lambda_robot=lambda_robot,
            human_pref=self.human_pref,
        )

    # --- Κεντρική μέθοδος για experiments ---

    def plan_with_human_lambda(
        self,
        lambda_robot: float,
        *args,
        **kwargs,
    ) -> Tuple[Any, float]:
        """
        Παίρνει:
            lambda_robot: risk weight που θα είχε ο planner χωρίς άνθρωπο
            *args, **kwargs: τα υπόλοιπα arguments του planner.plan(...)

        Κάνει:
        - Υπολογίζει lambda_effective.
        - Αν ο planner έχει attribute 'lambda_weight', το ορίζει.
        - Προσπαθεί να καλέσει plan(...) με keyword για το λ.
        - Αν δεν ταιριάξει το signature, κάνει fallback σε απλή κλήση plan().

        Επιστρέφει:
            (result, lambda_effective)
        όπου result είναι ό,τι επιστρέφει ο underlying planner.
        """
        lambda_eff = self.compute_lambda_effective(lambda_robot)

        # 1) Γράφουμε το λ ως attribute αν υπάρχει
        if hasattr(self.planner, "lambda_weight"):
            setattr(self.planner, "lambda_weight", lambda_eff)

        # 2) Προσπαθούμε να το περάσουμε ως keyword σε plan(...)
        try:
            return (
                self.planner.plan(*args, lambda_weight=lambda_eff, **kwargs),
                lambda_eff,
            )
        except TypeError:
            try:
                return (
                    self.planner.plan(*args, risk_weight=lambda_eff, **kwargs),
                    lambda_eff,
                )
            except TypeError:
                try:
                    return (
                        self.planner.plan(*args, lambda_risk=lambda_eff, **kwargs),
                        lambda_eff,
                    )
                except TypeError:
                    # 3) Τελικό fallback: καλούμε χωρίς λ,
                    # αλλά ο planner έχει ήδη self.lambda_weight αν το υποστηρίζει.
                    return self.planner.plan(*args, **kwargs), lambda_eff
