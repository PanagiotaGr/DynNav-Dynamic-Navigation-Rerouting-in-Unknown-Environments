"""
human_risk_policy.py

Layer που συνδυάζει:
- robot-side risk weight (lambda_robot)
- human risk preference (HumanPreference)
- περιβαλλοντικούς περιορισμούς (avoid dark / low-feature)

Χρησιμοποιείται πάνω από τον υπάρχοντα risk-aware planner.
"""

from dataclasses import dataclass
from typing import Optional

from user_preferences import HumanPreference


@dataclass
class HumanRiskConfig:
    """
    Ρυθμίσεις για το πώς ο άνθρωπος επηρεάζει το risk weight.
    """
    human_influence_scale: float = 1.0  # alpha
    min_lambda_factor: float = 0.1      # κάτω όριο στο factor
    max_lambda_factor: float = 3.0      # πάνω όριο στο factor

    # penalties για dark / low-feature περιοχές
    low_feature_penalty: float = 5.0
    dark_area_penalty: float = 5.0


class HumanRiskPolicy:
    """
    Policy που παίρνει:
    - λ_robot από το σύστημα (self-trust / OOD / drift / οτιδήποτε)
    - HumanPreference από φυσική γλώσσα

    και επιστρέφει:
    - λ_effective για τον planner
    - optional penalty terms για edge-level κόστος.
    """

    def __init__(self, config: Optional[HumanRiskConfig] = None):
        self.config = config or HumanRiskConfig()

    def compute_lambda_effective(
        self,
        lambda_robot: float,
        human_pref: HumanPreference,
    ) -> float:
        """
        Συνδυάζουμε λ_robot και human risk preference.

        human_pref.risk_preference = h ∈ [0,1]:
            h = 0 -> πολύ safe
            h = 0.5 -> balanced
            h = 1 -> aggressive

        Χρησιμοποιούμε:
            factor = (1 + alpha) - 2 * alpha * h

        με alpha = human_influence_scale.

        h = 0   -> factor = 1 + alpha  (safe, αυξάνουμε λ)
        h = 0.5 -> factor = 1          (balanced)
        h = 1   -> factor = 1 - alpha  (aggressive, μειώνουμε λ)

        Με clamp [min_lambda_factor, max_lambda_factor] για σταθερότητα.
        """
        h = human_pref.risk_preference
        alpha = self.config.human_influence_scale

        factor = (1.0 + alpha) - 2.0 * alpha * h

        # clamp factor
        factor = max(self.config.min_lambda_factor,
                     min(self.config.max_lambda_factor, factor))

        return lambda_robot * factor

    def edge_cost(
        self,
        base_length: float,
        edge_risk: float,
        lambda_effective: float,
        human_pref: HumanPreference,
        is_dark: bool = False,
        is_low_feature: bool = False,
    ) -> float:
        """
        Υπολογισμός κόστους ενός edge/path segment με ανθρώπινες προτιμήσεις.

        base_length: π.χ. γεωμετρικό μήκος / χρόνο
        edge_risk: risk metric για το edge
        lambda_effective: risk weight μετά τον συνδυασμό robot + human
        is_dark: αν αυτή η ακμή αφορά σκοτεινή περιοχή
        is_low_feature: αν αυτή η ακμή αφορά low-feature περιοχή
        """
        cost = base_length + lambda_effective * edge_risk

        # Penalties από human preferences
        if human_pref.avoid_dark_areas and is_dark:
            cost += self.config.dark_area_penalty

        if human_pref.avoid_low_feature_areas and is_low_feature:
            cost += self.config.low_feature_penalty

        return cost
