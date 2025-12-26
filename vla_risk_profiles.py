"""
vla_risk_profiles.py

Language-level intents -> planner weight profiles.

Χρησιμοποιείται για multi-objective navigation:
- path length
- fused risk / uncertainty
- entropy / information gain
- coverage

Η ιδέα είναι ότι ένα high-level "intent" (π.χ. "cautious_navigation")
χαρτογραφείται σε διαφορετικά βάρη / παραμέτρους.
"""

from dataclasses import dataclass


@dataclass
class PlannerWeights:
    """Multi-objective βάρη + ρυθμίσεις risk για ένα intent."""
    length_weight: float       # πόσο penalize το path length
    risk_weight: float         # πόσο penalize fused risk / uncertainty
    entropy_weight: float      # πόσο reward IG / entropy reduction
    coverage_weight: float     # πόσο reward coverage
    lambda_scale: float = 1.0  # scale factor για belief–risk lambda
    risk_budget_factor: float = 1.0  # scale για risk budget (αν χρησιμοποιείται)


# Βασικό λεξικό intent -> profile
INTENT_PROFILES = {
    # Πολύ προσεκτική πλοήγηση: μεγάλο βάρος στο risk,
    # λίγο μικρότερο στο length, ήπιο σε entropy/coverage.
    "cautious_navigation": PlannerWeights(
        length_weight=0.6,
        risk_weight=1.0,
        entropy_weight=0.4,
        coverage_weight=0.4,
        lambda_scale=1.5,
        risk_budget_factor=0.7,
    ),

    # Γρήγορη πλοήγηση: προτεραιότητα στο μικρό path length,
    # μικρότερο βάρος στο risk.
    "fast_navigation": PlannerWeights(
        length_weight=1.0,
        risk_weight=0.4,
        entropy_weight=0.3,
        coverage_weight=0.3,
        lambda_scale=0.7,
        risk_budget_factor=1.5,
    ),

    # Exploration / mapping: emphasizes entropy & coverage.
    "exploration_focused": PlannerWeights(
        length_weight=0.4,
        risk_weight=0.6,
        entropy_weight=1.0,
        coverage_weight=1.0,
        lambda_scale=1.0,
        risk_budget_factor=1.0,
    ),

    # Βελτίωση localization (π.χ. σε φτωχό texture): IG-heavy
    "localization_focused": PlannerWeights(
        length_weight=0.4,
        risk_weight=0.7,
        entropy_weight=1.0,
        coverage_weight=0.5,
        lambda_scale=1.2,
        risk_budget_factor=0.9,
    ),
}


def get_planner_profile(intent: str) -> PlannerWeights:
    """
    Επιστρέφει το PlannerWeights για το δοσμένο intent.
    Αν το intent δεν είναι γνωστό, γυρνάει ένα ουδέτερο profile.
    """
    intent = intent.strip().lower()
    if intent in INTENT_PROFILES:
        return INTENT_PROFILES[intent]

    # default ουδέτερο profile
    return PlannerWeights(
        length_weight=0.7,
        risk_weight=0.7,
        entropy_weight=0.7,
        coverage_weight=0.7,
        lambda_scale=1.0,
        risk_budget_factor=1.0,
    )
