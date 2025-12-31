"""
risk_cost_utils.py

Utility συναρτήσεις για cost σε risk-aware navigation με ανθρώπινες προτιμήσεις.

Κύρια συνάρτηση:

    human_aware_edge_cost(
        base_length,
        edge_risk,
        lambda_effective,
        human_pref,
        is_dark=False,
        is_low_feature=False,
        low_feature_penalty=5.0,
        dark_area_penalty=5.0,
    )

Αυτή υλοποιεί:

    cost = base_length + lambda_effective * edge_risk
           + penalties(αν ο άνθρωπος θέλει να αποφεύγει σκοτεινές / low-feature περιοχές)
"""

from user_preferences import HumanPreference


def human_aware_edge_cost(
    base_length: float,
    edge_risk: float,
    lambda_effective: float,
    human_pref: HumanPreference,
    is_dark: bool = False,
    is_low_feature: bool = False,
    low_feature_penalty: float = 5.0,
    dark_area_penalty: float = 5.0,
) -> float:
    """
    Υπολογίζει το κόστος ενός edge / segment με βάση:

        base_length           : γεωμετρικό μήκος / χρόνο / energy
        edge_risk             : κάποιο risk metric (π.χ. κοντά σε εμπόδια, μεγάλη αβεβαιότητα)
        lambda_effective      : risk weight μετά τον συνδυασμό robot + human
        human_pref            : αντικείμενο HumanPreference
        is_dark               : αν η ακμή περνάει από σκοτεινή περιοχή
        is_low_feature        : αν η ακμή περνάει από low-feature περιοχή
        low_feature_penalty   : penalty για low-feature περιοχές (αν το ζητά ο άνθρωπος)
        dark_area_penalty     : penalty για σκοτεινές περιοχές (αν το ζητά ο άνθρωπος)

    Επιστρέφει:
        cost = base_length + lambda_effective * edge_risk + penalties
    """
    cost = base_length + lambda_effective * edge_risk

    if human_pref.avoid_dark_areas and is_dark:
        cost += dark_area_penalty

    if human_pref.avoid_low_feature_areas and is_low_feature:
        cost += low_feature_penalty

    return cost
