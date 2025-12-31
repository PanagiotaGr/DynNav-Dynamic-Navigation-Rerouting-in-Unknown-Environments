"""
Μικρό demo script για human risk preferences.

Τι κάνει:
- Ορίζει μερικές φυσικές φράσεις προτίμησης (στα Ελληνικά / Αγγλικά).
- Τις περνάει στον parser (user_preferences.parse_human_preference).
- Υπολογίζει ένα "effective" lambda για τον planner.
    lambda_robot = 1.0 (dummy)
    lambda_effective = lambda_robot * factor(h)

Έτσι μπορείς να δεις άμεσα ότι ο parser δουλεύει
και ότι αλλάζει το risk weight ανάλογα με το comfort level.
"""

from user_preferences import parse_human_preference, HumanPreference
from human_risk_policy import HumanRiskPolicy, HumanRiskConfig


def compute_lambda_effective(lambda_robot: float,
                             human_pref: HumanPreference,
                             human_influence_scale: float = 1.0) -> float:
    """
    Wrapper γύρω από HumanRiskPolicy για να κρατήσουμε το ίδιο interface
    με το προηγούμενο demo.
    """
    cfg = HumanRiskConfig(human_influence_scale=human_influence_scale)
    policy = HumanRiskPolicy(config=cfg)
    return policy.compute_lambda_effective(lambda_robot, human_pref)


def main():
    # Μερικά examples σαν αυτά που έγραψες
    preference_texts = [
        "Προτίμηση: πιο ασφαλής διαδρομή, ακόμη κι αν είναι αργή",
        "Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο",
        "Αποφεύγετε σκοτεινές / low-feature περιοχές",
        "balanced",
        "χωρίς ρίσκο, θέλω πολύ ασφαλή διαδρομή",
        "aggressive driving, accept high risk for speed",
    ]

    lambda_robot = 1.0      # dummy baseline
    human_alpha = 1.0       # πόσο δυνατά επηρεάζει ο άνθρωπος

    print("=== Human preference demo ===")
    print(f"lambda_robot (baseline) = {lambda_robot}")
    print(f"human_influence_scale (alpha) = {human_alpha}")
    print()

    for text in preference_texts:
        pref = parse_human_preference(text)
        lambda_eff = compute_lambda_effective(lambda_robot, pref, human_alpha)

        print("Preference text:", text)
        print(f"  -> risk_preference h = {pref.risk_preference:.2f}")
        print(f"  -> avoid_low_feature_areas = {pref.avoid_low_feature_areas}")
        print(f"  -> avoid_dark_areas       = {pref.avoid_dark_areas}")
        print(f"  -> prefer_well_mapped     = {pref.prefer_well_mapped_areas}")
        print(f"  -> lambda_effective       = {lambda_eff:.3f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
