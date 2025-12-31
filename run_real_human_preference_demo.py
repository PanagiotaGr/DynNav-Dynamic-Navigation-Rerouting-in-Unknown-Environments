"""
run_real_human_preference_demo.py

Demo: Human-aware risk adaptation πάνω σε έναν ΠΡΑΓΜΑΤΙΚΟ planner του repo:
    modules/graph_planning/prm_planner.py -> class PRMPlanner

Χρησιμοποιεί:
- HumanAwarePlannerWrapper (human_aware_real_planner.py)
- HumanPreference (user_preferences.py)
- HumanRiskPolicy (human_risk_policy.py)

ΣΕ ΣΗΜΕΙΑ ΜΕ TODO πρέπει να βάλεις τα δικά σου arguments,
ανάλογα με το πώς φτιάχνεις συνήθως τον PRMPlanner και πώς καλείς το plan().
"""

from human_aware_real_planner import HumanAwarePlannerWrapper
from modules.graph_planning.prm_planner import PRMPlanner


# =========================
# 1. Δημιουργία base planner
# =========================
def build_base_planner():
    """
    ΕΔΩ προσαρμόζεις τον constructor του PRMPlanner.

    Άνοιξε το modules/graph_planning/prm_planner.py και δες:

        class PRMPlanner:
            def __init__(self, ...):

    και βάλε εδώ τα ίδια arguments που βάζεις και στα κανονικά experiments.
    Προς το παρόν έχουμε placeholder χωρίς arguments.
    """
    # TODO: άλλαξε τα arguments σύμφωνα με τον δικό σου PRMPlanner.
    # Παράδειγμα (ΔΕΝ ΕΙΝΑΙ ΑΛΗΘΙΝΟ, είναι μόνο template):
    #
    #   planner = PRMPlanner(
    #       occupancy_grid=occ_grid,
    #       resolution=0.1,
    #       num_samples=2000,
    #       ...
    #   )
    #
    # Για τώρα, το αφήνουμε όσο πιο απλό γίνεται:
    planner = PRMPlanner()
    return planner


# ======================================
# 2. Demo χρήσης με human-aware wrapper
# ======================================
def main():
    # --- 2.1 Φτιάχνουμε baseline PRMPlanner ---
    base_planner = build_base_planner()

    # --- 2.2 Τυλίγουμε σε HumanAwarePlannerWrapper ---
    # Μπορείς να αλλάζεις το κείμενο προτίμησης εδώ:
    human_pref_text = "Προτίμηση: πιο ασφαλής διαδρομή, ακόμη κι αν είναι αργή"
    # human_pref_text = "Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο"
    # human_pref_text = "Αποφεύγετε σκοτεινές / low-feature περιοχές"

    human_influence_scale = 1.0

    human_planner = HumanAwarePlannerWrapper(
        underlying_planner=base_planner,
        human_pref_text=human_pref_text,
        human_influence_scale=human_influence_scale,
    )

    # --- 2.3 Ορίζουμε κάποιο start / goal για demo ---
    # ΕΔΩ βάζεις ό,τι χρησιμοποιεί ο PRMPlanner σου.
    # Συνήθως κάτι σαν (x, y) ή node IDs.

    # TODO: βάλε πραγματικές τιμές:
    start = (0.0, 0.0)
    goal = (5.0, 5.0)

    # --- 2.4 Υποθέτουμε ότι το λ_robot έρχεται από self-trust/OOD/drift ---
    lambda_robot = 1.0  # προς το παρόν απλό demo

    print("=== Real PRMPlanner + Human Preference Demo ===")
    print(f"Human preference text      : {human_pref_text}")
    print(f"Human influence scale (α)  : {human_influence_scale}")
    print(f"Baseline lambda_robot      : {lambda_robot}")
    print(f"Start                      : {start}")
    print(f"Goal                       : {goal}")
    print()

    try:
        # ΕΔΩ πρέπει να ταιριάξεις τα arguments του PRMPlanner.plan(...)
        #
        # Ο HumanAwarePlannerWrapper θα δοκιμάσει:
        #   planner.plan(..., lambda_weight=lambda_eff, **kwargs)
        # μετά:
        #   planner.plan(..., risk_weight=lambda_eff, **kwargs)
        # μετά:
        #   planner.plan(..., lambda_risk=lambda_eff, **kwargs)
        # και αν όλα αποτύχουν, καλεί:
        #   planner.plan(..., **kwargs)
        #
        # Άρα ΕΣΥ πρέπει να βάλεις τα σωστά positional args εδώ.
        #
        # Παράδειγμα (ΔΕΝ είναι πραγματικός κώδικας, μόνο template):
        #   result, lambda_eff = human_planner.plan_with_human_lambda(
        #       lambda_robot=lambda_robot,
        #       start=start,
        #       goal=goal,
        #       map=occupancy_grid,
        #   )

        result, lambda_eff = human_planner.plan_with_human_lambda(
            lambda_robot=lambda_robot,
            start=start,
            goal=goal,
        )

        print("Planner result (από PRMPlanner):", result)
        print("lambda_effective used           :", lambda_eff)

    except TypeError as e:
        print("⚠ TypeError κατά την κλήση του PRMPlanner.plan(...).")
        print("Πιθανότατα:")
        print("  - Το PRMPlanner.plan(...) θέλει άλλα arguments (π.χ. διαφορετικά ονόματα).")
        print("  - Ή δεν παίρνει καθόλου lambda_weight / risk_weight ως keyword.")
        print()
        print("Τρέχον error μήνυμα:")
        print(e)
        print()
        print("Βήματα που μπορείς να κάνεις:")
        print("  1) Άνοιξε modules/graph_planning/prm_planner.py και δες:")
        print("        def plan(self, ...):")
        print("     για να δεις ακριβώς τα ορίσματα.")
        print("  2) Προσαρμόσε το call:")
        print("        human_planner.plan_with_human_lambda(")
        print("            lambda_robot=lambda_robot,")
        print("            <τα σωστά positional/keyword args>")
        print("        )")
        print("  3) Αν ο PRMPlanner ΔΕΝ παίρνει λ ως argument, τότε:")
        print("        - μπορούμε να το βάλουμε ως attribute (π.χ. self.lambda_risk)")
        print("        - ή να τυλίξουμε το cost function μέσα στο PRMPlanner με HumanRiskPolicy.")


if __name__ == "__main__":
    main()
