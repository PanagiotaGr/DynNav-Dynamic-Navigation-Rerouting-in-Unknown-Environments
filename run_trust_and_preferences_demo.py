# run_trust_and_preferences_demo.py
#
# Συνδυάζει:
#   - TrustDynamics (self_trust_robot, human_trust_in_robot)
#   - HumanPreference (από κείμενο)
#   - HumanRiskPolicy (για lambda_effective)
#
# Για κάθε βήμα:
#   1) συμβαίνει ένα event (SUCCESS / FAILURE / κτλ.)
#   2) ενημερώνεται η εμπιστοσύνη
#   3) προκύπτουν:
#        - lambda_robot (από trust)
#        - human_influence_scale (από trust)
#   4) δίνουμε κάποιο human preference text
#   5) υπολογίζουμε lambda_effective = f(lambda_robot, human_pref, human_influence_scale)
#
# Έτσι βλέπουμε πώς "μαθαίνουν" άνθρωπος και ρομπότ
# και πώς αυτό αλλάζει το risk weighting στην πλοήγηση.


from trust_dynamics import TrustDynamics, TrustConfig, TrustEventType
from human_risk_policy import HumanRiskPolicy, HumanRiskConfig
from user_preferences import parse_human_preference


def main():
    # 1. Ορίζουμε TrustDynamics
    trust_cfg = TrustConfig()
    trust = TrustDynamics(config=trust_cfg)

    # 2. Ορίζουμε μια ακολουθία από (event, human_pref_text)
    #    Σαν να έχουμε επεισόδια πλοήγησης με feedback και διαφορετικές προτιμήσεις.
    steps = [
        # (event, human_pref_text)
        (
            TrustEventType.SUCCESS,
            "Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο",
        ),
        (
            TrustEventType.NEAR_MISS,
            "Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο",
        ),
        (
            TrustEventType.FAILURE,
            "Προτίμηση: πιο ασφαλής διαδρομή, ακόμη κι αν είναι αργή",
        ),
        (
            TrustEventType.HUMAN_OVERRIDE,
            "Αποφεύγετε σκοτεινές / low-feature περιοχές",
        ),
        (
            TrustEventType.SUCCESS,
            "balanced",  # default / ουδέτερη προτίμηση
        ),
        (
            TrustEventType.HUMAN_APPROVAL,
            "χωρίς ρίσκο, θέλω πολύ ασφαλή διαδρομή",
        ),
    ]

    print("=== Trust + Human Preference Integration Demo ===\n")
    print("Initial trust state & policy knobs:")
    print(trust.export_policy_knobs())
    print("-" * 80)

    for i, (event, pref_text) in enumerate(steps, start=1):
        print(f"Step {i}:")
        print(f"  Event            : {event.name}")
        print(f"  Human preference : {pref_text}")

        # 1) Ενημέρωση εμπιστοσύνης με βάση το event
        trust.update_trust(event)
        knobs = trust.export_policy_knobs()

        lambda_robot = knobs["lambda_robot"]
        human_influence_scale = knobs["human_influence_scale"]

        # 2) Φτιάχνουμε HumanRiskPolicy με το τρέχον human_influence_scale
        risk_cfg = HumanRiskConfig(
            human_influence_scale=human_influence_scale,
            # Μπορείς να αλλάξεις penalties αν θέλεις
            low_feature_penalty=5.0,
            dark_area_penalty=5.0,
        )
        risk_policy = HumanRiskPolicy(config=risk_cfg)

        # 3) Κάνουμε parse το human preference text
        human_pref = parse_human_preference(pref_text)

        # 4) Υπολογίζουμε lambda_effective
        lambda_eff = risk_policy.compute_lambda_effective(
            lambda_robot=lambda_robot,
            human_pref=human_pref,
        )

        print(f"  self_trust_robot      = {knobs['self_trust_robot']:.3f}")
        print(f"  human_trust_in_robot  = {knobs['human_trust_in_robot']:.3f}")
        print(f"  lambda_robot          = {lambda_robot:.3f}")
        print(f"  human_influence_scale = {human_influence_scale:.3f}")
        print(f"  -> risk_preference h  = {human_pref.risk_preference:.3f}")
        print(f"  -> lambda_effective   = {lambda_eff:.3f}")
        print(f"  safe_mode             = {knobs['safe_mode']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
