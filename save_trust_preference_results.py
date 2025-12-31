# save_trust_preference_results.py
#
# Αποθηκεύει σε CSV τη δυναμική:
#   - TrustDynamics (self_trust, human_trust)
#   - HumanPreference (h)
#   - lambda_robot, human_influence_scale, lambda_effective
#
# Χρήσιμο για plots / πίνακες σε εργασία.

import csv

from trust_dynamics import TrustDynamics, TrustConfig, TrustEventType
from human_risk_policy import HumanRiskPolicy, HumanRiskConfig
from user_preferences import parse_human_preference


def main():
    csv_path = "trust_preference_results.csv"

    trust_cfg = TrustConfig()
    trust = TrustDynamics(config=trust_cfg)

    # Ίδια sequence με το demo (μπορείς να την αλλάξεις / επεκτείνεις)
    steps = [
        (TrustEventType.SUCCESS, "Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο"),
        (TrustEventType.NEAR_MISS, "Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο"),
        (TrustEventType.FAILURE, "Προτίμηση: πιο ασφαλής διαδρομή, ακόμη κι αν είναι αργή"),
        (TrustEventType.HUMAN_OVERRIDE, "Αποφεύγετε σκοτεινές / low-feature περιοχές"),
        (TrustEventType.SUCCESS, "balanced"),
        (TrustEventType.HUMAN_APPROVAL, "χωρίς ρίσκο, θέλω πολύ ασφαλή διαδρομή"),
    ]

    fieldnames = [
        "step",
        "event",
        "human_pref_text",
        "self_trust_robot",
        "human_trust_in_robot",
        "lambda_robot",
        "human_influence_scale",
        "risk_preference_h",
        "lambda_effective",
        "safe_mode",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # initial state (optional step 0)
        knobs0 = trust.export_policy_knobs()
        writer.writerow({
            "step": 0,
            "event": "INIT",
            "human_pref_text": "",
            "self_trust_robot": knobs0["self_trust_robot"],
            "human_trust_in_robot": knobs0["human_trust_in_robot"],
            "lambda_robot": knobs0["lambda_robot"],
            "human_influence_scale": knobs0["human_influence_scale"],
            "risk_preference_h": "",
            "lambda_effective": "",
            "safe_mode": knobs0["safe_mode"],
        })

        # main steps
        for i, (event, pref_text) in enumerate(steps, start=1):
            # update trust
            trust.update_trust(event)
            knobs = trust.export_policy_knobs()

            lambda_robot = knobs["lambda_robot"]
            human_influence_scale = knobs["human_influence_scale"]

            # risk policy for this step
            risk_cfg = HumanRiskConfig(
                human_influence_scale=human_influence_scale,
                low_feature_penalty=5.0,
                dark_area_penalty=5.0,
            )
            risk_policy = HumanRiskPolicy(config=risk_cfg)

            # parse human preference
            human_pref = parse_human_preference(pref_text)

            # compute lambda_effective
            lambda_eff = risk_policy.compute_lambda_effective(
                lambda_robot=lambda_robot,
                human_pref=human_pref,
            )

            writer.writerow({
                "step": i,
                "event": event.name,
                "human_pref_text": pref_text,
                "self_trust_robot": knobs["self_trust_robot"],
                "human_trust_in_robot": knobs["human_trust_in_robot"],
                "lambda_robot": lambda_robot,
                "human_influence_scale": human_influence_scale,
                "risk_preference_h": human_pref.risk_preference,
                "lambda_effective": lambda_eff,
                "safe_mode": knobs["safe_mode"],
            })

    print(f"Saved trust + preference results to: {csv_path}")


if __name__ == "__main__":
    main()
