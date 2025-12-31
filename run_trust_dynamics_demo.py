# run_trust_dynamics_demo.py
#
# Demo για Human–Robot Trust Dynamics.
#
# Δείχνει:
# - Πώς εξελίσσονται self_trust_robot & human_trust_in_robot σε ακολουθία από events
# - Πώς αυτό αλλάζει:
#     * lambda_robot
#     * human_influence_scale
#     * safe_mode flag
#
# Μπορείς μετά να τα connectάρεις κατευθείαν με:
# - HumanRiskPolicy (για human_influence_scale)
# - HumanAwarePlannerWrapper (για λ_robot)


from trust_dynamics import TrustDynamics, TrustConfig, TrustEventType


def main():
    cfg = TrustConfig()
    trust = TrustDynamics(config=cfg)

    # Demo sequence: μπορείς να αλλάξεις events κατά βούληση.
    event_sequence = [
        TrustEventType.SUCCESS,
        TrustEventType.SUCCESS,
        TrustEventType.NEAR_MISS,
        TrustEventType.SUCCESS,
        TrustEventType.FAILURE,
        TrustEventType.HUMAN_OVERRIDE,
        TrustEventType.SUCCESS,
        TrustEventType.HUMAN_APPROVAL,
        TrustEventType.SUCCESS,
    ]

    print("=== Human–Robot Trust Dynamics Demo ===\n")
    print("Initial state:")
    print(trust.export_policy_knobs())
    print("-" * 72)

    for step, ev in enumerate(event_sequence, start=1):
        print(f"Step {step}: event = {ev.name}")
        trust.update_trust(ev)
        knobs = trust.export_policy_knobs()

        print(
            f"  self_trust_robot      = {knobs['self_trust_robot']:.3f}\n"
            f"  human_trust_in_robot  = {knobs['human_trust_in_robot']:.3f}\n"
            f"  lambda_robot          = {knobs['lambda_robot']:.3f}\n"
            f"  human_influence_scale = {knobs['human_influence_scale']:.3f}\n"
            f"  safe_mode             = {knobs['safe_mode']}"
        )
        print("-" * 72)


if __name__ == "__main__":
    main()
