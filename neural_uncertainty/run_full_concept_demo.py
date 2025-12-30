# =================================================================
# FULL CONCEPT DEMO
#
# Δείχνει ενιαίο pipeline:
# 1) Language → navigation policy
# 2) Online uncertainty estimation (synthetic demo)
# 3) Self-Trust computation
# 4) Adaptive risk budget
# 5) Path selection
# 6) Explainable decision
# =================================================================

import numpy as np
from language_to_navigation_policy import LanguageToNavigationPolicy
from risk_explainer import PathStats, choose_best_path, explain_preference


# ---------- Self-Trust Model ----------
def compute_self_trust(error: float, var: float) -> float:
    # απλό normalized metric
    S = 1.0 / (1.0 + error + var * 10.0)
    return float(np.clip(S, 0.0, 1.0))


# ---------- Adaptive λ ----------
def lambda_from_self_trust(S: float) -> float:
    lambda_min = 0.3
    lambda_max = 3.0
    return lambda_min + (lambda_max - lambda_min) * (1.0 - S)


# ---------- Demo Paths ----------
def get_demo_paths():
    return [
        PathStats("A", length=10.0, drift_exposure=5.0, uncertainty_exposure=2.0),
        PathStats("B", length=11.0, drift_exposure=3.0, uncertainty_exposure=1.5),
        PathStats("C", length=13.0, drift_exposure=2.0, uncertainty_exposure=1.0),
    ]


# ---------- MAIN DEMO ----------
def run_demo(user_text: str, simulated_error: float, simulated_var: float):
    print("\n============================================================")
    print(f"[DEMO] USER SAID: \"{user_text}\"")

    # ---------------- Language policy ----------------
    mapper = LanguageToNavigationPolicy()
    lang = mapper.interpret(user_text)

    print("\n[LANG POLICY]")
    print(f"  lambda_risk override : {lang.lambda_risk}")
    print(f"  speed_scale          : {lang.speed_scale}")
    print(f"  avoid_right_wall     : {lang.avoid_right_wall}")
    print(f"  avoid_left_wall      : {lang.avoid_left_wall}")
    print(f"  exploration bias     : {lang.increase_exploration}")
    print(f"  Reasoning            : {lang.explanation}")

    # ---------------- Online uncertainty / error feedback ---------------
    print("\n[UNCERTAINTY FEEDBACK]")
    print(f"  online error estimate : {simulated_error:.4f}")
    print(f"  online variance       : {simulated_var:.6f}")

    # ---------------- Self Trust ----------------
    S = compute_self_trust(simulated_error, simulated_var)
    print(f"\n[SELF TRUST]")
    print(f"  Self-Trust S = {S:.3f}")

    if S > 0.7:
        mode = "NORMAL"
    elif S > 0.4:
        mode = "CAUTIOUS"
    else:
        mode = "SAFE / CONSERVATIVE"
    print(f"  Mode selected = {mode}")

    # ---------------- Adaptive Risk Budget ----------------
    lam = lambda_from_self_trust(S)

    # αν η γλώσσα έβαλε συγκεκριμένο λ, το σεβόμαστε
    if lang.lambda_risk is not None:
        lam = lang.lambda_risk

    print(f"\n[ADAPTIVE RISK]")
    print(f"  Final λ = {lam:.3f}")

    # ---------------- Path Decision ----------------
    paths = get_demo_paths()
    best = choose_best_path(paths, lambda_risk=lam)

    print("\n[PATH DECISION]")
    print(f"  Selected path = {best.name}")

    print("\n[EXPLANATION]")
    for p in paths:
        if p.name != best.name:
            print(" ", explain_preference(best, p, lambda_risk=lam))


def main():
    # 3 σενάρια
    run_demo(
        "πήγαινε γρήγορα",
        simulated_error=0.2,
        simulated_var=0.0008,
    )

    run_demo(
        "πρόσεχε, είναι επικίνδυνο",
        simulated_error=0.9,
        simulated_var=0.004,
    )

    run_demo(
        "πήγαινε αργά και πρόσεχε δεξιά",
        simulated_error=0.6,
        simulated_var=0.002,
    )


if __name__ == "__main__":
    main()
