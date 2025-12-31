"""
simple_risk_planner_human.py

Ίδιο demo με simple_risk_planner.py, αλλά:

- Χρησιμοποιούμε HumanPreference + HumanRiskPolicy.
- Συνδυάζουμε:
      lambda_robot (σύστημα)
      + human_pref.risk_preference (άνθρωπος)
  για να πάρουμε:
      lambda_effective

- Το cost γίνεται:
      cost = length + lambda_effective * risk
"""

from user_preferences import parse_human_preference, HumanPreference
from human_risk_policy import HumanRiskPolicy, HumanRiskConfig


class SimpleHumanAwareRiskPlanner:
    def __init__(
        self,
        lambda_robot: float = 1.0,
        human_pref_text: str = "Προτίμηση: πιο ασφαλής διαδρομή, ακόμη κι αν είναι αργή",
        human_influence_scale: float = 1.0,
    ):
        self.lambda_robot = lambda_robot

        # Human preference από text
        self.human_pref_text = human_pref_text
        self.human_pref: HumanPreference = parse_human_preference(human_pref_text)

        # Policy config
        cfg = HumanRiskConfig(
            human_influence_scale=human_influence_scale,
            low_feature_penalty=5.0,
            dark_area_penalty=5.0,
        )
        self.human_risk_policy = HumanRiskPolicy(config=cfg)

    def compute_lambda_effective(self) -> float:
        """
        Συνδυασμός robot-side λ με ανθρώπινη προτίμηση.
        """
        return self.human_risk_policy.compute_lambda_effective(
            lambda_robot=self.lambda_robot,
            human_pref=self.human_pref,
        )

    def evaluate_path_cost(
        self,
        path_length: float,
        path_risk: float,
        is_dark: bool = False,
        is_low_feature: bool = False,
    ) -> float:
        """
        Κόστος path με χρήση HumanRiskPolicy.edge_cost, για να μπορούμε
        να αξιοποιήσουμε avoid_dark / avoid_low_feature αν χρειαστεί.
        """
        lambda_eff = self.compute_lambda_effective()
        cost = self.human_risk_policy.edge_cost(
            base_length=path_length,
            edge_risk=path_risk,
            lambda_effective=lambda_eff,
            human_pref=self.human_pref,
            is_dark=is_dark,
            is_low_feature=is_low_feature,
        )
        return cost, lambda_eff

    def select_best_path(self, candidates):
        best = None
        best_cost = float("inf")
        best_lambda = None

        for c in candidates:
            cost, lambda_eff = self.evaluate_path_cost(
                path_length=c["length"],
                path_risk=c["risk"],
                is_dark=c.get("is_dark", False),
                is_low_feature=c.get("is_low_feature", False),
            )
            if cost < best_cost:
                best_cost = cost
                best = c
                best_lambda = lambda_eff

        return best, best_cost, best_lambda


def main():
    # Ίδια paths όπως στο simple_risk_planner, αλλά εδώ μπορείς
    # να βάλεις flags για σκοτεινές / low-feature περιοχές.
    candidates = [
        {"name": "A", "length": 10.0, "risk": 0.2, "is_dark": False, "is_low_feature": False},
        {"name": "B", "length": 8.0, "risk": 0.6, "is_dark": True,  "is_low_feature": True},
        {"name": "C", "length": 12.0, "risk": 0.1, "is_dark": False, "is_low_feature": True},
    ]

    lambda_robot = 1.0

    # Δοκίμασε διαφορετικά human preference texts:
    human_pref_text = "Προτίμηση: πιο ασφαλής διαδρομή, ακόμη κι αν είναι αργή"
    # human_pref_text = "Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο"
    # human_pref_text = "Αποφεύγετε σκοτεινές / low-feature περιοχές"

    human_influence_scale = 1.0

    planner = SimpleHumanAwareRiskPlanner(
        lambda_robot=lambda_robot,
        human_pref_text=human_pref_text,
        human_influence_scale=human_influence_scale,
    )

    print("=== Human-aware risk planner ===")
    print(f"lambda_robot (baseline)      = {lambda_robot}")
    print(f"human_pref_text              = {human_pref_text}")
    print(f"human_influence_scale (alpha)= {human_influence_scale}")
    print()

    for c in candidates:
        cost, lambda_eff = planner.evaluate_path_cost(
            path_length=c["length"],
            path_risk=c["risk"],
            is_dark=c.get("is_dark", False),
            is_low_feature=c.get("is_low_feature", False),
        )
        print(
            f"Path {c['name']}: length={c['length']}, risk={c['risk']}, "
            f"is_dark={c.get('is_dark', False)}, "
            f"is_low_feature={c.get('is_low_feature', False)}"
        )
        print(f"    lambda_effective = {lambda_eff:.3f}, cost = {cost:.3f}")

    best, best_cost, best_lambda = planner.select_best_path(candidates)
    print("\nBest path:", best["name"])
    print(f"  -> cost = {best_cost:.3f}")
    print(f"  -> lambda_effective used = {best_lambda:.3f}")


if __name__ == "__main__":
    main()
