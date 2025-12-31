"""
simple_risk_planner.py

Απλός risk-aware planner demo ΧΩΡΙΣ ανθρώπινη προτίμηση.

- Έχουμε υποθετικά 3 paths.
- Κάθε path έχει (length, risk).
- Χρησιμοποιούμε:
      cost = length + lambda_robot * risk
- Επιλέγουμε το path με το μικρότερο cost.
"""


class SimpleRiskPlanner:
    def __init__(self, lambda_robot: float = 1.0):
        self.lambda_robot = lambda_robot

    def evaluate_path_cost(self, path_length: float, path_risk: float) -> float:
        """
        Βασικό cost:
            cost = length + λ * risk
        """
        return path_length + self.lambda_robot * path_risk

    def select_best_path(self, candidates):
        """
        candidates: λίστα από dicts με keys:
            - 'name'
            - 'length'
            - 'risk'
        """
        best = None
        best_cost = float("inf")

        for c in candidates:
            cost = self.evaluate_path_cost(c["length"], c["risk"])
            if cost < best_cost:
                best_cost = cost
                best = (c, cost)

        return best


def main():
    # Υποθετικά 3 paths
    candidates = [
        {"name": "A", "length": 10.0, "risk": 0.2},
        {"name": "B", "length": 8.0, "risk": 0.6},
        {"name": "C", "length": 12.0, "risk": 0.1},
    ]

    lambda_robot = 1.0  # baseline risk weight
    planner = SimpleRiskPlanner(lambda_robot=lambda_robot)

    print("=== Baseline risk-aware planner (NO human preference) ===")
    print(f"lambda_robot = {lambda_robot}\n")

    for c in candidates:
        cost = planner.evaluate_path_cost(c["length"], c["risk"])
        print(
            f"Path {c['name']}: length={c['length']}, "
            f"risk={c['risk']}, cost={cost:.3f}"
        )

    best, best_cost = planner.select_best_path(candidates)
    print("\nBest path:", best["name"], "with cost:", best_cost)


if __name__ == "__main__":
    main()
