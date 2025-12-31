"""
multi_robot_risk_allocation.py

Risk-aware multi-robot task allocation for coverage / exploration.

Βασική ιδέα:
- Έχουμε M ρομπότ και N στόχους (coverage cells / frontiers / viewpoints).
- Κάθε στόχος έχει θέση (x, y) και risk score ∈ [0, 1].
- Φτιάχνουμε cost matrix που συνδυάζει απόσταση + ρίσκο.
- Χρησιμοποιούμε Hungarian algorithm για global optimal assignment.

Μπορεί να χρησιμοποιηθεί μέσα στο multi_robot_coverage_sim / run_multi_robot_experiment
για να αναθέτεις στόχους στα ρομπότ με βάση την αβεβαιότητα.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

try:
    # Χρειαζόμαστε SciPy για το Hungarian (linear_sum_assignment).
    # Αν δεν υπάρχει, πρόσθεσε "scipy" στο requirements.txt και κάνε:
    #   pip install -r requirements.txt
    from scipy.optimize import linear_sum_assignment
except ImportError as e:
    raise ImportError(
        "scipy is required for multi_robot_risk_allocation. "
        "Install it with `pip install scipy` or add it to requirements.txt."
    ) from e


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class RobotState:
    """
    Κατάσταση ρομπότ σε 2D.

    Parameters
    ----------
    id : int
        Μοναδικό ID ρομπότ (π.χ. index ή ROS namespace).
    x, y : float
        Τρέχουσα θέση σε world / map frame.
    """
    id: int
    x: float
    y: float


@dataclass
class CellTarget:
    """
    Στόχος κάλυψης / εξερεύνησης.

    Parameters
    ----------
    id : int
        Μοναδικό ID κελιού / στόχου (π.χ. index σε grid).
    x, y : float
        Θέση του στόχου στο ίδιο frame με τα ρομπότ.
    risk : float
        Risk score ∈ [0, 1], όπου 1.0 = πολύ υψηλό ρίσκο
        (π.χ. υψηλό drift, χαμηλή ορατότητα, χαμηλή υφή).
    """
    id: int
    x: float
    y: float
    risk: float


# --------------------------------------------------------------------------- #
# Cost matrix construction
# --------------------------------------------------------------------------- #

def build_cost_matrix(
    robots: List[RobotState],
    targets: List[CellTarget],
    w_dist: float = 1.0,
    w_risk: float = 1.0,
) -> np.ndarray:
    """
    Χτίζει cost matrix (num_robots x num_targets) με συνδυασμό distance + risk.

    cost[i, j] = w_dist * euclidean_distance(robot_i, target_j)
               + w_risk * risk(target_j)

    Parameters
    ----------
    robots : list of RobotState
    targets : list of CellTarget
    w_dist : float
        Βάρος στην γεωμετρική απόσταση.
    w_risk : float
        Βάρος στο ρίσκο. Μεγαλύτερο w_risk => πιο συντηρητική ανάθεση.

    Returns
    -------
    cost : ndarray, shape (len(robots), len(targets))
    """
    num_robots = len(robots)
    num_targets = len(targets)
    cost = np.zeros((num_robots, num_targets), dtype=float)

    for i, r in enumerate(robots):
        for j, t in enumerate(targets):
            dist = np.hypot(r.x - t.x, r.y - t.y)
            cost[i, j] = w_dist * dist + w_risk * float(t.risk)

    return cost


# --------------------------------------------------------------------------- #
# Homogeneous robots: όλοι με την ίδια στάση απέναντι στο ρίσκο
# --------------------------------------------------------------------------- #

def assign_tasks_hungarian(
    robots: List[RobotState],
    targets: List[CellTarget],
    w_dist: float = 1.0,
    w_risk: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Βρίσκει global optimal ανάθεση ρομπότ-στόχων (Hungarian algorithm)
    με κοινό βάρος ρίσκου για όλα τα ρομπότ.

    Parameters
    ----------
    robots : list of RobotState
    targets : list of CellTarget
    w_dist : float
        Βάρος στην απόσταση.
    w_risk : float
        Βάρος στο ρίσκο.

    Returns
    -------
    assignments : list of (robot_id, target_id)
        Ένα ζεύγος για κάθε αντιστοίχιση.
    total_cost : float
        Άθροισμα των επιλεγμένων costs.
    """
    if len(robots) == 0 or len(targets) == 0:
        return [], 0.0

    cost = build_cost_matrix(robots, targets, w_dist=w_dist, w_risk=w_risk)

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments: List[Tuple[int, int]] = []
    for i, j in zip(row_ind, col_ind):
        assignments.append((robots[i].id, targets[j].id))

    total_cost = float(cost[row_ind, col_ind].sum())
    return assignments, total_cost


# --------------------------------------------------------------------------- #
# Heterogeneous robots: διαφορετικά risk profiles (explorer vs conservative)
# --------------------------------------------------------------------------- #

def assign_with_risk_profiles(
    robots: List[RobotState],
    targets: List[CellTarget],
    risk_profile: Dict[int, float],
    base_w_dist: float = 1.0,
    base_w_risk: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Ανάθεση ρομπότ-στόχων όπου κάθε ρομπότ έχει διαφορετική ευαισθησία στο ρίσκο.

    Για ρομπότ i με profile factor alpha_i:

        cost_i,j = base_w_dist * dist(i, j)
                 + (alpha_i * base_w_risk) * risk(j)

    Παράδειγμα:
    - high-risk explorer: alpha_i < 1.0  (λιγότερο ευαίσθητος σε ρίσκο)
    - conservative monitor: alpha_i > 1.0 (πιο ευαίσθητος σε ρίσκο)

    Parameters
    ----------
    robots : list of RobotState
    targets : list of CellTarget
    risk_profile : dict robot_id -> alpha_i
        Αν κάποιο id λείπει, alpha_i = 1.0 (ουδέτερο).
    base_w_dist : float
        Βασικό βάρος απόστασης.
    base_w_risk : float
        Βασικό βάρος ρίσκου (πολλαπλασιάζεται με alpha_i).

    Returns
    -------
    assignments : list of (robot_id, target_id)
    total_cost : float
    """
    if len(robots) == 0 or len(targets) == 0:
        return [], 0.0

    num_robots = len(robots)
    num_targets = len(targets)
    cost = np.zeros((num_robots, num_targets), dtype=float)

    for i, r in enumerate(robots):
        alpha_i = float(risk_profile.get(r.id, 1.0))
        for j, t in enumerate(targets):
            dist = np.hypot(r.x - t.x, r.y - t.y)
            cost[i, j] = (
                base_w_dist * dist + (alpha_i * base_w_risk) * float(t.risk)
            )

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments: List[Tuple[int, int]] = []
    for i, j in zip(row_ind, col_ind):
        assignments.append((robots[i].id, targets[j].id))

    total_cost = float(cost[row_ind, col_ind].sum())
    return assignments, total_cost


# --------------------------------------------------------------------------- #
# Minimal example (for quick local testing)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    robots = [
        RobotState(id=0, x=0.0, y=0.0),
        RobotState(id=1, x=5.0, y=0.0),
    ]

    targets = [
        CellTarget(id=10, x=1.0, y=1.0, risk=0.2),
        CellTarget(id=11, x=6.0, y=0.5, risk=0.8),
    ]

    print("=== Homogeneous risk weight (όλα τα ρομπότ ίδια) ===")
    assignments, total_cost = assign_tasks_hungarian(
        robots, targets, w_dist=1.0, w_risk=2.0
    )
    print("Assignments:", assignments)
    print("Total cost:", total_cost)

    print("\n=== Heterogeneous risk profiles ===")
    # Robot 0: explorer (λιγότερο ευαίσθητος στο ρίσκο)
    # Robot 1: conservative (πιο ευαίσθητος στο ρίσκο)
    risk_profile = {
        0: 0.5,  # alpha_0
        1: 2.0,  # alpha_1
    }
    assignments2, total_cost2 = assign_with_risk_profiles(
        robots,
        targets,
        risk_profile=risk_profile,
        base_w_dist=1.0,
        base_w_risk=2.0,
    )
    print("Assignments:", assignments2)
    print("Total cost:", total_cost2)
