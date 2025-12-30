# ================================================================
# Risk Explanation Utility for Path Selection
# ================================================================

from dataclasses import dataclass
from typing import List


@dataclass
class PathStats:
    name: str
    length: float              # συνολικό μήκος διαδρομής (π.χ. μέτρα)
    drift_exposure: float      # π.χ. μέσο drift ή integrated drift
    uncertainty_exposure: float  # π.χ. μέση ή συνολική epistemic variance


def explain_preference(
    path_a: PathStats,
    path_b: PathStats,
    lambda_risk: float = 1.0,
) -> str:
    """
    Παράγει λεκτική εξήγηση γιατί προτιμάμε τη μία διαδρομή έναντι της άλλης,
    με βάση trade-off μεταξύ μήκους και risk (drift + uncertainty).
    """

    # "κόστος" κάθε διαδρομής
    cost_a = path_a.length + lambda_risk * (
        path_a.drift_exposure + path_a.uncertainty_exposure
    )
    cost_b = path_b.length + lambda_risk * (
        path_b.drift_exposure + path_b.uncertainty_exposure
    )

    # Επιλογή προτιμώμενης διαδρομής
    prefer_b = cost_b < cost_a
    chosen = path_b if prefer_b else path_a
    other = path_a if prefer_b else path_b

    # -------------------------
    # Διαφορές για explainability
    # -------------------------
    drift_diff = other.drift_exposure - chosen.drift_exposure
    unc_diff = other.uncertainty_exposure - chosen.uncertainty_exposure
    length_diff = chosen.length - other.length

    def pct(delta, base):
        if base == 0:
            return 0.0
        return 100.0 * abs(delta) / base

    max_drift = max(path_a.drift_exposure, path_b.drift_exposure)
    max_unc = max(path_a.uncertainty_exposure, path_b.uncertainty_exposure)
    max_length = max(path_a.length, path_b.length)

    drift_pct = pct(drift_diff, max_drift)
    unc_pct = pct(unc_diff, max_unc)
    length_pct = pct(length_diff, max_length)

    # -------------------------
    # Σωστή λογική κατεύθυνσης
    # -------------------------
    if chosen.drift_exposure < other.drift_exposure:
        drift_dir = "μειώνει"
    elif chosen.drift_exposure > other.drift_exposure:
        drift_dir = "αυξάνει"
    else:
        drift_dir = "δεν αλλάζει"

    if chosen.uncertainty_exposure < other.uncertainty_exposure:
        unc_dir = "μειώνει"
    elif chosen.uncertainty_exposure > other.uncertainty_exposure:
        unc_dir = "αυξάνει"
    else:
        unc_dir = "δεν αλλάζει"

    # -------------------------
    # Narrative
    # -------------------------
    if length_diff > 0:
        length_part = f"με τίμημα περίπου {length_pct:.1f}% μεγαλύτερο μήκος διαδρομής"
    elif length_diff < 0:
        length_part = f"προσφέροντας και περίπου {length_pct:.1f}% μικρότερο μήκος διαδρομής"
    else:
        length_part = "χωρίς ουσιαστική διαφορά στο μήκος της διαδρομής"

    drift_part = (
        f"{drift_dir} την έκθεση σε drift περίπου κατά {drift_pct:.1f}%"
        if max_drift > 0 else
        "έχει παρόμοια έκθεση σε drift"
    )

    unc_part = (
        f"και {unc_dir} την αβεβαιότητα περίπου κατά {unc_pct:.1f}%"
        if max_unc > 0 else
        "και παρόμοια αβεβαιότητα"
    )

    explanation = (
        f"Επιλέγω τη διαδρομή {chosen.name} αντί για τη {other.name} γιατί "
        f"{drift_part} {unc_part}, "
        f"{length_part} (λ={lambda_risk:.2f})."
    )

    return explanation


def choose_best_path(paths: List[PathStats], lambda_risk: float = 1.0) -> PathStats:
    """
    Επιλογή "βέλτιστης" διαδρομής με βάση:
        cost = length + λ * (drift_exposure + uncertainty_exposure)
    """
    best = None
    best_cost = float("inf")

    for p in paths:
        cost = p.length + lambda_risk * (p.drift_exposure + p.uncertainty_exposure)
        if cost < best_cost:
            best_cost = cost
            best = p

    return best
