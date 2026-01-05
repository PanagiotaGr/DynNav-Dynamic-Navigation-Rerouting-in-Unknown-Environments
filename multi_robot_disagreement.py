from __future__ import annotations
import numpy as np
from typing import Any, Iterable, Optional


def disagreement_maxmin(values: Iterable[float]) -> float:
    """Max-min spread (robust)."""
    v = np.asarray(list(values), dtype=float)
    if v.size < 2:
        return 0.0
    return float(np.max(v) - np.min(v))


def disagreement_variance(values: Iterable[float]) -> float:
    """Variance across robots."""
    v = np.asarray(list(values), dtype=float)
    if v.size < 2:
        return 0.0
    return float(np.var(v))


def _get_first_attr(obj: Any, candidates: list[str]) -> Optional[float]:
    """Try to read a numeric attribute or dict key from obj."""
    # dict-like
    if isinstance(obj, dict):
        for k in candidates:
            if k in obj:
                try:
                    return float(obj[k])
                except Exception:
                    pass

    # attribute-like
    for k in candidates:
        if hasattr(obj, k):
            try:
                return float(getattr(obj, k))
            except Exception:
                pass
    return None


def extract_robot_risk(robot: Any) -> Optional[float]:
    """
    Best-effort: tries common fields used in this repo / robotics codebases.
    Returns None if not found.
    """
    candidates = [
        # common naming
        "risk", "current_risk", "risk_est", "risk_value",
        "R", "R_t", "risk_t",
        # trust/risk layers sometimes store "lambda_eff" or "tau" but we want risk
        "belief_risk", "path_risk", "max_risk",
    ]
    return _get_first_attr(robot, candidates)


def extract_robot_trust(robot: Any) -> Optional[float]:
    """Optional: if you want to couple disagreement with trust later."""
    candidates = ["trust", "vo_trust", "trust_value", "trust_eff", "current_trust"]
    return _get_first_attr(robot, candidates)
