# failure_taxonomy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FailureType:
    code: str
    description: str


# Core categories
FAIL_START_VIOLATES = FailureType(
    "START_VIOLATES",
    "Start state violates constraint (I(start) > tau) or invalid start.",
)
FAIL_GOAL_VIOLATES = FailureType(
    "GOAL_VIOLATES",
    "Goal state violates constraint (I(goal) > tau) or invalid goal.",
)
FAIL_NO_PATH_CONSTRAINT = FailureType(
    "NO_PATH_CONSTRAINT",
    "No path exists under the irreversibility constraint (constraint-induced disconnection).",
)
FAIL_NO_PATH_GEOMETRY = FailureType(
    "NO_PATH_GEOMETRY",
    "No path exists in the free-space geometry (true disconnection/obstacles).",
)
FAIL_OTHER = FailureType(
    "OTHER",
    "Other / unexpected failure.",
)


def classify_failure_reason(reason: str) -> FailureType:
    r = (reason or "").lower()

    # match the reasons we already print in your planners
    if "start violates" in r or ("start" in r and "violates" in r):
        return FAIL_START_VIOLATES
    if "goal violates" in r or ("goal" in r and "violates" in r):
        return FAIL_GOAL_VIOLATES
    if "no path under irreversibility constraint" in r or ("no path" in r and "irreversibility" in r):
        return FAIL_NO_PATH_CONSTRAINT
    if "no path" in r:
        return FAIL_NO_PATH_GEOMETRY
    return FAIL_OTHER
