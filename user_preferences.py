from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HumanPreference:
    """
    Continuous risk preference + extra semantic constraints.

    risk_preference in [0,1]:
        0.0 -> πολύ ασφαλής / risk-averse
        0.5 -> balanced
        1.0 -> αποδέχεται ρίσκο για speed
    """
    risk_preference: float
    avoid_low_feature_areas: bool = False
    avoid_dark_areas: bool = False
    prefer_well_mapped_areas: bool = False
    meta: Optional[Dict] = None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parse_human_preference(text: str) -> HumanPreference:
    """
    Πολύ απλός rule-based parser για Ελληνικά + Αγγλικά.
    ΔΕΝ χρειάζεται έξτρα libraries.
    """
    if text is None:
        text = ""
    t = text.lower()

    # Βάση: balanced
    h = 0.5
    avoid_low_feature = False
    avoid_dark = False
    prefer_well_mapped = False

    # --- Risk preference rules ---

    # πιο ασφαλής / more safe
    if "πιο ασφαλής" in t or "more safe" in t or "safer" in t:
        h = 0.1
    if "αργή" in t or "αργά" in t or "slow" in t:
        h = min(h, 0.2)

    # γρήγορα / fast
    if "φτάσε γρήγορα" in t or "γρήγορα" in t or "fast" in t or "quick" in t:
        h = 0.8
    if "αποδέχομαι ρίσκο" in t or "accept risk" in t or "high risk" in t:
        h = 0.9

    # explicit labels
    if "risk-averse" in t or "χωρίς ρίσκο" in t or "no risk" in t:
        h = 0.0
    if "aggressive" in t or "τολμηρή" in t:
        h = 1.0

    # --- Environment constraints ---

    # σκοτεινές / dark περιοχές
    if "σκοτειν" in t or "dark" in t:
        avoid_dark = True
        avoid_low_feature = True  # συνήθως πάνε μαζί

    if "low-feature" in t or "χαμηλό feature" in t or "λίγα features" in t:
        avoid_low_feature = True

    if "καλά χαρτογραφημένες" in t or "well-mapped" in t:
        prefer_well_mapped = True

    return HumanPreference(
        risk_preference=clamp01(h),
        avoid_low_feature_areas=avoid_low_feature,
        avoid_dark_areas=avoid_dark,
        prefer_well_mapped_areas=prefer_well_mapped,
        meta={},
    )
