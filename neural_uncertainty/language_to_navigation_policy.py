# ================================================================
# Language-Conditioned Navigation Policy Mapping
# ================================================================

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class LanguagePolicyResult:
    """Αποτέλεσμα language policy mapping."""
    lambda_risk: Optional[float] = None        # αλλαγή risk weight
    speed_scale: Optional[float] = None        # scaling ταχύτητας (1.0 = unchanged)
    avoid_right_wall: bool = False             # directional constraint
    avoid_left_wall: bool = False
    increase_exploration: bool = False         # NBV emphasis
    explanation: str = ""                      # human readable summary


class LanguageToNavigationPolicy:
    """
    Πολύ απλό rule-based mapping φυσικής γλώσσας → navigation policy parameters.
    Δεν είναι NLP model — είναι interpretable mapping module.
    """

    def __init__(self) -> None:
        # keywords mapping
        self.rules: Dict[str, Dict] = {
            "γρήγορα": dict(lambda_risk=0.5, speed_scale=1.3,
                            explanation="Δίνεται προτεραιότητα στην ταχύτητα έναντι του ρίσκου."),
            "αργά": dict(lambda_risk=2.0, speed_scale=0.6,
                         explanation="Δίνεται προτεραιότητα στην ασφάλεια έναντι της ταχύτητας."),
            "πρόσεχε": dict(lambda_risk=2.5, speed_scale=0.7,
                            explanation="Αυξάνεται σημαντικά η σημασία του ρίσκου."),
            "επικίνδυνο": dict(lambda_risk=3.0, speed_scale=0.6,
                               explanation="Φαίνεται επικίνδυνο — γίνεται πολύ συντηρητική πολιτική."),
            "δεξιά": dict(avoid_right_wall=True,
                          explanation="Αποφυγή προσέγγισης δεξιάς πλευράς."),
            "αριστερά": dict(avoid_left_wall=True,
                             explanation="Αποφυγή προσέγγισης αριστερής πλευράς."),
            "εξερεύνησε": dict(increase_exploration=True,
                               explanation="Ενθαρρύνεται ενεργή εξερεύνηση."),
        }

    def interpret(self, text: str) -> LanguagePolicyResult:
        """
        Παίρνει input φυσικής γλώσσας και ενεργοποιεί πολιτικές.
        """
        text = text.lower()

        result = LanguagePolicyResult(
            lambda_risk=None,
            speed_scale=None,
            avoid_right_wall=False,
            avoid_left_wall=False,
            increase_exploration=False,
            explanation="",
        )

        applied_explanations = []

        for keyword, effects in self.rules.items():
            if keyword in text:
                # apply effects
                if "lambda_risk" in effects:
                    result.lambda_risk = effects["lambda_risk"]
                if "speed_scale" in effects:
                    result.speed_scale = effects["speed_scale"]
                if "avoid_right_wall" in effects:
                    result.avoid_right_wall = True
                if "avoid_left_wall" in effects:
                    result.avoid_left_wall = True
                if "increase_exploration" in effects:
                    result.increase_exploration = True

                applied_explanations.append(effects["explanation"])

        if applied_explanations:
            result.explanation = " ".join(applied_explanations)
        else:
            result.explanation = "Δεν ανιχνεύτηκαν συγκεκριμένες navigation εντολές."

        return result
