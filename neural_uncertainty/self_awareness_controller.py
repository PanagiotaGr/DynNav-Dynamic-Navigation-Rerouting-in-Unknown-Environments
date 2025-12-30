# ================================================================
# Self-Awareness Controller: Self-Trust Index & Navigation Modes
# ================================================================

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


class NavigationMode(Enum):
    """
    Διακριτές καταστάσεις "συμπεριφοράς" του ρομπότ
    ανάλογα με το πόσο εμπιστεύεται τον εαυτό του.
    """
    NORMAL = auto()      # Κανονική πλοήγηση
    CAUTIOUS = auto()    # Προσεκτική πλοήγηση (π.χ. πιο αργά, περισσότερη σάρωση)
    SAFE_STOP = auto()   # Ασφαλές σταμάτημα / fallback


@dataclass
class SelfTrustConfig:
    """
    Ρυθμίσεις για τον υπολογισμό Self-Trust Index.
    """
    base_mse: float          # baseline MSE (από offline / baseline run)
    base_var: float          # baseline epistemic variance
    alpha_err: float = 1.0   # βάρος σφάλματος
    alpha_var: float = 1.0   # βάρος variance
    eps: float = 1e-8        # για αποφυγή division by zero

    # Κατώφλια για τα modes
    normal_threshold: float = 0.7
    cautious_threshold: float = 0.4


class SelfAwarenessController:
    """
    Υπολογίζει Self-Trust Index S(t) και επιλέγει navigation mode.

    S ~ 1 → πολύ εμπιστοσύνη στο μοντέλο
    S ~ 0 → καθόλου εμπιστοσύνη
    """

    def __init__(self, config: SelfTrustConfig) -> None:
        self.config = config

    def compute_self_trust(self, error_sq: float, var: float) -> float:
        """
        Υπολογισμός Self-Trust Index με βάση το normalized σφάλμα και variance.

        error_sq : (y_hat - y)^2
        var      : epistemic variance για την τρέχουσα πρόβλεψη
        """
        # Κανονικοποίηση ως προς baseline επίπεδα
        e_norm = error_sq / (self.config.base_mse + self.config.eps)
        v_norm = var / (self.config.base_var + self.config.eps)

        # Συνολική "penalty"
        penalty = self.config.alpha_err * e_norm + self.config.alpha_var * v_norm

        # Χαρτογράφηση σε [0, 1]
        S = 1.0 / (1.0 + penalty)
        return float(S)

    def choose_mode(self, self_trust: float) -> NavigationMode:
        """
        Επιλογή navigation mode με βάση το Self-Trust.

        S >= normal_threshold       → NORMAL
        cautious_threshold <= S < normal_threshold → CAUTIOUS
        S < cautious_threshold      → SAFE_STOP
        """
        if self_trust >= self.config.normal_threshold:
            return NavigationMode.NORMAL
        elif self_trust >= self.config.cautious_threshold:
            return NavigationMode.CAUTIOUS
        else:
            return NavigationMode.SAFE_STOP

    def evaluate_step(self, error_sq: float, var: float) -> Tuple[float, NavigationMode]:
        """
        Convenience μέθοδος: δέχεται error & variance και επιστρέφει (S, mode).
        """
        S = self.compute_self_trust(error_sq, var)
        mode = self.choose_mode(S)
        return S, mode
