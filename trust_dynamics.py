# trust_dynamics.py
#
# Human–Robot Trust Dynamics model.
#
# Στόχος:
# - Να έχουμε μια ενιαία αναπαράσταση για:
#     * self_trust του robot (π.χ. από OOD, drift, calibration κ.λπ.)
#     * human_trust_in_robot (εκτίμηση από το robot: πόσο μας εμπιστεύεται ο άνθρωπος)
# - Να ενημερώνουμε αυτές τις ποσότητες όταν συμβαίνουν events:
#     * "success" (παντού πήγε καλά)
#     * "near_miss" (οριακά, αλλά χωρίς failure)
#     * "failure" (collision, έντονο replan, manual takeover, κ.λπ.)
# - Να παράγουμε policy knobs:
#     * λ_robot (risk weight του robot)
#     * human_influence_scale (πόσο βαρύτητα έχουν οι human preferences)
#     * safe_mode flag όταν self_trust ή human_trust_in_robot είναι χαμηλά.
#
# Τα νούμερα είναι normalized σε [0, 1] για απλότητα.


from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict


class TrustEventType(Enum):
    SUCCESS = auto()
    NEAR_MISS = auto()
    FAILURE = auto()
    HUMAN_OVERRIDE = auto()  # π.χ. ο άνθρωπος παίρνει χειροκίνητο έλεγχο
    HUMAN_APPROVAL = auto()  # π.χ. ο άνθρωπος επιβεβαιώνει τη διαδρομή


@dataclass
class TrustState:
    """
    Κατάσταση εμπιστοσύνης στο σύστημα.

    - self_trust_robot: internal confidence του robot (0 = καθόλου, 1 = πλήρης)
    - human_trust_in_robot: εκτίμηση του robot για το πόσο το εμπιστεύεται ο άνθρωπος
    """
    self_trust_robot: float = 0.7
    human_trust_in_robot: float = 0.7

    def clip(self):
        """Κρατάμε τις τιμές μέσα στο [0, 1]."""
        self.self_trust_robot = max(0.0, min(1.0, self.self_trust_robot))
        self.human_trust_in_robot = max(0.0, min(1.0, self.human_trust_in_robot))


@dataclass
class TrustConfig:
    """
    Ρυθμίσεις για το πόσο δυνατά αντιδρά η εμπιστοσύνη σε events.
    """
    # Πόσο αλλάζει η self-trust του robot
    delta_self_trust: Dict[TrustEventType, float] = field(default_factory=lambda: {
        TrustEventType.SUCCESS:  +0.02,
        TrustEventType.NEAR_MISS: -0.03,
        TrustEventType.FAILURE:  -0.10,
        TrustEventType.HUMAN_OVERRIDE: -0.05,
        TrustEventType.HUMAN_APPROVAL: +0.03,
    })

    # Πόσο αλλάζει η human_trust_in_robot (εκτίμηση)
    delta_human_trust: Dict[TrustEventType, float] = field(default_factory=lambda: {
        TrustEventType.SUCCESS:  +0.03,
        TrustEventType.NEAR_MISS: -0.05,
        TrustEventType.FAILURE:  -0.15,
        TrustEventType.HUMAN_OVERRIDE: -0.08,
        TrustEventType.HUMAN_APPROVAL: +0.05,
    })

    # Mapping trust → policy knobs
    base_lambda_robot: float = 1.0
    min_lambda_robot_factor: float = 0.5   # όταν η self_trust είναι πολύ χαμηλή
    max_lambda_robot_factor: float = 2.0   # όταν η self_trust είναι πολύ υψηλή

    # Human influence (π.χ. πόσο δυνατά μετράνε οι human preferences)
    min_human_influence: float = 0.3
    max_human_influence: float = 1.5

    # Safe mode thresholds
    safe_mode_self_trust_threshold: float = 0.3
    safe_mode_human_trust_threshold: float = 0.3


class TrustDynamics:
    """
    Μοντέλο δυναμικής εμπιστοσύνης.

    - Διατηρεί TrustState
    - Ενημερώνεται με events (success, failure, near_miss, human_override, ...)
    - Παράγει:
        * lambda_robot (risk weight του robot)
        * human_influence_scale (για HumanRiskPolicy, HumanAwarePlannerWrapper)
        * safe_mode flag
    """

    def __init__(self, config: TrustConfig | None = None, initial_state: TrustState | None = None):
        self.config = config if config is not None else TrustConfig()
        self.state = initial_state if initial_state is not None else TrustState()
        self.state.clip()

    # === Update logic ===

    def update_trust(self, event_type: TrustEventType) -> TrustState:
        """
        Ενημέρωση εμπιστοσύνης με βάση ένα event.
        """
        d_self = self.config.delta_self_trust.get(event_type, 0.0)
        d_human = self.config.delta_human_trust.get(event_type, 0.0)

        self.state.self_trust_robot += d_self
        self.state.human_trust_in_robot += d_human
        self.state.clip()
        return self.state

    # === Mapping trust → control parameters ===

    def compute_lambda_robot(self) -> float:
        """
        Υπολογισμός lambda_robot με βάση τη self-trust του robot.

        - Αν self_trust_robot ~ 0  → factor ~ max_lambda_robot_factor (πιο safe)
        - Αν self_trust_robot ~ 1  → factor ~ min_lambda_robot_factor (πιο aggressive)

        Ουσιαστικά:
            factor = interp(self_trust_robot from [0,1] to [max_factor, min_factor])
        """
        s = self.state.self_trust_robot
        c = self.config

        # ανάποδο interpolation: low trust -> high λ, high trust -> low λ
        factor = c.max_lambda_robot_factor - s * (c.max_lambda_robot_factor - c.min_lambda_robot_factor)
        lambda_robot = c.base_lambda_robot * factor
        return lambda_robot

    def compute_human_influence_scale(self) -> float:
        """
        Πόσο δυνατά επηρεάζουν οι human preferences.

        - Αν human_trust_in_robot είναι χαμηλή: μικρό human influence (ο άνθρωπος δεν μας εμπιστεύεται, πάμε πιο conservative & εξηγήσιμα).
        - Αν human_trust_in_robot είναι υψηλή: μεγαλύτερο human influence (δεχόμαστε πιο δυνατές προτιμήσεις του).
        """
        h = self.state.human_trust_in_robot
        c = self.config
        return c.min_human_influence + h * (c.max_human_influence - c.min_human_influence)

    def in_safe_mode(self) -> bool:
        """
        Safe mode αν είτε:
        - η self_trust του robot είναι πολύ χαμηλή, ή
        - η εκτιμώμενη human_trust_in_robot είναι πολύ χαμηλή.
        """
        c = self.config
        return (
            self.state.self_trust_robot < c.safe_mode_self_trust_threshold
            or self.state.human_trust_in_robot < c.safe_mode_human_trust_threshold
        )

    def export_policy_knobs(self) -> dict:
        """
        Convenience: Επιστρέφει όλα τα policy-related μεγέθη σε ένα dict, για logging / experiments.
        """
        lambda_robot = self.compute_lambda_robot()
        human_influence_scale = self.compute_human_influence_scale()
        return {
            "self_trust_robot": self.state.self_trust_robot,
            "human_trust_in_robot": self.state.human_trust_in_robot,
            "lambda_robot": lambda_robot,
            "human_influence_scale": human_influence_scale,
            "safe_mode": self.in_safe_mode(),
        }
