import numpy as np
from dataclasses import dataclass


@dataclass
class EWMAConfig:
    lam: float = 0.05  # smoothing in (0,1]


@dataclass
class CUSUMConfig:
    k: float = 1.5     # reference value (drift); larger -> less sensitive
    h: float = 20.0    # threshold on cumulative sum (alarm when g>=h)


class EWMADetector:
    """
    EWMA over a scalar score stream (e.g., NIS).
    Alarm if ewma >= threshold externally chosen.
    """
    def __init__(self, cfg: EWMAConfig):
        self.cfg = cfg
        self.s = None

    def reset(self):
        self.s = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.s is None:
            self.s = x
        else:
            lam = self.cfg.lam
            self.s = lam * x + (1.0 - lam) * self.s
        return float(self.s)


class CUSUMDetector:
    """
    One-sided CUSUM on scalar score stream (e.g., NIS).
    g_t = max(0, g_{t-1} + (x_t - k))
    Alarm if g_t >= h.
    """
    def __init__(self, cfg: CUSUMConfig):
        self.cfg = cfg
        self.g = 0.0

    def reset(self):
        self.g = 0.0

    def update(self, x: float) -> float:
        x = float(x)
        self.g = max(0.0, self.g + (x - self.cfg.k))
        return float(self.g)

    def is_alarm(self) -> bool:
        return bool(self.g >= self.cfg.h)


def nis_from_innovation(nu: np.ndarray, S: np.ndarray) -> float:
    """
    NIS = nu^T S^{-1} nu
    nu: (m,) or (m,1)
    S:  (m,m)
    """
    nu = np.asarray(nu, dtype=float).reshape(-1, 1)
    S = np.asarray(S, dtype=float)
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
    return float((nu.T @ Sinv @ nu).squeeze())
