from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from sequential_ids import CUSUMDetector, CUSUMConfig, EWMADetector, EWMAConfig, nis_from_innovation


@dataclass
class IDSState:
    nis: float
    ewma: float
    cusum: float
    alarm: bool


class SecurityMonitorCUSUM:
    """
    Wraps sequential detectors for each sensor stream.
    You provide either:
      - nis directly, OR
      - innovation nu and covariance S to compute nis.
    """

    def __init__(
        self,
        sensors: Optional[list] = None,
        ewma_cfg: EWMAConfig = EWMAConfig(lam=0.05),
        cusum_cfg: CUSUMConfig = CUSUMConfig(k=1.5, h=20.0),
    ):
        if sensors is None:
            sensors = ["vo", "wheel"]

        self.detectors: Dict[str, Dict[str, object]] = {}
        for s in sensors:
            self.detectors[s] = {
                "ewma": EWMADetector(ewma_cfg),
                "cusum": CUSUMDetector(cusum_cfg),
            }

    def reset(self, sensor: Optional[str] = None):
        if sensor is None:
            for s in self.detectors:
                self.detectors[s]["ewma"].reset()
                self.detectors[s]["cusum"].reset()
        else:
            self.detectors[sensor]["ewma"].reset()
            self.detectors[sensor]["cusum"].reset()

    def update_from_nis(self, sensor: str, nis: float) -> IDSState:
        ew = self.detectors[sensor]["ewma"].update(nis)
        cu = self.detectors[sensor]["cusum"].update(nis)
        alarm = self.detectors[sensor]["cusum"].is_alarm()
        return IDSState(nis=float(nis), ewma=float(ew), cusum=float(cu), alarm=bool(alarm))

    def update_from_innovation(self, sensor: str, nu: np.ndarray, S: np.ndarray) -> IDSState:
        nis = nis_from_innovation(nu, S)
        return self.update_from_nis(sensor, nis)
