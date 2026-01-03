from dataclasses import dataclass


@dataclass
class MitigationConfig:
    vo_trust_min: float = 0.05
    lambda_boost: float = 0.35          # how much to increase risk weight
    safe_mode_steps: int = 50           # how long to stay conservative after alarm
    cooldown_recover: float = 0.01      # trust recovery per step (slow)


class IDSMitigationPolicy:
    """
    Keeps a short-term 'safe mode' latch after alarm.
    You can feed its outputs into your existing trust/self_trust_manager + risk policy.
    """

    def __init__(self, cfg: MitigationConfig = MitigationConfig()):
        self.cfg = cfg
        self.safe_countdown = 0
        self.vo_trust_override = 1.0

    def step(self, alarm_vo: bool) -> dict:
        if alarm_vo:
            self.safe_countdown = self.cfg.safe_mode_steps
            self.vo_trust_override = self.cfg.vo_trust_min
        else:
            if self.safe_countdown > 0:
                self.safe_countdown -= 1
            # recover trust slowly when not alarming
            self.vo_trust_override = min(1.0, self.vo_trust_override + self.cfg.cooldown_recover)

        safe_mode = self.safe_countdown > 0
        lambda_add = self.cfg.lambda_boost if safe_mode else 0.0

        return {
            "safe_mode": safe_mode,
            "vo_trust": self.vo_trust_override,
            "lambda_add": lambda_add,
            "countdown": self.safe_countdown
        }
