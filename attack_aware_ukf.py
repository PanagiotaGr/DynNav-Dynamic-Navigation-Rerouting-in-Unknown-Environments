import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any


def _ensure_sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _chol_psd(A: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    """
    Cholesky for PSD matrices with jitter fallback.
    """
    A = _ensure_sym(A)
    for k in range(6):
        try:
            return np.linalg.cholesky(A + (10 ** k) * jitter * np.eye(A.shape[0]))
        except np.linalg.LinAlgError:
            continue
    # Last resort: eig fix
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    return np.linalg.cholesky(V @ np.diag(w) @ V.T + jitter * np.eye(A.shape[0]))


def chi2_threshold_approx(df: int, p: float) -> float:
    """
    Approx inverse CDF for chi-square using Wilson-Hilferty.
    Good enough for gating thresholds.
    p in (0,1). Typical: 0.95, 0.99, 0.997.
    """
    # Normal quantile approx (Acklam-ish light version)
    # We'll use scipy-free approximation via inverse error function expansion is messy,
    # so we embed a robust rational approximation for N(0,1) quantile.
    # Reference: Peter John Acklam approximation (implemented compactly).
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    # Coefficients for Acklam approximation
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = np.sqrt(-2*np.log(p))
        z = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p > phigh:
        q = np.sqrt(-2*np.log(1-p))
        z = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    else:
        q = p - 0.5
        r = q*q
        z = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    # Wilson-Hilferty transform for chi-square inverse
    # x ≈ df * (1 - 2/(9df) + z*sqrt(2/(9df)))^3
    df = float(df)
    t = 1.0 - 2.0/(9.0*df) + z*np.sqrt(2.0/(9.0*df))
    return float(df * (t**3))


@dataclass
class SensorConfig:
    name: str
    R_base: np.ndarray
    nis_p: float = 0.99            # gating percentile
    trust_decay: float = 0.15      # how fast trust drops when anomaly happens
    trust_recover: float = 0.02    # how fast trust recovers when normal
    min_trust: float = 0.05
    max_trust: float = 1.0
    dropout_trust: float = 0.08    # if trust below this, skip update
    inflate_max: float = 200.0     # cap for R inflation factor


class AttackAwareUKF:
    """
    UKF with innovation-based anomaly scoring (NIS) and adaptive trust-weighted measurement noise.
    Supports multiple sensors: each sensor has its own h(x) and R, gating and trust state.
    """

    def __init__(
        self,
        n: int,
        f: Callable[[np.ndarray, float], np.ndarray],
        Q: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        n: state dimension
        f: process model f(x, dt)
        Q: process noise covariance
        """
        self.n = n
        self.f = f
        self.Q = Q

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.x = np.zeros(n)
        self.P = np.eye(n)

        self.sensors: Dict[str, Dict[str, Any]] = {}

        # UKF weights
        self.lmbda = (alpha**2) * (n + kappa) - n
        self.gamma = np.sqrt(n + self.lmbda)

        Wm = np.full(2*n + 1, 1.0 / (2.0*(n + self.lmbda)))
        Wc = np.full(2*n + 1, 1.0 / (2.0*(n + self.lmbda)))
        Wm[0] = self.lmbda / (n + self.lmbda)
        Wc[0] = self.lmbda / (n + self.lmbda) + (1 - alpha**2 + beta)
        self.Wm = Wm
        self.Wc = Wc

    def set_state(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.x = x0.astype(float).copy()
        self.P = _ensure_sym(P0.astype(float).copy())

    def add_sensor(self, cfg: SensorConfig, h: Callable[[np.ndarray], np.ndarray]) -> None:
        m = cfg.R_base.shape[0]
        if cfg.R_base.shape[0] != cfg.R_base.shape[1]:
            raise ValueError(f"R_base for {cfg.name} must be square")
        nis_thr = chi2_threshold_approx(m, cfg.nis_p)

        self.sensors[cfg.name] = {
            "cfg": cfg,
            "h": h,
            "trust": cfg.max_trust,
            "nis_thr": nis_thr,
            "last": {}
        }

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        S = _chol_psd(P)
        X = np.zeros((2*self.n + 1, self.n))
        X[0] = x
        for i in range(self.n):
            d = self.gamma * S[:, i]
            X[i + 1] = x + d
            X[self.n + i + 1] = x - d
        return X

    def predict(self, dt: float) -> None:
        X = self._sigma_points(self.x, self.P)
        Xp = np.array([self.f(xi, dt) for xi in X])

        x_pred = np.sum(self.Wm[:, None] * Xp, axis=0)
        P_pred = np.zeros((self.n, self.n))
        for i in range(Xp.shape[0]):
            dx = (Xp[i] - x_pred).reshape(-1, 1)
            P_pred += self.Wc[i] * (dx @ dx.T)
        P_pred += self.Q

        self.x = x_pred
        self.P = _ensure_sym(P_pred)

    def update(self, sensor_name: str, z: np.ndarray) -> Dict[str, float]:
        """
        Perform an attack-aware measurement update for a given sensor.
        Returns telemetry dict: nis, trust, used (0/1), inflation.
        """
        if sensor_name not in self.sensors:
            raise KeyError(f"Unknown sensor: {sensor_name}")

        entry = self.sensors[sensor_name]
        cfg: SensorConfig = entry["cfg"]
        h = entry["h"]

        trust = float(entry["trust"])
        if trust < cfg.dropout_trust:
            # soft dropout: skip measurement update
            entry["last"] = {"nis": np.nan, "inflation": np.nan, "used": 0.0, "trust": trust}
            return entry["last"]

        # Sigma points
        X = self._sigma_points(self.x, self.P)
        Zsig = np.array([h(xi) for xi in X])  # (2n+1, m)

        z_hat = np.sum(self.Wm[:, None] * Zsig, axis=0)

        # Innovation covariance S and cross-covariance Pxz
        m = z_hat.shape[0]
        S = np.zeros((m, m))
        Pxz = np.zeros((self.n, m))

        for i in range(2*self.n + 1):
            dz = (Zsig[i] - z_hat).reshape(-1, 1)
            dx = (X[i] - self.x).reshape(-1, 1)
            S += self.Wc[i] * (dz @ dz.T)
            Pxz += self.Wc[i] * (dx @ dz.T)

        # Adaptive noise: inflate R based on trust (lower trust => higher R)
        # Option A: R_eff = R / trust  (trust in (0,1])
        # We'll cap inflation factor to keep stability.
        inflation = min(cfg.inflate_max, 1.0 / max(trust, cfg.min_trust))
        R_eff = inflation * cfg.R_base
        S = _ensure_sym(S + R_eff)

        # NIS computation
        nu = (z - z_hat).reshape(-1, 1)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        nis = float((nu.T @ Sinv @ nu).squeeze())

        # Trust update: if anomaly, decay; else recover
        # We use a smooth anomaly score: ratio = nis / thr
        ratio = nis / float(entry["nis_thr"])
        if ratio > 1.0:
            # anomaly: decay trust more when ratio larger
            trust = trust * np.exp(-cfg.trust_decay * (ratio - 1.0))
        else:
            # normal: recover slowly
            trust = trust + cfg.trust_recover * (cfg.max_trust - trust)

        trust = float(np.clip(trust, cfg.min_trust, cfg.max_trust))
        entry["trust"] = trust

        # If after update trust is extremely low, you can decide to skip this update.
        # But we already used it for scoring; for robustness, we can still apply update using inflated R.
        # Kalman gain
        K = Pxz @ Sinv
        x_new = self.x + (K @ nu).reshape(-1)
        P_new = self.P - K @ S @ K.T

        self.x = x_new
        self.P = _ensure_sym(P_new)

        entry["last"] = {"nis": nis, "inflation": inflation, "used": 1.0, "trust": trust}
        return entry["last"]

    def get_trust(self, sensor_name: str) -> float:
        return float(self.sensors[sensor_name]["trust"])

    def get_last(self, sensor_name: str) -> Dict[str, float]:
        return dict(self.sensors[sensor_name].get("last", {}))


# --------------------------
# Example models (you θα τα προσαρμόσεις στα δικά σου)
# --------------------------

def f_constant_velocity(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Example 2D constant velocity:
    x = [px, py, vx, vy]
    """
    px, py, vx, vy = x
    return np.array([px + vx*dt, py + vy*dt, vx, vy], dtype=float)


def h_position(x: np.ndarray) -> np.ndarray:
    """
    Measures position only: z = [px, py]
    """
    return x[:2].copy()


def demo_run():
    np.random.seed(0)

    # State: [px, py, vx, vy]
    n = 4
    dt = 0.1

    Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])

    ukf = AttackAwareUKF(n=n, f=f_constant_velocity, Q=Q)
    ukf.set_state(np.array([0, 0, 1, 0.5], float), np.diag([0.1, 0.1, 0.2, 0.2]))

    # Two sensors measuring position with different noise
    vo_cfg = SensorConfig(
        name="vo",
        R_base=np.diag([0.05**2, 0.05**2]),
        nis_p=0.99,
        trust_decay=0.25,
        trust_recover=0.02,
        dropout_trust=0.06
    )
    wheel_cfg = SensorConfig(
        name="wheel",
        R_base=np.diag([0.10**2, 0.10**2]),
        nis_p=0.99,
        trust_decay=0.12,
        trust_recover=0.03,
        dropout_trust=0.06
    )

    ukf.add_sensor(vo_cfg, h_position)
    ukf.add_sensor(wheel_cfg, h_position)

    # Simulate with an attack on VO: add slowly growing bias after t=50
    x_true = np.array([0, 0, 1, 0.5], float)

    logs = []
    for t in range(200):
        # true evolution
        x_true = f_constant_velocity(x_true, dt)

        # predict
        ukf.predict(dt)

        # measurements
        z_vo = x_true[:2] + np.random.randn(2) * 0.05
        z_w  = x_true[:2] + np.random.randn(2) * 0.10

        if t > 50:
            bias = np.array([0.002*(t-50), -0.0015*(t-50)])
            z_vo = z_vo + bias  # stealthy-ish bias

        # updates
        tel_vo = ukf.update("vo", z_vo)
        tel_w  = ukf.update("wheel", z_w)

        logs.append([t, *ukf.x[:2], tel_vo["nis"], tel_vo["trust"], tel_vo["inflation"],
                     tel_w["nis"], tel_w["trust"], tel_w["inflation"]])

    logs = np.array(logs, float)
    print("Done. Final position estimate:", ukf.x[:2])
    print("Final trusts: vo=%.3f wheel=%.3f" % (ukf.get_trust("vo"), ukf.get_trust("wheel")))

    # Save CSV for plotting in your existing plot scripts
    header = "t,est_px,est_py,vo_nis,vo_trust,vo_infl,w_nis,w_trust,w_infl"
    np.savetxt("attack_aware_ukf_demo_log.csv", logs, delimiter=",", header=header, comments="")

if __name__ == "__main__":
    demo_run()
