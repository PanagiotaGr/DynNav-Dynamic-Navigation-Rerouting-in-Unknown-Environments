import numpy as np

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

class ConnectivityMap:
    """
    Simple connectivity model over a 2D grid.
    Produces:
      - C(x) in [0,1] : normalized connectivity quality
      - P_loss(x) in [0,1] : packet loss probability
    """

    def __init__(
        self,
        grid_shape,
        ap_xy=(5.0, 5.0),
        cell_size=1.0,
        snr0_db=30.0,
        pathloss_exp=2.2,
        shadow_sigma_db=2.0,
        obstacle_penalty_db=8.0,
        rng_seed=0,
    ):
        self.H, self.W = grid_shape
        self.ap_xy = np.array(ap_xy, dtype=float)
        self.cell_size = float(cell_size)
        self.snr0_db = float(snr0_db)
        self.pathloss_exp = float(pathloss_exp)
        self.shadow_sigma_db = float(shadow_sigma_db)
        self.obstacle_penalty_db = float(obstacle_penalty_db)
        self.rng = np.random.default_rng(rng_seed)

    def _grid_xy(self):
        ys = np.arange(self.H) * self.cell_size
        xs = np.arange(self.W) * self.cell_size
        X, Y = np.meshgrid(xs, ys)
        return X, Y

    def build(self, occupancy: np.ndarray, add_shadowing=True):
        """
        occupancy: (H,W) with 1=obstacle, 0=free
        """
        occ = occupancy.astype(float)
        X, Y = self._grid_xy()
        dx = X - self.ap_xy[0]
        dy = Y - self.ap_xy[1]
        d = np.sqrt(dx * dx + dy * dy) + 1e-6

        # Log-distance style path loss on SNR (relative model)
        # snr_db = snr0 - 10*n*log10(d/d0)
        d0 = 1.0
        snr_db = self.snr0_db - 10.0 * self.pathloss_exp * np.log10(d / d0)

        if add_shadowing:
            snr_db = snr_db + self.rng.normal(0.0, self.shadow_sigma_db, size=snr_db.shape)

        # Obstacles reduce SNR
        snr_db = snr_db - self.obstacle_penalty_db * occ

        # Normalize SNR -> C in [0,1] using a smooth mapping
        # Choose thresholds so that -5 dB is "bad", 15 dB is "good"
        snr_min, snr_max = -5.0, 15.0
        C = clamp01((snr_db - snr_min) / (snr_max - snr_min))

        # Packet loss probability: logistic in SNR
        # around 5 dB midpoint; steeper slope -> sharper transition
        mid_db = 5.0
        slope = 0.6
        P_loss = clamp01(1.0 - sigmoid(slope * (snr_db - mid_db)))

        return snr_db, C, P_loss
