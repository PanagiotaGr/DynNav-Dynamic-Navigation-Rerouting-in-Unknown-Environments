# calibration_state.py
import numpy as np

class CalibrationState:
    """
    State vector for online sensor calibration:
    [LiDAR->Cam rotation (3), translation (3), gyro_bias (3), accel_bias (3), vo_scale (1)]
    Total dimension = 13
    """
    
    def __init__(self):
        self.x = np.zeros(13)
        # initial extrinsic rotation small perturbations
        self.x[0:3] = 0.0       # rotation vector
        self.x[3:6] = 0.0       # translation
        self.x[6:9] = 0.0       # gyro bias
        self.x[9:12] = 0.0      # accel bias
        self.x[12] = 1.0        # VO scale
        
        self.P = np.eye(13) * 0.01   # small initial uncertainty

    def get_rotation_matrix(self):
        """Rodrigues rotation."""
        theta = np.linalg.norm(self.x[:3])
        if theta < 1e-6:
            return np.eye(3)
        r = self.x[:3] / theta
        K = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
