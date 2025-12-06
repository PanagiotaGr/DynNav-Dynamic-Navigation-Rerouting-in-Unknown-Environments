import numpy as np
from .sensor_models import (
    lidar_camera_residual,
    imu_bias_residual,
    vo_scale_residual,
)


class CalibrationEKF:
    def __init__(self, calib_state):
        self.state = calib_state

    def predict(self, Q=None):
        """
        Simple random walk model for calibration parameters.
        """
        if Q is None:
            Q = np.eye(13) * 1e-6
        self.state.P = self.state.P + Q

    def update(self, residual, H, R):
        """
        Generic EKF update:
        x = x + K r
        P = (I - K H) P
        """
        P = self.state.P
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        self.state.x = self.state.x + K @ residual
        I = np.eye(P.shape[0])
        self.state.P = (I - K @ H) @ P
