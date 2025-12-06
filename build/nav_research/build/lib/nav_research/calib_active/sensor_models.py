# sensor_models.py
import numpy as np

def lidar_camera_residual(lidar_pos, cam_pos, calib_state):
    """
    Compute extrinsic calibration residual:
    R * p_lidar + t - p_cam
    """
    R = calib_state.get_rotation_matrix()
    t = calib_state.x[3:6]

    pred_cam = R @ lidar_pos + t
    residual = pred_cam - cam_pos
    return residual


def imu_bias_residual(imu_measurement, predicted, calib_state):
    """
    IMU bias estimation residual.
    """
    gyro_bias = calib_state.x[6:9]
    accel_bias = calib_state.x[9:12]

    gyro_res = imu_measurement[:3] - predicted[:3] - gyro_bias
    accel_res = imu_measurement[3:6] - predicted[3:6] - accel_bias
    return np.hstack([gyro_res, accel_res])


def vo_scale_residual(vo_distance, true_distance, calib_state):
    scale = calib_state.x[12]
    return scale * vo_distance - true_distance
