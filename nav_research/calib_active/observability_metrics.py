# observability_metrics.py
import numpy as np

def observability_score(H):
    """
    Compute observability metric using determinant of HᵀH.
    Large determinant → system more observable.
    """
    M = H.T @ H
    return np.linalg.det(M + 1e-6 * np.eye(M.shape[0]))
