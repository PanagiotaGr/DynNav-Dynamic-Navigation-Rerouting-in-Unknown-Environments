"""
uncertainty_heuristic_net.py

Neural network heuristic that outputs both mean and uncertainty (std).

Architecture:
  - Shared backbone: 3 FC layers with ReLU
  - Head 1 (mean):  linear output  -> h_mean >= 0 (via Softplus)
  - Head 2 (std):   linear output  -> h_std  >  0 (via Softplus)

Training loss: Gaussian Negative Log-Likelihood
  L = log(sigma) + (y - mu)^2 / (2 * sigma^2)

This jointly trains both heads — the network learns not only where the
optimal cost is, but also where its own predictions are uncertain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class UncertaintyHeuristicNet(nn.Module):
    """
    Two-head network: (h_mean, h_std).

    Parameters
    ----------
    input_dim : int
        Dimension of the feature vector (default 11, same as existing module).
    hidden_dim : int
        Width of each hidden layer.
    n_hidden : int
        Number of hidden layers in the shared backbone.
    min_std : float
        Floor for h_std to avoid numerical issues.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        n_hidden: int = 3,
        min_std: float = 1e-3,
    ):
        super().__init__()

        self.min_std = min_std

        # --- shared backbone ---
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        # --- output heads ---
        self.head_mean = nn.Linear(hidden_dim, 1)
        self.head_log_std = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, input_dim) or (input_dim,)

        Returns
        -------
        h_mean : (batch, 1)  predicted heuristic (non-negative via Softplus)
        h_std  : (batch, 1)  predicted uncertainty (positive via Softplus + floor)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        feat = self.backbone(x)

        h_mean = torch.nn.functional.softplus(self.head_mean(feat))
        h_std = torch.nn.functional.softplus(self.head_log_std(feat)) + self.min_std

        return h_mean, h_std

    @torch.no_grad()
    def predict(self, x: np.ndarray | torch.Tensor) -> tuple[float, float]:
        """
        Single-sample inference. Returns (h_mean, h_std) as Python floats.
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        mean, std = self.forward(x)
        return float(mean.squeeze()), float(std.squeeze())


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood (mean over batch):

        L = log(sigma) + (y - mu)^2 / (2 * sigma^2)

    Parameters
    ----------
    mu     : (batch, 1)
    sigma  : (batch, 1)  must be > 0
    target : (batch,) or (batch, 1)
    """
    if target.ndim == 1:
        target = target.unsqueeze(1)
    sigma = sigma.clamp(min=eps)
    nll = torch.log(sigma) + (target - mu) ** 2 / (2.0 * sigma ** 2)
    return nll.mean()
