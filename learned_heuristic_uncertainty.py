import torch
import torch.nn as nn
import torch.nn.functional as F


class HeuristicNetUncertainty(nn.Module):
    """
    Neural heuristic with uncertainty (mu, log_var).
    Input: 4 dims → [x, y, gx, gy]
    Output: mu(s), log_var(s)
    """
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mu = nn.Linear(hidden, 1)
        self.fc_logvar = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

