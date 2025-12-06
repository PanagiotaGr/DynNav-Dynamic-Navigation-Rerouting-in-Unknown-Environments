from typing import Tuple
import numpy as np
import torch

from .heuristics import manhattan_heuristic
from learned_heuristic import HeuristicNet  # από root


class ResidualLearnedHeuristic:
    """
    Heuristic = Manhattan + learned residual.

    Εκπαιδεύεται με target r(s) = h*(s) - h_man(s).
    Στο inference:
        h_hat(s) = h_man(s) + r_hat(s)
    """
    def __init__(
        self,
        model_path: str = "heuristic_net_residual.pt",
        stats_path: str = "planner_dataset_norm_stats_residual.npz",
        device: str = "cpu",
        hidden: int = 64,
    ):
        self.device = device

        # φορτώνουμε το μοντέλο residual
        self.model = HeuristicNet(in_dim=4, hidden=hidden)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # φορτώνουμε τα normalization stats
        data = np.load(stats_path)
        self.X_mean = data["mean"]
        self.X_std = data["std"]

    def _features(self, state: Tuple[int, int], problem) -> np.ndarray:
        x, y = state
        gx, gy = problem.goal
        return np.array([x, y, gx, gy], dtype=np.float32)

    def __call__(self, state: Tuple[int, int], problem) -> float:
        # baseline heuristic
        h_man = manhattan_heuristic(state, problem)

        # features
        x_np = self._features(state, problem)
        x_norm = (x_np - self.X_mean) / self.X_std

        inp = torch.tensor([x_norm], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            r_hat = self.model(inp).squeeze().item()

        h_hat = h_man + r_hat
        return max(0.0, float(h_hat))
