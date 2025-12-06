from typing import Tuple, Any, Dict
import math
import torch

# HeuristicNet: το βασικό δίκτυο (point estimate) από train_heuristic / learned_heuristic
from train_heuristic import HeuristicNet

# HeuristicNetUncertainty: το δίκτυο με (mu, log_var)
from learned_heuristic_uncertainty import HeuristicNetUncertainty


# 1) Κλασικό heuristic (Manhattan για 4-connected grid)
def manhattan_heuristic(state: Tuple[int, int], problem) -> float:
    x, y = state
    gx, gy = problem.goal
    return abs(x - gx) + abs(y - gy)


# 2) Learned heuristic (point estimate)
class LearnedHeuristic:
    """
    Τυλίγει το HeuristicNet που έχεις εκπαιδεύσει και αποθηκεύσει σε heuristic_net.pt.
    Υποθέτει input [x, y, gx, gy].
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        # αν το HeuristicNet θέλει input_dim αντί για in_dim, άλλαξέ το ανάλογα
        self.model = HeuristicNet(in_dim=4)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def __call__(self, state: Tuple[int, int], problem) -> float:
        x, y = state
        gx, gy = problem.goal
        inp = torch.tensor([[x, y, gx, gy]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h_pred = self.model(inp).squeeze().item()
        return max(0.0, float(h_pred))


# 3) Uncertainty-aware heuristic (μ, σ) → LCB = μ - kσ
class UncertaintyAwareHeuristic:
    """
    Χρησιμοποιεί HeuristicNetUncertainty για να βγάλει (mu, log_var)
    και εφαρμόζει LCB heuristic: h(s) = max(0, mu - k * sigma).
    """
    def __init__(self, model_path: str, k: float = 1.0, device: str = "cpu"):
        self.device = device
        self.model = HeuristicNetUncertainty(in_dim=4, hidden=64)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.k = k

    def __call__(self, state: Tuple[int, int], problem) -> float:
        x, y = state
        gx, gy = problem.goal
        inp = torch.tensor([[x, y, gx, gy]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, log_var = self.model(inp)
            mu = mu.squeeze().item()
            log_var = log_var.squeeze().item()
        sigma = math.sqrt(math.exp(log_var))
        h_lcb = mu - self.k * sigma
        return max(0.0, float(h_lcb))


# 4) Online-update heuristic (TD-style) ΠΑΝΩ σε μια βάση (π.χ. Manhattan)
class OnlineUpdateHeuristic:
    """
    Βάση: οποιοδήποτε heuristic base_h(state, problem)
    Online TD update:
        h(s) <- (1-η)h(s) + η (c(s,s') + h(s'))
    Κρατάμε dictionary h_values ανά state.
    """
    def __init__(self, base_h, eta: float = 0.1):
        self.base_h = base_h
        self.eta = eta
        self.h_values: Dict[Any, float] = {}

    def get_h(self, s, problem) -> float:
        if s not in self.h_values:
            self.h_values[s] = float(self.base_h(s, problem))
        return self.h_values[s]

    def update(self, s, s_next, cost, problem):
        """
        TD-style update χρησιμοποιώντας τοπική Bellman relation.
        """
        h_s = self.get_h(s, problem)
        h_next = self.get_h(s_next, problem)
        target = cost + h_next
        new_val = (1 - self.eta) * h_s + self.eta * target
        self.h_values[s] = new_val

    def __call__(self, state, problem) -> float:
        return self.get_h(state, problem)

