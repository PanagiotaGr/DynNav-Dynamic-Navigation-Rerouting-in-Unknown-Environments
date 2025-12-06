import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from learned_heuristic import HeuristicNet  # το ίδιο net που έχεις ήδη


def manhattan_from_features(X: np.ndarray) -> np.ndarray:
    """
    Υποθέτουμε ότι X[:,0:4] = [x, y, gx, gy].
    Υπολόγισε Manhattan distance για κάθε sample.
    """
    x = X[:, 0]
    y = X[:, 1]
    gx = X[:, 2]
    gy = X[:, 3]
    return np.abs(x - gx) + np.abs(y - gy)


class ResidualDataset(Dataset):
    """
    Dataset για residual training: target = h*(s) - h_man(s)
    """
    def __init__(self, npz_path: str = "planner_dataset.npz"):
        data = np.load(npz_path)
        self.X = data["X"]          # features [x, y, gx, gy]
        self.y = data["y"]          # ground-truth cost-to-go h*(s)

        # υπολογίζουμε Manhattan baseline
        self.h_man = manhattan_from_features(self.X)
        self.residual = self.y - self.h_man

        # κανονικοποίηση X (όπως πριν)
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0) + 1e-6
        self.X = (self.X - self.X_mean) / self.X_std

        # σώζουμε τα normalization stats για inference
        np.savez(
            "planner_dataset_norm_stats_residual.npz",
            mean=self.X_mean,
            std=self.X_std,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        r = torch.tensor(self.residual[i], dtype=torch.float32)
        return x, r


def train(
    npz_path: str = "planner_dataset.npz",
    out_path: str = "heuristic_net_residual.pt",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
):
    ds = ResidualDataset(npz_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    net = HeuristicNet(in_dim=4, hidden=hidden)
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for Xb, rb in dl:
            pred_r = net(Xb).squeeze(-1)
            loss = loss_fn(pred_r, rb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(Xb)

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch+1}: residual MSE = {avg_loss:.4f}")

    torch.save(net.state_dict(), out_path)
    print(f"Saved residual model to {out_path}")


if __name__ == "__main__":
    train()
