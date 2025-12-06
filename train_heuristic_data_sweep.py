import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from learned_heuristic import HeuristicNet


class PlannerDataset(Dataset):
    def __init__(self, npz_path="planner_dataset.npz", fraction: float = 1.0, seed: int = 0):
        data = np.load(npz_path)
        X = data["X"]
        y = data["y"]

        # shuffle & subsample
        rng = np.random.default_rng(seed)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        n = int(len(X) * fraction)
        idx = idx[:n]

        self.X = X[idx]
        self.y = y[idx]

        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-6
        self.X = (self.X - self.mean) / self.std

        np.savez(f"planner_dataset_norm_stats_frac_{int(fraction*100)}.npz",
                 mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


def train_for_fraction(fraction: float, epochs: int = 30, lr: float = 1e-3, hidden: int = 64):
    print(f"=== Training with fraction={fraction:.2f} ===")
    ds = PlannerDataset("planner_dataset.npz", fraction=fraction, seed=0)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    net = HeuristicNet(in_dim=4, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for Xb, yb in dl:
            pred = net(Xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(Xb)
        print(f"[frac={fraction:.2f}] Epoch {epoch+1}: loss = {total_loss / len(ds):.4f}")

    out_path = f"heuristic_net_frac_{int(fraction*100)}.pt"
    torch.save(net.state_dict(), out_path)
    print(f"[frac={fraction:.2f}] Saved {out_path}")


def main():
    for frac in [0.1, 0.25, 0.5, 1.0]:
        train_for_fraction(frac)


if __name__ == "__main__":
    main()
