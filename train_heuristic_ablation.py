import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from learned_heuristic import HeuristicNet  # πρέπει να έχει in_dim, hidden


class PlannerDataset(Dataset):
    def __init__(self, npz_path="planner_dataset.npz"):
        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]

        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-6
        self.X = (self.X - self.mean) / self.std

        np.savez("planner_dataset_norm_stats.npz",
                 mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


def train_for_hidden(hidden: int, epochs: int = 30, lr: float = 1e-3):
    ds = PlannerDataset("planner_dataset.npz")
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
        print(f"[hidden={hidden}] Epoch {epoch+1}: loss = {total_loss / len(ds):.4f}")

    out_path = f"heuristic_net_h{hidden}.pt"
    torch.save(net.state_dict(), out_path)
    print(f"[hidden={hidden}] Saved {out_path}")


def main():
    for hidden in [32, 64, 128]:
        print(f"=== Training HeuristicNet with hidden={hidden} ===")
        train_for_hidden(hidden)


if __name__ == "__main__":
    main()
