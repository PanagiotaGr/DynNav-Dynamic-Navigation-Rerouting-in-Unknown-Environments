import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Χρησιμοποιούμε το HeuristicNet από το learned_heuristic.py
from learned_heuristic import HeuristicNet


class PlannerDataset(Dataset):
    def __init__(self, npz_path: str = "planner_dataset.npz"):
        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]

        # προαιρετική κανονικοποίηση των inputs
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0) + 1e-6
        self.X = (self.X - self.X_mean) / self.X_std

        # σώζουμε τα στατιστικά για inference
        np.savez(
            "planner_dataset_norm_stats.npz",
            mean=self.X_mean,
            std=self.X_std,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


def train(
    npz_path: str = "planner_dataset.npz",
    out_path: str = "heuristic_net.pt",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    # Dataset + DataLoader
    ds = PlannerDataset(npz_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Δίκτυο: από learned_heuristic.py
    # Αν στο learned_heuristic.HeuristicNet χρειάζονται args, π.χ. in_dim=4, hidden=64,
    # βάλ' τα εδώ (π.χ. HeuristicNet(in_dim=4, hidden=64)).
    net = HeuristicNet(in_dim=4)
    net.train()

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

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    torch.save(net.state_dict(), out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    train()
