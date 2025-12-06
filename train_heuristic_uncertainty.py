import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from learned_heuristic_uncertainty import HeuristicNetUncertainty


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


def gaussian_nll(mu, log_var, target):
    return 0.5 * (log_var + (target - mu)**2 / torch.exp(log_var))


def train():
    ds = PlannerDataset("planner_dataset.npz")
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    net = HeuristicNetUncertainty(in_dim=4, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(40):
        total_loss = 0.0
        for xb, yb in dl:
            mu, log_var = net(xb)
            loss = gaussian_nll(mu.squeeze(), log_var.squeeze(), yb).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(xb)

        print(f"Epoch {epoch+1}: loss = {total_loss/len(ds):.4f}")

    torch.save(net.state_dict(), "heuristic_net_unc.pt")
    print("Saved heuristic_net_unc.pt")


if __name__ == "__main__":
    train()
