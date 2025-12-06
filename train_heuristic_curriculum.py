import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from learned_heuristic import HeuristicNet


class PlannerDatasetCurriculum(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = (X - mean) / std
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


def load_all_sizes(base_dir="data_curriculum"):
    sizes = [8, 16, 32]
    X_dict = {}
    y_dict = {}

    for sz in sizes:
        data = np.load(f"{base_dir}/planner_dataset_size{sz}.npz")
        X_dict[sz] = data["X"]
        y_dict[sz] = data["y"]
        print(f"[load] size={sz}: {X_dict[sz].shape[0]} samples")

    # global mean/std για όλα τα sizes μαζί
    X_concat = np.concatenate([X_dict[8], X_dict[16], X_dict[32]], axis=0)
    mean = X_concat.mean(axis=0)
    std = X_concat.std(axis=0) + 1e-6
    print("[norm] mean:", mean, "std:", std)

    np.savez("planner_dataset_norm_stats_curriculum.npz", mean=mean, std=std)

    datasets = {}
    for sz in sizes:
        datasets[sz] = PlannerDatasetCurriculum(X_dict[sz], y_dict[sz], mean, std)

    return datasets, mean, std


def train_curriculum(
    base_dir="data_curriculum",
    out_path="heuristic_net_curriculum_8_16_32.pt",
    hidden=64,
    epochs_per_stage=10,
    batch_size=256,
    lr=1e-3,
):
    datasets, mean, std = load_all_sizes(base_dir)

    net = HeuristicNet(in_dim=4, hidden=hidden)
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    def train_on_dataset(ds, label: str):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs_per_stage):
            total_loss = 0.0
            for Xb, yb in dl:
                pred = net(Xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(Xb)
            print(f"[stage={label}] Epoch {epoch+1}/{epochs_per_stage}: loss={total_loss/len(ds):.4f}")

    # Stage 1: size 8
    train_on_dataset(datasets[8], "8x8")

    # Stage 2: size 16
    train_on_dataset(datasets[16], "16x16")

    # Stage 3: size 32
    train_on_dataset(datasets[32], "32x32")

    torch.save(net.state_dict(), out_path)
    print(f"[curriculum] Saved model to {out_path}")


if __name__ == "__main__":
    train_curriculum()
