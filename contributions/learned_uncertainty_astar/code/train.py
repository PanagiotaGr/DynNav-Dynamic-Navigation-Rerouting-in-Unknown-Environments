"""
train.py

Training pipeline for UncertaintyHeuristicNet.

Dataset generation:
  - Run classic A* on randomly generated grids.
  - For each node on the optimal path, extract 11-D features
    and record the remaining cost to goal (ground truth).

Training:
  - Loss: Gaussian NLL (jointly trains mean and std heads).
  - Adam optimizer with cosine LR schedule.
  - Early stopping on validation NLL.

Usage
-----
    python train.py                     # quick smoke-test (few grids)
    python train.py --grids 500 --epochs 100 --out model.pt

    # from another module:
    from train import build_dataset, train_model
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from uncertainty_heuristic_net import UncertaintyHeuristicNet, gaussian_nll_loss
from uncertainty_astar import (
    EuclideanHeuristic,
    astar,
    extract_features,
)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def make_random_grid(
    H: int = 40,
    W: int = 40,
    obstacle_prob: float = 0.20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Random occupancy grid with short horizontal/vertical wall segments.
    obstacle_prob controls approximate density.
    """
    if rng is None:
        rng = np.random.default_rng()

    grid = np.zeros((H, W), dtype=np.int32)

    n_walls = int(H * W * obstacle_prob / 8)
    for _ in range(n_walls):
        r = int(rng.integers(1, H - 1))
        c = int(rng.integers(1, W - 1))
        length = int(rng.integers(3, 10))
        horizontal = bool(rng.integers(0, 2))
        if horizontal:
            grid[r, c: min(c + length, W - 1)] = 1
        else:
            grid[r: min(r + length, H - 1), c] = 1

    # keep corners free
    for corner in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
        grid[corner] = 0

    return grid


def _random_free_cell(grid: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    H, W = grid.shape
    while True:
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, W))
        if grid[r, c] == 0:
            return c, r  # (x, y)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(
    n_grids: int = 200,
    grid_h: int = 40,
    grid_w: int = 40,
    obstacle_prob: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run classic A* on `n_grids` random grids.
    For every node on each optimal path, collect:
      - 11-D feature vector
      - remaining cost to goal (ground truth)

    Returns
    -------
    X : (N, 11)  feature matrix
    y : (N,)     target remaining costs
    """
    rng = np.random.default_rng(seed)
    h_classic = EuclideanHeuristic()

    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for _ in range(n_grids):
        grid = make_random_grid(grid_h, grid_w, obstacle_prob, rng)
        start = _random_free_cell(grid, rng)
        goal = _random_free_cell(grid, rng)
        if start == goal:
            continue

        result = astar(grid, start, goal, h_classic, beta=0.0)
        if not result.found:
            continue

        path = result.path
        L = len(path)
        for i, node in enumerate(path):
            remaining = float(L - i - 1)
            feat = extract_features(node, goal, grid)
            X_list.append(feat)
            y_list.append(remaining)

    if len(X_list) == 0:
        raise RuntimeError("No training samples collected. Check grid generation.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 64,
    n_hidden: int = 3,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-3,
    val_fraction: float = 0.15,
    patience: int = 10,
    device: torch.device | None = None,
    seed: int = 0,
) -> UncertaintyHeuristicNet:
    """
    Train UncertaintyHeuristicNet with Gaussian NLL loss.

    Parameters
    ----------
    X, y        : dataset from build_dataset()
    hidden_dim  : neurons per layer
    n_hidden    : number of hidden layers
    epochs      : max training epochs
    batch_size  : mini-batch size
    lr          : initial Adam learning rate
    val_fraction: fraction of data held out for validation / early stopping
    patience    : early-stopping patience (epochs without val improvement)
    device      : torch device (auto-detected if None)
    seed        : reproducibility

    Returns
    -------
    Best model (by validation NLL).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X.shape[1]
    model = UncertaintyHeuristicNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    ).to(device)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(X_t, y_t)

    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 4)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            mu, sigma = model(X_batch)
            loss = gaussian_nll_loss(mu, sigma, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)

        scheduler.step()

        # --- validate ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                mu, sigma = model(X_batch)
                val_loss_sum += gaussian_nll_loss(mu, sigma, y_batch).item() * len(X_batch)

        train_nll = train_loss_sum / n_train
        val_nll = val_loss_sum / n_val

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train NLL={train_nll:.4f} | val NLL={val_nll:.4f}"
            )

        if val_nll < best_val_loss - 1e-4:
            best_val_loss = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    print(f"Training complete. Best val NLL={best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: UncertaintyHeuristicNet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.backbone[0].in_features,
            "hidden_dim": model.backbone[0].out_features,
            "n_hidden": sum(1 for m in model.backbone if isinstance(m, torch.nn.Linear)),
        },
        path,
    )
    print(f"Model saved to {path}")


def load_model(path: str | Path, device: torch.device | None = None) -> UncertaintyHeuristicNet:
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model = UncertaintyHeuristicNet(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        n_hidden=ckpt["n_hidden"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train uncertainty heuristic net")
    p.add_argument("--grids",   type=int,   default=100,  help="number of training grids")
    p.add_argument("--epochs",  type=int,   default=80,   help="max epochs")
    p.add_argument("--hidden",  type=int,   default=64,   help="hidden layer width")
    p.add_argument("--lr",      type=float, default=3e-3, help="learning rate")
    p.add_argument("--batch",   type=int,   default=256,  help="batch size")
    p.add_argument("--out",     type=str,   default="results/uncertainty_heuristic.pt")
    p.add_argument("--seed",    type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"[TRAIN] Building dataset from {args.grids} grids...")
    X, y = build_dataset(n_grids=args.grids, seed=args.seed)
    print(f"[TRAIN] Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    print("[TRAIN] Training model...")
    model = train_model(
        X, y,
        hidden_dim=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        seed=args.seed,
    )

    save_model(model, args.out)
