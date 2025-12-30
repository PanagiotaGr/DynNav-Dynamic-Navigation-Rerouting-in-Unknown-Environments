# ================================================================
# Online Drift Uncertainty - Experimental MC Dropout Version
# ================================================================

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim


# ---------------------------------------------------------------
# MLP για drift regression με Dropout (epistemic uncertainty)
# ---------------------------------------------------------------
class SimpleDriftUncertaintyNetExp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        output_dim: int = 1,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        dims = [input_dim, *hidden_dims]

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))   # Dropout για epistemic uncertainty

        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------
# Experimental Online Adapter
# ---------------------------------------------------------------
class OnlineDriftUncertaintyAdapterExp:
    """
    Experimental online adapter για drift uncertainty learning
    με MC Dropout για epistemic uncertainty.
    """

    def __init__(
        self,
        model_path: Optional[str],
        input_dim: int,
        lr: float = 1e-4,
        device: str = "cpu",
        max_buffer_size: int = 512,
        weight_decay: float = 0.0,
        hidden_dims: Sequence[int] = (64, 64),
        dropout_p: float = 0.1,
    ) -> None:

        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None

        # -----------------------------------------------------
        # Προσπάθεια φόρτωσης προϋπάρχοντος πειραματικού μοντέλου
        # -----------------------------------------------------
        if model_path is not None and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)

            if isinstance(state, nn.Module):
                self.model = state

            elif isinstance(state, dict):
                net = SimpleDriftUncertaintyNetExp(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    dropout_p=dropout_p,
                )

                if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
                    try:
                        net.load_state_dict(state["model_state_dict"])
                        self.model = net
                        print("[EXP-INFO] Loaded model_state_dict from checkpoint.")
                    except Exception as e:
                        print(f"[EXP-WARN] Failed to load model_state_dict: {e}")

                if self.model is None:
                    try:
                        net.load_state_dict(state)
                        self.model = net
                        print("[EXP-INFO] Loaded plain state_dict from file.")
                    except Exception as e:
                        print(f"[EXP-WARN] Failed to load state_dict: {e}")

        # -----------------------------------------------------
        # Αν δεν βρέθηκε μοντέλο, δημιουργούμε νέο πειραματικό
        # -----------------------------------------------------
        if self.model is None:
            print(
                "[EXP-WARN] No compatible experimental pretrained model found. "
                "Initializing SimpleDriftUncertaintyNetExp from scratch."
            )
            self.model = SimpleDriftUncertaintyNetExp(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_p=dropout_p,
            )

        self.model.to(self.device)
        self.model.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.loss_fn = nn.MSELoss()

        # Buffer δεδομένων
        self.max_buffer_size = max_buffer_size
        self._x_buffer: List[np.ndarray] = []
        self._y_buffer: List[np.ndarray] = []

    # ---------------------------------------------------------
    # Κανονική πρόβλεψη (χωρίς MC sampling)
    # ---------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()

        if x.ndim == 1:
            x = x[None, :]

        x_tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)

        with torch.no_grad():
            y_pred = self.model(x_tensor)

        return y_pred.detach().cpu().numpy()

    # ---------------------------------------------------------
    # Πρόβλεψη με epistemic uncertainty (MC Dropout)
    # ---------------------------------------------------------
    def predict_with_uncertainty(
        self, x: np.ndarray, n_samples: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:

        if x.ndim == 1:
            x = x[None, :]

        x_tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)

        self.model.train()  # ενεργό dropout
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                y = self.model(x_tensor)
                preds.append(y.detach().cpu().numpy())

        preds = np.stack(preds, axis=0)   # (S, N, 1)
        mean = preds.mean(axis=0)         # (N, 1)
        var = preds.var(axis=0)           # (N, 1)

        self.model.eval()
        return mean, var

    # ---------------------------------------------------------
    # Προσθήκη νέας παρατήρησης στο buffer
    # ---------------------------------------------------------
    def add_observation(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x_buffer.append(np.array(x, copy=True))
        self._y_buffer.append(np.array(y, copy=True))

        if len(self._x_buffer) > self.max_buffer_size:
            self._x_buffer.pop(0)
            self._y_buffer.pop(0)

    # ---------------------------------------------------------
    # Online update (ένα βήμα βελτιστοποίησης)
    # ---------------------------------------------------------
    def online_update(self, batch_size: int = 32) -> float:
        if len(self._x_buffer) == 0:
            return 0.0

        self.model.train()

        n = len(self._x_buffer)
        bsz = min(batch_size, n)
        idx = np.random.choice(n, size=bsz, replace=False)

        x_batch = np.stack([self._x_buffer[i] for i in idx], axis=0).astype(np.float32)
        y_batch = np.stack([self._y_buffer[i] for i in idx], axis=0).astype(np.float32)

        x_tensor = torch.from_numpy(x_batch).to(self.device)
        y_tensor = torch.from_numpy(y_batch).reshape(-1, 1).to(self.device)

        self.optimizer.zero_grad()
        y_pred = self.model(x_tensor)

        if y_pred.ndim == 1:
            y_pred = y_pred.view(-1, 1)

        loss = self.loss_fn(y_pred, y_tensor)
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        return float(loss.item())

    # ---------------------------------------------------------
    # Αποθήκευση πειραματικού μοντέλου
    # ---------------------------------------------------------
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        print(f"[EXP-INFO] Saved experimental model state_dict to: {path}")
