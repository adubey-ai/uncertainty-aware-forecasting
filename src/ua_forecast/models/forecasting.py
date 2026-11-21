"""Deep forecasting baselines: point LSTM, Gaussian NLL, quantile regression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PointLSTM(nn.Module):
    def __init__(
        self,
        lookback: int,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lookback = lookback
        self.lstm = nn.LSTM(
            1,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :]).squeeze(-1)


class GaussianLSTM(nn.Module):
    """Probabilistic LSTM predicting mean and log-std."""

    def __init__(
        self,
        lookback: int,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lookback = lookback
        self.lstm = nn.LSTM(
            1,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x.unsqueeze(-1))
        params = self.head(out[:, -1, :])
        mu = params[:, 0]
        log_sigma = params[:, 1].clamp(-6.0, 2.0)
        return mu, log_sigma


class QuantileLSTM(nn.Module):
    """Joint quantile head for lower/median/upper (default 0.05/0.5/0.95)."""

    def __init__(
        self,
        lookback: int,
        hidden: int = 64,
        n_layers: int = 2,
        quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lookback = lookback
        self.quantiles = quantiles
        self.lstm = nn.LSTM(
            1,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :])


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    target = target.unsqueeze(-1).expand_as(pred)
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, -1)
    err = target - pred
    return torch.mean(torch.maximum(q * err, (q - 1.0) * err))


def _gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(log_sigma)
    return torch.mean(0.5 * ((y - mu) / sigma) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi))


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    kind: str = "point",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    patience: int = 8,
    grad_clip: float = 1.0,
    weight_decay: float = 1e-5,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if kind == "point":
                pred = model(xb)
                loss = nn.functional.mse_loss(pred, yb)
            elif kind == "gaussian":
                mu, log_sigma = model(xb)
                loss = _gaussian_nll(mu, log_sigma, yb)
            elif kind == "quantile":
                pred = model(xb)
                loss = _pinball_loss(pred, yb, quantiles)
            else:
                raise ValueError(kind)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        train_loss = total / max(n, 1)

        model.eval()
        with torch.no_grad():
            if kind == "point":
                val_loss = float(nn.functional.mse_loss(model(x_val_t), y_val_t).item())
            elif kind == "gaussian":
                mu, log_sigma = model(x_val_t)
                val_loss = float(_gaussian_nll(mu, log_sigma, y_val_t).item())
            else:
                val_loss = float(_pinball_loss(model(x_val_t), y_val_t, quantiles).item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


@torch.no_grad()
def predict_point(model: PointLSTM, x: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    pred = model(torch.from_numpy(x).to(device))
    return pred.cpu().numpy()


@torch.no_grad()
def predict_gaussian(
    model: GaussianLSTM, x: np.ndarray, device: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    mu, log_sigma = model(torch.from_numpy(x).to(device))
    return mu.cpu().numpy(), np.exp(log_sigma.cpu().numpy())


@torch.no_grad()
def predict_quantile(
    model: QuantileLSTM, x: np.ndarray, device: str | None = None
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model(torch.from_numpy(x).to(device)).cpu().numpy()
