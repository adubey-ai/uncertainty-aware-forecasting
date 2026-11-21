"""Synthetic regime-switching and finance-like time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeriesBundle:
    y: np.ndarray
    regimes: np.ndarray
    name: str


def make_regime_switching_ar(
    n: int = 4000,
    n_regimes: int = 3,
    switch_prob: float = 0.01,
    seed: int = 42,
) -> SeriesBundle:
    """Piecewise AR(1) with abrupt mean/volatility regime switches."""
    rng = np.random.default_rng(seed)
    means = np.array([-0.4, 0.0, 0.55])[:n_regimes]
    phis = np.array([0.2, 0.7, -0.3])[:n_regimes]
    sigmas = np.array([0.35, 0.15, 0.55])[:n_regimes]

    regimes = np.zeros(n, dtype=int)
    y = np.zeros(n, dtype=float)
    r = int(rng.integers(0, n_regimes))
    regimes[0] = r
    y[0] = means[r] + sigmas[r] * rng.standard_normal()

    for t in range(1, n):
        if rng.random() < switch_prob:
            r = int(rng.integers(0, n_regimes))
        regimes[t] = r
        y[t] = means[r] + phis[r] * y[t - 1] + sigmas[r] * rng.standard_normal()

    return SeriesBundle(y=y, regimes=regimes, name="synthetic_regime_ar")


def make_finance_like_returns(
    n: int = 3000,
    vol_regimes: list[float] | None = None,
    seed: int = 42,
) -> SeriesBundle:
    """
    Log-return style series with stochastic volatility regime switches.
    Designed to be reproducible without external market downloads.
    """
    rng = np.random.default_rng(seed + 7)
    vols = np.array(vol_regimes or [0.008, 0.02, 0.045], dtype=float)
    n_regimes = len(vols)
    regimes = np.zeros(n, dtype=int)
    rets = np.zeros(n, dtype=float)
    r = 0
    for t in range(n):
        if rng.random() < 0.008:
            r = int(rng.integers(0, n_regimes))
        regimes[t] = r
        # mild leverage: negative returns bump next-step vol
        shock = rng.standard_normal()
        rets[t] = vols[r] * shock
        if shock < -1.5 and r < n_regimes - 1 and rng.random() < 0.3:
            r = min(n_regimes - 1, r + 1)
    # Forecast returns (approximately locally stationary within regimes), not integrated price.
    y = (rets - rets.mean()) / (rets.std() + 1e-8)
    return SeriesBundle(y=y.astype(float), regimes=regimes, name="finance_like_returns")


def window_dataset(
    y: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised windows X[t-L:t] -> y[t+h-1]."""
    xs, ys = [], []
    end = len(y) - horizon + 1
    for t in range(lookback, end):
        xs.append(y[t - lookback : t])
        ys.append(y[t + horizon - 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def temporal_split(
    x: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray | None,
    lookback: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict:
    """Chronological train/val/test split (no shuffle)."""
    n = len(y)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough samples for temporal split")

    slices = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }
    out = {}
    for name, sl in slices.items():
        out[name] = {"x": x[sl], "y": y[sl]}
        if regimes is not None:
            # regime aligned to window end index
            aligned = regimes[lookback : lookback + n]
            out[name]["regimes"] = aligned[sl]
    return out
