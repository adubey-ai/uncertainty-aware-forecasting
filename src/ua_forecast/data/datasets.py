"""Synthetic regime-switching, finance-like, and hardness-ladder series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeriesBundle:
    y: np.ndarray
    regimes: np.ndarray
    name: str
    meta: dict | None = None


def make_regime_switching_ar(
    n: int = 4000,
    n_regimes: int = 3,
    switch_prob: float = 0.01,
    seed: int = 42,
    gradual: bool = False,
    transition_len: int = 25,
) -> SeriesBundle:
    """
    Piecewise AR(1) with abrupt or gradual mean/volatility regime switches.

    If gradual=True, parameters linearly interpolate over `transition_len`
    steps after each switch (hardness ladder: easier local stationarity).
    """
    rng = np.random.default_rng(seed)
    means = np.array([-0.4, 0.0, 0.55])[:n_regimes]
    phis = np.array([0.2, 0.7, -0.3])[:n_regimes]
    sigmas = np.array([0.35, 0.15, 0.55])[:n_regimes]

    regimes = np.zeros(n, dtype=int)
    y = np.zeros(n, dtype=float)
    r = int(rng.integers(0, n_regimes))
    regimes[0] = r
    y[0] = means[r] + sigmas[r] * rng.standard_normal()

    # transition state
    r_prev = r
    trans_left = 0

    for t in range(1, n):
        if rng.random() < switch_prob:
            r_prev = r
            r = int(rng.integers(0, n_regimes))
            trans_left = transition_len if gradual else 0
        regimes[t] = r

        if gradual and trans_left > 0:
            w = 1.0 - (trans_left / max(transition_len, 1))
            mean_t = (1 - w) * means[r_prev] + w * means[r]
            phi_t = (1 - w) * phis[r_prev] + w * phis[r]
            sig_t = (1 - w) * sigmas[r_prev] + w * sigmas[r]
            trans_left -= 1
        else:
            mean_t, phi_t, sig_t = means[r], phis[r], sigmas[r]

        y[t] = mean_t + phi_t * y[t - 1] + sig_t * rng.standard_normal()

    name = "synthetic_regime_ar_gradual" if gradual else "synthetic_regime_ar"
    return SeriesBundle(
        y=y,
        regimes=regimes,
        name=name,
        meta={"gradual": gradual, "switch_prob": switch_prob, "n_regimes": n_regimes},
    )


def make_finance_like_returns(
    n: int = 3000,
    vol_regimes: list[float] | None = None,
    seed: int = 42,
    leverage: bool = True,
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
        shock = rng.standard_normal()
        rets[t] = vols[r] * shock
        if leverage and shock < -1.5 and r < n_regimes - 1 and rng.random() < 0.3:
            r = min(n_regimes - 1, r + 1)
    y = (rets - rets.mean()) / (rets.std() + 1e-8)
    name = "finance_like_returns" if leverage else "finance_like_returns_noleverage"
    return SeriesBundle(
        y=y.astype(float),
        regimes=regimes,
        name=name,
        meta={"leverage": leverage, "vol_regimes": vols.tolist()},
    )


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
            aligned = regimes[lookback : lookback + n]
            out[name]["regimes"] = aligned[sl]
    return out
