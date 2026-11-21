"""Conformal prediction wrappers for sequential forecasting."""

from __future__ import annotations

import numpy as np
from scipy import stats


def gaussian_interval(
    mu: np.ndarray, sigma: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric Gaussian predictive interval at coverage 1-alpha."""
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return mu - z * sigma, mu + z * sigma


class SplitConformalRegressor:
    """
    Split conformal on absolute residuals of a point forecaster.
    Calibration set is chronological (validation window).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat_: float | None = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformalRegressor":
        scores = np.abs(y_true - y_pred)
        n = len(scores)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(max(level, 0.0), 1.0)
        self.qhat_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.qhat_ is None:
            raise RuntimeError("Call calibrate() first")
        return y_pred - self.qhat_, y_pred + self.qhat_


class RollingConformalWrapper:
    """
    Rolling residual conformal (EnbPI-style light):
    use the most recent `window` absolute residuals to set the quantile.
    """

    def __init__(self, alpha: float = 0.1, window: int = 200):
        self.alpha = alpha
        self.window = window

    def predict_interval(
        self,
        y_true_calib: np.ndarray,
        y_pred_calib: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        residuals = np.abs(y_true_calib - y_pred_calib)
        lo, hi = [], []
        hist = list(residuals[-self.window :])
        for p in y_pred_test:
            q = float(np.quantile(hist, 1.0 - self.alpha, method="higher")) if hist else 0.0
            lo.append(p - q)
            hi.append(p + q)
            # freeze history for offline evaluation (no test leakage)
        return np.asarray(lo), np.asarray(hi)
