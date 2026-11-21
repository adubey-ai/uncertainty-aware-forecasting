"""Conformal prediction wrappers for sequential forecasting.

Includes:
- Split conformal (absolute residual)
- Frozen-window residual quantile (legacy offline baseline; not EnbPI)
- True EnbPI-style online residual updates
- ACI (Adaptive Conformal Inference)
- CQR (Conformalized Quantile Regression)
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import stats


def gaussian_interval(
    mu: np.ndarray, sigma: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric Gaussian predictive interval at coverage 1-alpha."""
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return mu - z * sigma, mu + z * sigma


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conservative finite-sample residual quantile for coverage 1-alpha."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return 0.0
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(min(max(level, 0.0), 1.0))
    return float(np.quantile(scores, level, method="higher"))


class SplitConformalRegressor:
    """Split conformal on absolute residuals of a point forecaster."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat_: float | None = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformalRegressor":
        scores = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        self.qhat_ = _finite_sample_quantile(scores, self.alpha)
        return self

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.qhat_ is None:
            raise RuntimeError("Call calibrate() first")
        y_pred = np.asarray(y_pred)
        return y_pred - self.qhat_, y_pred + self.qhat_


class FrozenWindowConformal:
    """
    Offline residual conformal using the last `window` *calibration* residuals.

    The residual buffer is frozen during the test stream (no online updates).
    Kept as a baseline; do not confuse with EnbPI.
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
        residuals = np.abs(np.asarray(y_true_calib) - np.asarray(y_pred_calib))
        hist = residuals[-self.window :]
        q = _finite_sample_quantile(hist, self.alpha) if len(hist) else 0.0
        y_pred_test = np.asarray(y_pred_test)
        return y_pred_test - q, y_pred_test + q


# Backwards-compatible alias (honest naming preferred going forward)
RollingConformalWrapper = FrozenWindowConformal


class EnbPIConformal:
    """
    EnbPI-style sequential conformal with online residual updates.

    After each test step, the absolute residual |y_t - pred_t| is appended to a
    sliding buffer and used for subsequent quantiles. Requires y_true at
    evaluation time (standard sequential CP protocol).
    """

    def __init__(self, alpha: float = 0.1, window: int = 200):
        self.alpha = alpha
        self.window = window

    def predict_interval(
        self,
        y_true_calib: np.ndarray,
        y_pred_calib: np.ndarray,
        y_pred_test: np.ndarray,
        y_true_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        residuals = list(np.abs(np.asarray(y_true_calib) - np.asarray(y_pred_calib)))
        buf: deque[float] = deque(residuals[-self.window :], maxlen=self.window)
        y_pred_test = np.asarray(y_pred_test, dtype=float)
        y_true_test = np.asarray(y_true_test, dtype=float)

        lo, hi = [], []
        widths = []
        for t, p in enumerate(y_pred_test):
            q = _finite_sample_quantile(np.asarray(buf), self.alpha) if buf else 0.0
            lo.append(p - q)
            hi.append(p + q)
            widths.append(2.0 * q)
            buf.append(float(abs(y_true_test[t] - p)))

        extras = {"mean_q": float(np.mean(widths) / 2.0) if widths else 0.0}
        return np.asarray(lo), np.asarray(hi), extras


class ACIConformal:
    """
    Adaptive Conformal Inference (Gibbs & Candès).

    Maintains a time-varying miscoverage level alpha_t updated online:
        alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t))
    where err_t = 1 if y_t not in C_t. Uses absolute residual scores with a
    sliding residual buffer (initialized from calibration).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.05,
        window: int = 200,
        alpha_min: float = 1e-3,
        alpha_max: float = 0.5,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def predict_interval(
        self,
        y_true_calib: np.ndarray,
        y_pred_calib: np.ndarray,
        y_pred_test: np.ndarray,
        y_true_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        residuals = list(np.abs(np.asarray(y_true_calib) - np.asarray(y_pred_calib)))
        buf: deque[float] = deque(residuals[-self.window :], maxlen=self.window)
        y_pred_test = np.asarray(y_pred_test, dtype=float)
        y_true_test = np.asarray(y_true_test, dtype=float)

        alpha_t = float(self.alpha)
        lo, hi = [], []
        alpha_path = []
        covered_flags = []

        for t, p in enumerate(y_pred_test):
            q = _finite_sample_quantile(np.asarray(buf), alpha_t) if buf else 0.0
            l, h = p - q, p + q
            lo.append(l)
            hi.append(h)
            err = 0.0 if (y_true_test[t] >= l and y_true_test[t] <= h) else 1.0
            covered_flags.append(1.0 - err)
            alpha_path.append(alpha_t)
            alpha_t = float(
                np.clip(
                    alpha_t + self.gamma * (self.alpha - err),
                    self.alpha_min,
                    self.alpha_max,
                )
            )
            buf.append(float(abs(y_true_test[t] - p)))

        extras = {
            "alpha_path": alpha_path,
            "mean_alpha": float(np.mean(alpha_path)) if alpha_path else self.alpha,
            "final_alpha": float(alpha_t),
            "local_coverage": float(np.mean(covered_flags)) if covered_flags else 0.0,
        }
        return np.asarray(lo), np.asarray(hi), extras


class CQR:
    """
    Conformalized Quantile Regression (Romano et al.).

    Nonconformity: s = max(q_lo - y, y - q_hi). Interval:
        [q_lo - qhat, q_hi + qhat].
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat_: float | None = None

    def calibrate(
        self,
        y_true: np.ndarray,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
    ) -> "CQR":
        y_true = np.asarray(y_true)
        q_lo = np.asarray(q_lo)
        q_hi = np.asarray(q_hi)
        scores = np.maximum(q_lo - y_true, y_true - q_hi)
        self.qhat_ = _finite_sample_quantile(scores, self.alpha)
        return self

    def predict_interval(
        self, q_lo: np.ndarray, q_hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.qhat_ is None:
            raise RuntimeError("Call calibrate() first")
        q_lo = np.asarray(q_lo)
        q_hi = np.asarray(q_hi)
        return q_lo - self.qhat_, q_hi + self.qhat_
