"""Classical forecasting + uncertainty baselines."""

from __future__ import annotations

import warnings

import numpy as np

from ua_forecast.conformal.wrappers import _finite_sample_quantile


class HistoricalMeanConformal:
    """
    Expanding/rolling historical mean point forecast with residual conformal.
    Strong classical baseline for coverage under mild non-stationarity.
    """

    def __init__(self, alpha: float = 0.1, window: int | None = 200):
        self.alpha = alpha
        self.window = window

    def fit_predict_interval(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (pred_test, lo, hi) with online residual updates."""
        train = list(np.asarray(y_train, dtype=float))
        residuals: list[float] = []
        for yv in np.asarray(y_val, dtype=float):
            mu = float(np.mean(train[-self.window :] if self.window else train))
            residuals.append(abs(yv - mu))
            train.append(float(yv))

        buf = residuals[-(self.window or len(residuals)) :]
        series = list(train)
        preds, lo, hi = [], [], []
        for yt in np.asarray(y_test, dtype=float):
            mu = float(np.mean(series[-self.window :] if self.window else series))
            q = _finite_sample_quantile(np.asarray(buf), self.alpha) if buf else 0.0
            preds.append(mu)
            lo.append(mu - q)
            hi.append(mu + q)
            buf.append(float(abs(yt - mu)))
            if self.window is not None and len(buf) > self.window:
                buf = buf[-self.window :]
            series.append(float(yt))
        return np.asarray(preds), np.asarray(lo), np.asarray(hi)


class GARCHConformal:
    """
    GARCH(1,1) mean/vol forecast with conformalized standardized residuals.

    Fits once on train+val (via `arch` if available), then recurses the GARCH
    volatility equation on the test stream. Falls back to HistoricalMeanConformal
    if `arch` is missing or fitting fails.
    """

    def __init__(self, alpha: float = 0.1, window: int = 200):
        self.alpha = alpha
        self.window = window
        self.fallback_ = False

    def fit_predict_interval(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        y_train = np.asarray(y_train, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        extras: dict = {"backend": "arch_garch11"}

        try:
            from arch import arch_model  # type: ignore
        except Exception:
            return self._fallback(y_train, y_val, y_test, extras)

        # Percent-like scale for arch numerical stability
        scale = float(np.std(y_train) + 1e-8)
        y_fit = np.concatenate([y_train, y_val]) / scale * 100.0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(y_fit, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
                res = am.fit(disp="off", show_warning=False)
            params = res.params
            mu = float(params.get("mu", 0.0))
            omega = float(params.get("omega", 1e-4))
            alpha_g = float(params.get("alpha[1]", 0.05))
            beta_g = float(params.get("beta[1]", 0.9))
            cond_vol = np.asarray(res.conditional_volatility, dtype=float)
        except Exception:
            return self._fallback(y_train, y_val, y_test, extras)

        # Calibrate conformal on standardized val residuals
        n_train = len(y_train)
        vol_val = cond_vol[n_train:]
        y_val_scaled = y_val / scale * 100.0
        scores = np.abs(y_val_scaled - mu) / np.maximum(vol_val, 1e-8)
        buf = list(scores[-self.window :])

        # Seed recursion from last in-sample state
        sig2 = float(cond_vol[-1] ** 2)
        last_resid = float(y_fit[-1] - mu)

        preds, lo, hi = [], [], []
        for yt in y_test:
            # one-step GARCH recursion
            sig2 = omega + alpha_g * (last_resid**2) + beta_g * sig2
            sig = float(np.sqrt(max(sig2, 1e-12)))
            mu_orig = mu * scale / 100.0
            sig_orig = sig * scale / 100.0

            q = _finite_sample_quantile(np.asarray(buf), self.alpha) if buf else 1.64485
            preds.append(mu_orig)
            lo.append(mu_orig - q * sig_orig)
            hi.append(mu_orig + q * sig_orig)

            yt_scaled = float(yt / scale * 100.0)
            last_resid = yt_scaled - mu
            buf.append(float(abs(last_resid) / max(sig, 1e-8)))
            if len(buf) > self.window:
                buf = buf[-self.window :]

        return np.asarray(preds), np.asarray(lo), np.asarray(hi), extras

    def _fallback(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        extras: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        self.fallback_ = True
        extras["backend"] = "historical_fallback"
        hist = HistoricalMeanConformal(alpha=self.alpha, window=self.window)
        pred, lo, hi = hist.fit_predict_interval(y_train, y_val, y_test)
        return pred, lo, hi, extras
