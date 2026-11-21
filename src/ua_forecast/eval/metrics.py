"""Uncertainty evaluation metrics."""

from __future__ import annotations

import numpy as np


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def sharpness(lo: np.ndarray, hi: np.ndarray) -> float:
    """Alias for mean predictive interval width (lower is sharper)."""
    return mean_width(lo, hi)


def calibration_error(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    target_coverage: float = 0.9,
) -> float:
    """Absolute deviation from target marginal coverage."""
    return float(abs(coverage(y, lo, hi) - target_coverage))


def regime_coverage(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    regimes: np.ndarray,
) -> dict[int, float]:
    out = {}
    for r in sorted(np.unique(regimes)):
        m = regimes == r
        if np.any(m):
            out[int(r)] = coverage(y[m], lo[m], hi[m])
    return out


def summarize_intervals(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    target_coverage: float = 0.9,
    regimes: np.ndarray | None = None,
) -> dict:
    summary = {
        "coverage": coverage(y, lo, hi),
        "mean_width": mean_width(lo, hi),
        "sharpness": sharpness(lo, hi),
        "calibration_error": calibration_error(y, lo, hi, target_coverage),
        "target_coverage": target_coverage,
        "n": int(len(y)),
    }
    if regimes is not None:
        summary["regime_coverage"] = regime_coverage(y, lo, hi, regimes)
    return summary
