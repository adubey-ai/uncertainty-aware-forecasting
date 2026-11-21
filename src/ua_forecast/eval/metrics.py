"""Uncertainty evaluation metrics and temporal diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import stats


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


def under_coverage_rate(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of points falling below the lower endpoint."""
    return float(np.mean(y < lo))


def over_coverage_rate(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of points falling above the upper endpoint."""
    return float(np.mean(y > hi))


def winkler_score(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Mean Winkler / interval score (lower is better).
    Penalizes width and misses proportional to alpha.
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    score = width.copy()
    below = y < lo
    above = y > hi
    score[below] += (2.0 / alpha) * (lo[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - hi[above])
    return float(np.mean(score))


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-8)
    return float(np.mean(0.5 * ((y - mu) / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)))


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for Gaussian predictive distributions (Gneiting)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    z = (y - mu) / sigma
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_quantile_approx(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_mid: np.ndarray,
    q_hi: np.ndarray,
    levels: tuple[float, float, float] = (0.05, 0.5, 0.95),
) -> float:
    """
    Rough CRPS proxy from three quantiles via pinball average.
    Not a full CRPS integral; useful for ranking quantile models.
    """
    y = np.asarray(y, dtype=float)
    preds = [np.asarray(q_lo), np.asarray(q_mid), np.asarray(q_hi)]
    total = 0.0
    for q, p in zip(levels, preds):
        err = y - p
        total += np.mean(np.maximum(q * err, (q - 1.0) * err))
    return float(total / len(levels))


def rolling_coverage(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """Rolling local PICP over a sliding window."""
    y = np.asarray(y)
    hit = ((y >= lo) & (y <= hi)).astype(float)
    if len(hit) == 0:
        return np.asarray([])
    out = np.empty(len(hit), dtype=float)
    csum = np.cumsum(hit)
    for t in range(len(hit)):
        a = max(0, t - window + 1)
        n = t - a + 1
        prev = csum[a - 1] if a > 0 else 0.0
        out[t] = (csum[t] - prev) / n
    return out


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


def shift_onset_coverage(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    regimes: np.ndarray,
    post_window: int = 30,
) -> dict:
    """
    Mean coverage in the `post_window` steps after each regime switch.
    Captures calibration lag under non-stationarity.
    """
    regimes = np.asarray(regimes)
    switches = np.where(np.diff(regimes) != 0)[0] + 1
    if len(switches) == 0:
        return {"n_switches": 0, "post_switch_coverage": None, "post_switch_n": 0}

    hits = []
    for s in switches:
        end = min(len(y), s + post_window)
        if end <= s:
            continue
        hits.append(((y[s:end] >= lo[s:end]) & (y[s:end] <= hi[s:end])).astype(float))
    if not hits:
        return {"n_switches": int(len(switches)), "post_switch_coverage": None, "post_switch_n": 0}
    cat = np.concatenate(hits)
    return {
        "n_switches": int(len(switches)),
        "post_switch_coverage": float(np.mean(cat)),
        "post_switch_n": int(len(cat)),
    }


def summarize_intervals(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    target_coverage: float = 0.9,
    regimes: np.ndarray | None = None,
    alpha: float | None = None,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    q_mid: np.ndarray | None = None,
) -> dict:
    alpha = alpha if alpha is not None else (1.0 - target_coverage)
    summary = {
        "coverage": coverage(y, lo, hi),
        "mean_width": mean_width(lo, hi),
        "sharpness": sharpness(lo, hi),
        "calibration_error": calibration_error(y, lo, hi, target_coverage),
        "under_coverage": under_coverage_rate(y, lo, hi),
        "over_coverage": over_coverage_rate(y, lo, hi),
        "winkler": winkler_score(y, lo, hi, alpha=alpha),
        "target_coverage": target_coverage,
        "n": int(len(y)),
    }
    if regimes is not None:
        summary["regime_coverage"] = regime_coverage(y, lo, hi, regimes)
        summary["shift_onset"] = shift_onset_coverage(y, lo, hi, regimes)
    if mu is not None and sigma is not None:
        summary["nll"] = gaussian_nll(y, mu, sigma)
        summary["crps"] = crps_gaussian(y, mu, sigma)
    if q_mid is not None:
        summary["crps_quantile_proxy"] = crps_quantile_approx(y, lo, q_mid, hi)
    return summary


def aggregate_seed_metrics(seed_dicts: list[dict]) -> dict:
    """Mean ± stderr across seeds for scalar metrics."""
    if not seed_dicts:
        return {}
    keys = [
        k
        for k, v in seed_dicts[0].items()
        if isinstance(v, (int, float)) and k not in {"n", "target_coverage"}
    ]
    out: dict = {"n_seeds": len(seed_dicts)}
    for k in keys:
        vals = np.asarray([d[k] for d in seed_dicts if k in d], dtype=float)
        out[k] = {
            "mean": float(np.mean(vals)),
            "stderr": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "values": vals.tolist(),
        }
    # keep last seed's regime / shift diagnostics as illustrative
    if "regime_coverage" in seed_dicts[-1]:
        out["regime_coverage_last_seed"] = seed_dicts[-1]["regime_coverage"]
    if "shift_onset" in seed_dicts[-1]:
        out["shift_onset_last_seed"] = seed_dicts[-1]["shift_onset"]
    return out
