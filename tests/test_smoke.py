"""Smoke tests for data + metrics."""

import numpy as np

from ua_forecast.data import make_regime_switching_ar, window_dataset, temporal_split
from ua_forecast.eval import coverage, mean_width
from ua_forecast.conformal import SplitConformalRegressor


def test_regime_series_and_split():
    bundle = make_regime_switching_ar(n=500, seed=0)
    x, y = window_dataset(bundle.y, lookback=16, horizon=1)
    splits = temporal_split(x, y, bundle.regimes, lookback=16)
    assert splits["train"]["x"].shape[0] > 0
    assert splits["test"]["y"].ndim == 1


def test_split_conformal_coverage_sane():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    pred = y + rng.normal(scale=0.2, size=200)
    cp = SplitConformalRegressor(alpha=0.1).calibrate(y[:100], pred[:100])
    lo, hi = cp.predict_interval(pred[100:])
    cov = coverage(y[100:], lo, hi)
    assert 0.5 <= cov <= 1.0
    assert mean_width(lo, hi) >= 0.0
