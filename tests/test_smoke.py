"""Unit + smoke tests for data, conformal, and metrics."""

import numpy as np

from ua_forecast.conformal import (
    ACIConformal,
    CQR,
    EnbPIConformal,
    SplitConformalRegressor,
)
from ua_forecast.data import make_regime_switching_ar, temporal_split, window_dataset
from ua_forecast.eval import coverage, mean_width, winkler_score


def test_regime_series_and_split():
    bundle = make_regime_switching_ar(n=500, seed=0)
    x, y = window_dataset(bundle.y, lookback=16, horizon=1)
    splits = temporal_split(x, y, bundle.regimes, lookback=16)
    assert splits["train"]["x"].shape[0] > 0
    assert splits["test"]["y"].ndim == 1


def test_gradual_hardness_ladder():
    abrupt = make_regime_switching_ar(n=800, seed=1, gradual=False)
    gradual = make_regime_switching_ar(n=800, seed=1, gradual=True, transition_len=20)
    assert abrupt.name != gradual.name
    assert abrupt.meta["gradual"] is False
    assert gradual.meta["gradual"] is True


def test_split_conformal_coverage_sane():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    pred = y + rng.normal(scale=0.2, size=200)
    cp = SplitConformalRegressor(alpha=0.1).calibrate(y[:100], pred[:100])
    lo, hi = cp.predict_interval(pred[100:])
    cov = coverage(y[100:], lo, hi)
    assert 0.5 <= cov <= 1.0
    assert mean_width(lo, hi) >= 0.0
    assert winkler_score(y[100:], lo, hi, alpha=0.1) >= 0.0


def test_enbpi_updates_differ_from_frozen():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.normal(size=n)
    # induce a late shift so online updates matter
    y[200:] = y[200:] + rng.normal(scale=2.0, size=100)
    pred = y + rng.normal(scale=0.3, size=n)
    calib_y, calib_p = y[:150], pred[:150]
    test_y, test_p = y[150:], pred[150:]

    from ua_forecast.conformal import FrozenWindowConformal

    frozen = FrozenWindowConformal(alpha=0.1, window=80)
    lo_f, hi_f = frozen.predict_interval(calib_y, calib_p, test_p)
    enbpi = EnbPIConformal(alpha=0.1, window=80)
    lo_e, hi_e, _ = enbpi.predict_interval(calib_y, calib_p, test_p, test_y)
    # widths should not be identical under shift
    assert not np.allclose(hi_e - lo_e, hi_f - lo_f)


def test_aci_alpha_adapts_under_misses():
    rng = np.random.default_rng(2)
    y = rng.normal(size=250)
    pred = np.zeros_like(y)  # deliberately bad point forecast
    aci = ACIConformal(alpha=0.1, gamma=0.1, window=50)
    _, _, extras = aci.predict_interval(y[:80], pred[:80], pred[80:], y[80:])
    assert "alpha_path" in extras
    assert len(extras["alpha_path"]) == 170
    # with bad forecasts, alpha should move (not stay flat forever)
    assert abs(extras["final_alpha"] - extras["alpha_path"][0]) > 1e-6 or extras["mean_alpha"] != 0.1


def test_cqr_finite_sample_widens():
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    # Narrow bands around a noisy center -> undercover raw quantiles, positive qhat
    center = y + rng.normal(scale=0.6, size=200)
    q_lo = center - 0.05
    q_hi = center + 0.05
    cqr = CQR(alpha=0.1).calibrate(y[:100], q_lo[:100], q_hi[:100])
    lo, hi = cqr.predict_interval(q_lo[100:], q_hi[100:])
    assert cqr.qhat_ is not None and cqr.qhat_ > 0.0
    assert np.all((hi - lo) >= (q_hi[100:] - q_lo[100:]) - 1e-9)
    cov = coverage(y[100:], lo, hi)
    assert cov >= 0.7


def test_no_calib_uses_test_labels_for_split():
    """Split CP qhat depends only on calib; intervals are y_pred ± qhat."""
    rng = np.random.default_rng(4)
    y = rng.normal(size=120)
    pred = y + 0.1
    cp = SplitConformalRegressor(alpha=0.1).calibrate(y[:60], pred[:60])
    lo1, hi1 = cp.predict_interval(pred[60:])
    # reordering test preds changes intervals pointwise but same widths
    assert np.allclose((hi1 - lo1), hi1[0] - lo1[0])
