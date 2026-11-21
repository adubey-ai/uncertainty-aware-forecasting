# Experiment summary
Target coverage: 90%

Metrics are mean ± stderr across seeds (where available).

## synthetic_regime_ar
| Method | Coverage | Mean width | Winkler | Cal. error |
|---|---:|---:|---:|---:|
| point_lstm_split_conformal | 0.850±0.020 | 1.356 | 2.157 | 0.050 |
| point_lstm_frozen_window_cp | 0.908±0.026 | 1.683 | 2.154 | 0.051 |
| point_lstm_enbpi | 0.894±0.012 | 1.580 | 2.049 | 0.018 |
| point_lstm_aci | 0.899±0.001 | 1.605 | 2.047 | 0.002 |
| gaussian_lstm | 0.908±0.006 | 1.559 | 1.950 | 0.012 |
| quantile_lstm | 0.890±0.011 | 1.499 | 1.963 | 0.021 |
| quantile_lstm_cqr | 0.898±0.010 | 1.539 | 1.957 | 0.014 |
| historical_mean_cp | 0.898±0.008 | 1.924 | 2.461 | 0.012 |
| garch_conformal | 0.906±0.003 | 2.215 | 2.654 | 0.006 |

## synthetic_regime_ar_gradual
| Method | Coverage | Mean width | Winkler | Cal. error |
|---|---:|---:|---:|---:|
| point_lstm_split_conformal | 0.864±0.026 | 1.304 | 2.031 | 0.042 |
| point_lstm_frozen_window_cp | 0.847±0.053 | 1.326 | 2.164 | 0.086 |
| point_lstm_enbpi | 0.896±0.008 | 1.407 | 1.896 | 0.015 |
| point_lstm_aci | 0.899±0.002 | 1.392 | 1.795 | 0.003 |
| gaussian_lstm | 0.897±0.011 | 1.333 | 1.769 | 0.017 |
| quantile_lstm | 0.895±0.006 | 1.306 | 1.733 | 0.010 |
| quantile_lstm_cqr | 0.894±0.013 | 1.315 | 1.739 | 0.017 |
| historical_mean_cp | 0.898±0.006 | 1.733 | 2.228 | 0.010 |
| garch_conformal | 0.910±0.002 | 1.813 | 2.195 | 0.010 |

## finance_like_returns
| Method | Coverage | Mean width | Winkler | Cal. error |
|---|---:|---:|---:|---:|
| point_lstm_split_conformal | 0.887±0.011 | 3.353 | 4.428 | 0.023 |
| point_lstm_frozen_window_cp | 0.895±0.019 | 3.491 | 4.484 | 0.026 |
| point_lstm_enbpi | 0.895±0.005 | 3.364 | 4.438 | 0.009 |
| point_lstm_aci | 0.899±0.001 | 3.469 | 4.452 | 0.002 |
| gaussian_lstm | 0.896±0.010 | 3.290 | 4.258 | 0.017 |
| quantile_lstm | 0.889±0.009 | 3.296 | 4.349 | 0.020 |
| quantile_lstm_cqr | 0.890±0.012 | 3.299 | 4.352 | 0.022 |
| historical_mean_cp | 0.894±0.007 | 3.370 | 4.446 | 0.012 |
| garch_conformal | 0.900±0.002 | 3.431 | 4.351 | 0.004 |

## real_SPY_returns
| Method | Coverage | Mean width | Winkler | Cal. error |
|---|---:|---:|---:|---:|
| point_lstm_split_conformal | 0.931±0.001 | 3.170 | 4.370 | 0.031 |
| point_lstm_frozen_window_cp | 0.847±0.001 | 2.205 | 4.500 | 0.053 |
| point_lstm_enbpi | 0.884±0.000 | 2.743 | 4.423 | 0.016 |
| point_lstm_aci | 0.890±0.000 | 2.848 | 4.041 | 0.010 |
| gaussian_lstm | 0.917±0.005 | 2.742 | 3.776 | 0.017 |
| quantile_lstm | 0.917±0.003 | 2.718 | 3.747 | 0.017 |
| quantile_lstm_cqr | 0.928±0.001 | 2.828 | 3.771 | 0.028 |
| historical_mean_cp | 0.887±0.000 | 2.766 | 4.449 | 0.013 |
| garch_conformal | 0.900±0.000 | 2.859 | 4.111 | 0.000 |
