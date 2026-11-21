# Experiment summary
Target coverage: 90%

## synthetic_regime_ar
| Method | Coverage | Mean width | Calibration error |
|---|---:|---:|---:|
| point_lstm_split_conformal | 0.933 | 1.689 | 0.033 |
| point_lstm_rolling_conformal | 0.903 | 1.525 | 0.003 |
| gaussian_lstm | 0.926 | 1.389 | 0.026 |
| quantile_lstm | 0.917 | 1.332 | 0.017 |

## finance_like_returns
| Method | Coverage | Mean width | Calibration error |
|---|---:|---:|---:|
| point_lstm_split_conformal | 0.914 | 3.811 | 0.014 |
| point_lstm_rolling_conformal | 0.914 | 3.811 | 0.014 |
| gaussian_lstm | 0.913 | 3.703 | 0.013 |
| quantile_lstm | 0.897 | 3.550 | 0.003 |
