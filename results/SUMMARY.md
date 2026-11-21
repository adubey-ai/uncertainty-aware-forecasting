# Experiment summary
Target coverage: 90%

## synthetic_regime_ar
| Method | Coverage | Mean width | Calibration error |
|---|---:|---:|---:|
| point_lstm_split_conformal | 0.678 | 0.687 | 0.222 |
| point_lstm_rolling_conformal | 0.675 | 0.669 | 0.225 |
| gaussian_lstm | 0.881 | 1.507 | 0.019 |
| quantile_lstm | 0.878 | 1.447 | 0.022 |

## finance_like_returns
| Method | Coverage | Mean width | Calibration error |
|---|---:|---:|---:|
| point_lstm_split_conformal | 0.983 | 4.948 | 0.083 |
| point_lstm_rolling_conformal | 0.979 | 4.688 | 0.079 |
| gaussian_lstm | 0.919 | 2.962 | 0.019 |
| quantile_lstm | 0.945 | 3.671 | 0.045 |
