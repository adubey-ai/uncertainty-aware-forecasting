# Uncertainty-Aware Forecasting for Non-Stationary Time Series

**Author:** Adarsh Dubey  
**Status:** Preliminary technical report (2026)  
**Code:** https://github.com/adubey-ai/uncertainty-aware-forecasting  

## 1. Research question

How can forecasting systems produce **calibrated uncertainty estimates** under **regime shifts** and **distribution shift**?

Point forecasts alone are insufficient for sequential decision-making in finance and other non-stationary domains. We need intervals whose empirical coverage remains close to a nominal level when the data-generating process switches volatility or mean dynamics.

## 2. Method

### 2.1 Data

1. **Synthetic regime-switching AR(1):** three regimes with distinct means, autoregressive coefficients, and noise scales; switches occur with fixed probability.  
2. **Finance-like series:** cumulative normalized log-return process with discrete volatility regimes and mild leverage-like transitions.

Both generators are deterministic given a seed (no external market download required for reproducibility).

### 2.2 Models

| Family | Model | Uncertainty |
|---|---|---|
| Deep baseline | Point LSTM | Split conformal; rolling residual conformal |
| Probabilistic | Gaussian LSTM | Analytic Gaussian predictive interval |
| Probabilistic | Quantile LSTM | Pinball loss for 5%/50%/95% |

Training uses chronological train/validation splits. Conformal methods calibrate exclusively on validation residuals (no test leakage).

### 2.3 Evaluation protocol

For nominal coverage \(1-\alpha\) (default 90%):

- Empirical coverage  
- Mean interval width (sharpness)  
- Calibration error \(\lvert \widehat{\mathrm{cov}} - (1-\alpha)\rvert\)  
- Regime-conditional coverage on the test window  
- OOD stress: metrics restricted to the highest-volatility regime in test  

## 3. Results (auto-generated from `results/metrics.json`)

Target coverage: 90%.

### synthetic_regime_ar

| Method | Coverage | Mean width | Cal. error |
|---|---:|---:|---:|
| `point_lstm_split_conformal` | 0.933 | 1.689 | 0.033 |
| `point_lstm_rolling_conformal` | 0.903 | 1.525 | 0.003 |
| `gaussian_lstm` | 0.926 | 1.389 | 0.026 |
| `quantile_lstm` | 0.917 | 1.332 | 0.017 |

OOD stress regime id=2:

- `point_lstm_split_conformal` coverage=0.894, width=1.689
- `gaussian_lstm` coverage=0.913, width=1.868
- `quantile_lstm` coverage=0.913, width=1.835

### finance_like_returns

| Method | Coverage | Mean width | Cal. error |
|---|---:|---:|---:|
| `point_lstm_split_conformal` | 0.914 | 3.811 | 0.014 |
| `point_lstm_rolling_conformal` | 0.914 | 3.811 | 0.014 |
| `gaussian_lstm` | 0.913 | 3.703 | 0.013 |
| `quantile_lstm` | 0.897 | 3.550 | 0.003 |

OOD stress regime id=2:

- `point_lstm_split_conformal` coverage=0.905, width=3.811
- `gaussian_lstm` coverage=0.903, width=3.848
- `quantile_lstm` coverage=0.886, width=3.653

## 4. Limitations and next steps

- Current conformal scores use absolute residuals (symmetric intervals).  
- No online adaptation of conformal quantiles on the test stream (intentional offline protocol).  
- Finance-like generator is stylized; next: public market series + regime labels / realized-vol proxies.  
- Planned extensions: asymmetric conformal scores, ACI / EnbPI variants, and conformalized quantile regression with explicit under/over-coverage reporting.

## 5. Reproducibility

```bash
pip install -e .
python scripts/run_experiments.py --config configs/default.yaml
pytest -q
```

All randomness is controlled by `seed` in `configs/default.yaml`.
