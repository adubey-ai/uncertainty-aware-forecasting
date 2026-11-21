# Uncertainty-Aware Forecasting for Non-Stationary Time Series

**Probabilistic ML / Time Series** · 2026–present

Research question: *How can forecasting systems produce calibrated uncertainty estimates under regime shifts and distribution shift?*

This repository compares:

1. **Deep forecasting baselines** — point LSTM  
2. **Probabilistic models** — Gaussian LSTM (NLL) and quantile LSTM  
3. **Conformal-prediction wrappers** — split conformal and rolling residual conformal  

on **synthetic regime-switching** and **finance-like** non-stationary series.

## Method

| Component | Description |
|---|---|
| Data | Piecewise AR(1) with abrupt mean/vol switches; finance-like cumulative returns with volatility regimes |
| Split | Strictly chronological train / val / test (no shuffle) |
| Point + CP | LSTM point forecast + split / rolling conformal intervals |
| Probabilistic | Gaussian predictive intervals; pinball quantile intervals (5%/50%/95%) |
| OOD stress | Coverage evaluated on the highest-volatility regime in the test window |

## Evaluation

- Empirical **coverage** (PICP)  
- **Mean interval width** / sharpness  
- **Calibration error** = \|coverage − target\|  
- **Regime-conditional coverage** (distribution-shift diagnostic)  

Target coverage defaults to **90%** (`alpha=0.1`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# smoke run
python scripts/run_experiments.py --fast

# full run (default config)
python scripts/run_experiments.py

# tests
pytest -q
```

Outputs:

- `results/metrics.json` — numeric tables  
- `results/SUMMARY.md` — markdown summary  
- `results/figures/*.png` — interval plots and coverage–width trade-offs  
- `reports/technical_report.md` — short write-up  

## Preliminary results

Run `python scripts/run_experiments.py` to regenerate. Typical pattern on regime-switching data:

- Split conformal often lands near target coverage with wider intervals.  
- Gaussian LSTM can be sharper but under-cover under regime shift.  
- Quantile LSTM is competitive when pinball training is stable.  
- Stress-regime (OOD) coverage usually degrades vs pooled test coverage — the core non-stationarity challenge.

Exact numbers are experiment outputs in `results/`.

## Project layout

```
src/ua_forecast/
  data/        # synthetic + finance-like generators
  models/      # point / Gaussian / quantile LSTMs
  conformal/   # split + rolling conformal wrappers
  eval/        # coverage, width, calibration, regime metrics
scripts/run_experiments.py
configs/default.yaml
reports/technical_report.md
```

## Citation

If you use this codebase, please cite the repository:

```
Adarsh Dubey. Uncertainty-Aware Forecasting for Non-Stationary Time Series (2026).
https://github.com/adubey-ai/uncertainty-aware-forecasting
```

## License

MIT
