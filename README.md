# Uncertainty-Aware Forecasting for Non-Stationary Time Series

**Probabilistic ML / Time Series** · 2026–present

Research question: *How can forecasting systems produce calibrated uncertainty estimates under regime shifts and distribution shift?*

This repository studies **calibrated sequential prediction intervals** when exchangeability fails — comparing probabilistic deep models, classical GARCH, and conformal wrappers (split, frozen-window, **EnbPI**, **ACI**, **CQR**) on synthetic hardness ladders, finance-like returns, and real equity returns with crisis holdouts.

## Methods

| Family | Method | What it does |
|---|---|---|
| Point + CP | Split conformal | Absolute residual quantile on validation (fixed) |
| Point + CP | Frozen-window CP | Last-`W` val residuals only (offline baseline; not EnbPI) |
| Point + CP | **EnbPI** | Sliding residual buffer **updated online** on the test stream |
| Point + CP | **ACI** | Online \(\alpha_t\) adaptation from coverage feedback |
| Probabilistic | Gaussian LSTM | NLL mean/variance + analytic Gaussian intervals |
| Probabilistic | Quantile LSTM | Pinball 5%/50%/95% intervals |
| Probabilistic + CP | **CQR** | Conformalized quantile regression (asymmetric scores) |
| Classical | Historical mean + CP | Rolling mean with online residual conformal |
| Classical | **GARCH(1,1) + CP** | Parametric vol recursion + conformalized standardized residuals |

## Data

1. **Synthetic regime-switching AR** — abrupt mean/AR/vol switches  
2. **Gradual hardness ladder** — same DGP with linear parameter transitions  
3. **Finance-like returns** — discrete vol regimes + optional leverage  
4. **Real returns (SPY)** — cached daily log-returns; crisis masks for COVID (2020-Q1) and 2022 vol spike  

All generators are seed-deterministic. Real data prefers `yfinance`, then local parquet cache, then an offline SPY-like seed.

## Evaluation

- Coverage (PICP), mean width, calibration error  
- **Winkler** interval score; **CRPS** / **NLL** (Gaussian); quantile CRPS proxy  
- Under/over-coverage rates  
- Regime-conditional + **post-switch** coverage  
- **Coverage-over-time** and ACI \(\alpha_t\) paths  
- Multi-seed mean ± stderr (`configs/paper.yaml`)

Target coverage defaults to **90%** (`alpha=0.1`).

## Quick start

**Prefer the project `.venv`** (avoid broken conda `liblapack` torch installs):

```bash
cd uncertainty-aware-forecasting
source .venv/bin/activate
pip install -e ".[finance,dev]"

# Smoke
python scripts/run_experiments.py --config configs/fast.yaml

# Full single-seed (synthetic + finance-like + gradual)
python scripts/run_experiments.py --config configs/default.yaml

# Paper protocol (5 seeds + real SPY)
python scripts/run_experiments.py --config configs/paper.yaml

pytest -q
```

Outputs:

- `results/metrics.json` — tables (aggregated if multi-seed)  
- `results/metrics_per_seed.json` — raw seeds (paper mode)  
- `results/SUMMARY.md` — markdown summary  
- `results/figures/*.png` — intervals, trade-offs, coverage-over-time, ACI path  
- `reports/technical_report.md` — write-up  

## Project layout

```
src/ua_forecast/
  data/        # synthetic hardness ladder + real returns cache
  models/      # LSTM forecasters + GARCH / historical baselines
  conformal/   # split, frozen, EnbPI, ACI, CQR
  eval/        # coverage, Winkler, CRPS, temporal diagnostics
configs/{fast,default,paper}.yaml
scripts/run_experiments.py
```

## Citation

```
Adarsh Dubey. Uncertainty-Aware Forecasting for Non-Stationary Time Series (2026).
https://github.com/adubey-ai/uncertainty-aware-forecasting
```

## License

MIT
