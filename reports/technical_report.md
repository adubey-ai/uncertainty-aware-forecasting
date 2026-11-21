# Uncertainty-Aware Forecasting for Non-Stationary Time Series

**Author:** Adarsh Dubey  
**Status:** Research technical report (2026)  
**Code:** https://github.com/adubey-ai/uncertainty-aware-forecasting  

## 1. Research question

How can forecasting systems produce **calibrated uncertainty estimates** when **regime shifts** break the exchangeability assumptions behind classical conformal prediction?

Point forecasts are insufficient for sequential decisions in finance. We need intervals whose empirical coverage stays near a nominal level (here 90%) under volatility and mean non-stationarity — and we need methods that **adapt online** when fixed calibration sets go stale.

## 2. Related methods (implemented)

| Method | Role in this study |
|---|---|
| Split conformal | Exchangeability baseline (fixed val residual quantile) |
| Frozen-window CP | Offline rolling residual quantile (no test updates) |
| **EnbPI** | Sliding residual buffer updated after each test step |
| **ACI** | Online \(\alpha_t\) from coverage feedback (Gibbs & Candès) |
| Gaussian / Quantile LSTM | Probabilistic deep baselines |
| **CQR** | Conformalized quantile regression (asymmetric scores) |
| Historical mean + CP | Classical nonparametric baseline |
| **GARCH(1,1) + CP** | Finance-standard vol model with conformalized scores |

## 3. Protocol

### 3.1 Data

1. **Synthetic abrupt regime AR** — piecewise AR(1) with mean/φ/σ switches.  
2. **Gradual hardness ladder** — same DGP with linear parameter transitions.  
3. **Finance-like returns** — discrete vol regimes + leverage.  
4. **Real SPY log-returns** (cached; `yfinance` when online) with COVID/2022 crisis labels; high-\(|y|\) stress fallback on the chronological test window.

Chronological 60/20/20 train/val/test. Conformal calibration uses validation only; EnbPI/ACI update using revealed test labels in the sequential evaluation protocol (standard for online CP).

### 3.2 Metrics

Coverage, mean width, calibration error, **Winkler** score, Gaussian **NLL/CRPS**, under/over-coverage, regime / post-switch coverage, coverage-over-time, ACI \(\alpha_t\) path. Multi-seed: **5 seeds** (`configs/paper.yaml`), mean ± stderr.

## 4. Results (paper protocol)

Target coverage: 90%. Tables below are mean ± stderr across seeds from `results/SUMMARY.md`.

### 4.1 Synthetic abrupt regimes

| Method | Coverage | Mean width | Winkler | Cal. error |
|---|---:|---:|---:|---:|
| Split CP | 0.850±0.020 | 1.356 | 2.157 | 0.050 |
| Frozen-window CP | 0.908±0.026 | 1.683 | 2.154 | 0.051 |
| **EnbPI** | 0.894±0.012 | 1.580 | 2.049 | 0.018 |
| **ACI** | **0.899±0.001** | 1.605 | 2.047 | **0.002** |
| Gaussian LSTM | 0.908±0.006 | 1.559 | 1.950 | 0.012 |
| Quantile LSTM | 0.890±0.011 | 1.499 | 1.963 | 0.021 |
| **CQR** | 0.898±0.010 | 1.539 | 1.957 | 0.014 |
| Historical CP | 0.898±0.008 | 1.924 | 2.461 | 0.012 |
| GARCH+CP | 0.906±0.003 | 2.215 | 2.654 | 0.006 |

**Takeaway:** Under abrupt switches, fixed split CP under-covers (85%). ACI locks onto the target (89.9%, cal. err 0.002) with low seed variance; EnbPI recovers most of the gap.

### 4.2 High-volatility OOD slice (abrupt synthetic, last seed)

On the highest-vol regime in test, split CP coverage collapses to **66.7%**; EnbPI **79.3%**; ACI **88.7%**. Adaptive conformal is the difference between a broken and a usable interval under stress.

### 4.3 Finance-like returns

ACI again hits **0.899±0.001** coverage (cal. err 0.002). GARCH+CP is well calibrated (0.900±0.002) with competitive Winkler. CQR improves raw quantile calibration vs pinball-only intervals.

### 4.4 Real SPY returns

| Method | Coverage | Mean width | Winkler | Cal. error |
|---|---:|---:|---:|---:|
| Split CP | 0.931±0.001 | 3.170 | 4.370 | 0.031 |
| EnbPI | 0.884±0.000 | 2.743 | 4.423 | 0.016 |
| ACI | 0.890±0.000 | 2.848 | **4.041** | 0.010 |
| CQR | 0.928±0.001 | 2.828 | 3.771 | 0.028 |
| GARCH+CP | **0.900±0.000** | 2.859 | 4.111 | **0.000** |
| Quantile LSTM | 0.917±0.003 | 2.718 | **3.747** | 0.017 |

**Takeaway:** On real equity returns, GARCH+CP hits the nominal level exactly across seeds; ACI is near-target with the best Winkler among conformal point-forecast wrappers; raw split CP over-covers with wider bands.

## 5. Failure analysis under non-stationarity

1. **Exchangeability break:** Split CP’s validation quantile is stale after a regime switch → systematic under-coverage on stress regimes.  
2. **Frozen ≠ EnbPI:** Freezing the residual buffer (previous “rolling” baseline) does not adapt; true EnbPI updates after each step and moves coverage toward the target.  
3. **ACI tracks the gap:** \(\alpha_t\) paths (see `results/figures/*_aci_alpha.png`) expand intervals after misses and shrink after runs of coverage — visible as stable rolling PICP in `*_coverage_time.png`.  
4. **Sharpness vs validity:** Gaussian/quantile LSTMs can be sharper (lower Winkler) but need CQR or ACI-style wrappers when calibration drifts.

## 6. Limitations and next steps

- Horizon-1 univariate only; multi-step / path conformal remains open.  
- Crisis calendar labels on SPY often fall outside the final chronological test window; we report high-\(|y|\) stress as a complementary OOD slice.  
- Single backbone family (LSTM); method claims are intended to be backbone-agnostic.  
- Planned: AgACI, locally weighted / regime-weighted conformal, decision-aware evaluation (risk constraints).

## 7. Reproducibility

```bash
pip install -e ".[finance,dev]"
python scripts/run_experiments.py --config configs/fast.yaml    # smoke
python scripts/run_experiments.py --config configs/paper.yaml   # 5-seed + real
pytest -q
```

Configs: `configs/{fast,default,paper}.yaml`. Cached SPY parquet under `data/cache/` enables offline runs.
