#!/usr/bin/env python3
"""Run uncertainty-aware forecasting experiments and write results + figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ua_forecast.conformal import RollingConformalWrapper, SplitConformalRegressor, gaussian_interval
from ua_forecast.data import (
    make_finance_like_returns,
    make_regime_switching_ar,
    temporal_split,
    window_dataset,
)
from ua_forecast.eval import summarize_intervals
from ua_forecast.models import (
    GaussianLSTM,
    PointLSTM,
    QuantileLSTM,
    predict_gaussian,
    predict_point,
    predict_quantile,
    train_model,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_on_bundle(bundle, cfg: dict, out_dir: Path) -> dict:
    lookback = cfg["lookback"]
    alpha = cfg["alpha"]
    target = 1.0 - alpha

    x, y = window_dataset(bundle.y, lookback=lookback, horizon=cfg["horizon"])
    splits = temporal_split(
        x, y, bundle.regimes, lookback=lookback, train_frac=cfg["train_frac"], val_frac=cfg["val_frac"]
    )

    results = {"dataset": bundle.name, "n": int(len(bundle.y))}

    # --- Point LSTM + split conformal ---
    point = PointLSTM(lookback, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    train_model(
        point,
        splits["train"]["x"],
        splits["train"]["y"],
        splits["val"]["x"],
        splits["val"]["y"],
        kind="point",
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
    )
    pred_val = predict_point(point, splits["val"]["x"])
    pred_test = predict_point(point, splits["test"]["x"])
    split_cp = SplitConformalRegressor(alpha=alpha).calibrate(splits["val"]["y"], pred_val)
    lo_s, hi_s = split_cp.predict_interval(pred_test)
    results["point_lstm_split_conformal"] = summarize_intervals(
        splits["test"]["y"], lo_s, hi_s, target, splits["test"]["regimes"]
    )

    rolling = RollingConformalWrapper(alpha=alpha, window=200)
    lo_r, hi_r = rolling.predict_interval(splits["val"]["y"], pred_val, pred_test)
    results["point_lstm_rolling_conformal"] = summarize_intervals(
        splits["test"]["y"], lo_r, hi_r, target, splits["test"]["regimes"]
    )

    # --- Gaussian probabilistic LSTM ---
    gauss = GaussianLSTM(lookback, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    train_model(
        gauss,
        splits["train"]["x"],
        splits["train"]["y"],
        splits["val"]["x"],
        splits["val"]["y"],
        kind="gaussian",
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
    )
    mu, sigma = predict_gaussian(gauss, splits["test"]["x"])
    lo_g, hi_g = gaussian_interval(mu, sigma, alpha=alpha)
    results["gaussian_lstm"] = summarize_intervals(
        splits["test"]["y"], lo_g, hi_g, target, splits["test"]["regimes"]
    )

    # --- Quantile LSTM ---
    qmodel = QuantileLSTM(lookback, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    train_model(
        qmodel,
        splits["train"]["x"],
        splits["train"]["y"],
        splits["val"]["x"],
        splits["val"]["y"],
        kind="quantile",
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
    )
    qpred = predict_quantile(qmodel, splits["test"]["x"])
    lo_q, hi_q = qpred[:, 0], qpred[:, 2]
    results["quantile_lstm"] = summarize_intervals(
        splits["test"]["y"], lo_q, hi_q, target, splits["test"]["regimes"]
    )

    # --- OOD / regime-shift stress: evaluate only on highest-volatility regime in test ---
    test_reg = splits["test"]["regimes"]
    if len(np.unique(test_reg)) > 1:
        # treat last regime id as "stress" if present, else max-id regime
        stress_id = int(np.max(test_reg))
        m = test_reg == stress_id
        if np.sum(m) >= 20:
            results["ood_stress_regime"] = {
                "regime_id": stress_id,
                "point_lstm_split_conformal": summarize_intervals(
                    splits["test"]["y"][m], lo_s[m], hi_s[m], target
                ),
                "gaussian_lstm": summarize_intervals(
                    splits["test"]["y"][m], lo_g[m], hi_g[m], target
                ),
                "quantile_lstm": summarize_intervals(
                    splits["test"]["y"][m], lo_q[m], hi_q[m], target
                ),
            }

    # Figures
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(min(300, len(splits["test"]["y"])))
    y_plot = splits["test"]["y"][: len(idx)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        (axes[0, 0], "Split conformal", lo_s, hi_s, pred_test),
        (axes[0, 1], "Rolling conformal", lo_r, hi_r, pred_test),
        (axes[1, 0], "Gaussian LSTM", lo_g, hi_g, mu),
        (axes[1, 1], "Quantile LSTM", lo_q, hi_q, qpred[:, 1]),
    ]
    for ax, title, lo, hi, mid in panels:
        ax.fill_between(idx, lo[: len(idx)], hi[: len(idx)], alpha=0.25, label="90% interval")
        ax.plot(idx, y_plot, lw=1.0, label="y", color="black")
        ax.plot(idx, mid[: len(idx)], lw=1.0, label="pred")
        ax.set_title(f"{bundle.name}: {title}")
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{bundle.name}_intervals.png", dpi=140)
    plt.close(fig)

    # Coverage vs width scatter summary plot
    methods = ["point_lstm_split_conformal", "point_lstm_rolling_conformal", "gaussian_lstm", "quantile_lstm"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        s = results[m]
        ax.scatter(s["mean_width"], s["coverage"], s=70, label=m)
    ax.axhline(target, ls="--", color="gray", label=f"target={target:.2f}")
    ax.set_xlabel("Mean interval width")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Calibration–sharpness trade-off ({bundle.name})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{bundle.name}_tradeoff.png", dpi=140)
    plt.close(fig)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--out", type=str, default=str(ROOT / "results"))
    parser.add_argument("--fast", action="store_true", help="Fewer epochs for a quick smoke run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.fast:
        cfg["epochs"] = 8
        cfg["synthetic"]["n"] = 1500
        cfg["financial"]["n"] = 1200

    set_seed(cfg["seed"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = [
        make_regime_switching_ar(
            n=cfg["synthetic"]["n"],
            n_regimes=cfg["synthetic"]["regimes"],
            switch_prob=cfg["synthetic"]["switch_prob"],
            seed=cfg["seed"],
        ),
        make_finance_like_returns(
            n=cfg["financial"]["n"],
            vol_regimes=cfg["financial"]["vol_regimes"],
            seed=cfg["seed"],
        ),
    ]

    all_results = {}
    for bundle in bundles:
        print(f"==> Running on {bundle.name} ...")
        all_results[bundle.name] = run_on_bundle(bundle, cfg, out_dir)

    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out_json}")

    # Markdown summary table
    lines = ["# Experiment summary\n", f"Target coverage: {1.0 - cfg['alpha']:.0%}\n"]
    for ds, block in all_results.items():
        lines.append(f"\n## {ds}\n")
        lines.append("| Method | Coverage | Mean width | Calibration error |\n")
        lines.append("|---|---:|---:|---:|\n")
        for method, stats in block.items():
            if not isinstance(stats, dict) or "coverage" not in stats:
                continue
            lines.append(
                f"| {method} | {stats['coverage']:.3f} | {stats['mean_width']:.3f} | {stats['calibration_error']:.3f} |\n"
            )
    summary_path = out_dir / "SUMMARY.md"
    summary_path.write_text("".join(lines))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
