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

from ua_forecast.conformal import (
    ACIConformal,
    CQR,
    EnbPIConformal,
    FrozenWindowConformal,
    SplitConformalRegressor,
    gaussian_interval,
)
from ua_forecast.data import (
    make_finance_like_returns,
    make_real_returns_bundle,
    make_regime_switching_ar,
    temporal_split,
    window_dataset,
)
from ua_forecast.eval import (
    aggregate_seed_metrics,
    rolling_coverage,
    summarize_intervals,
)
from ua_forecast.models import (
    GARCHConformal,
    GaussianLSTM,
    HistoricalMeanConformal,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_kwargs(cfg: dict) -> dict:
    return {
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "patience": cfg.get("patience", 8),
        "grad_clip": cfg.get("grad_clip", 1.0),
        "weight_decay": cfg.get("weight_decay", 1e-5),
    }


def run_on_bundle(bundle, cfg: dict, out_dir: Path, seed: int) -> dict:
    lookback = cfg["lookback"]
    alpha = cfg["alpha"]
    target = 1.0 - alpha
    dropout = float(cfg.get("dropout", 0.0))
    cp_window = int(cfg.get("cp_window", 200))
    aci_gamma = float(cfg.get("aci_gamma", 0.05))

    x, y = window_dataset(bundle.y, lookback=lookback, horizon=cfg["horizon"])
    splits = temporal_split(
        x,
        y,
        bundle.regimes,
        lookback=lookback,
        train_frac=cfg["train_frac"],
        val_frac=cfg["val_frac"],
    )
    y_test = splits["test"]["y"]
    reg_test = splits["test"].get("regimes")

    results: dict = {
        "dataset": bundle.name,
        "n": int(len(bundle.y)),
        "seed": seed,
        "meta": bundle.meta or {},
    }
    interval_store: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}

    # --- Point LSTM ---
    point = PointLSTM(lookback, hidden=cfg["hidden"], n_layers=cfg["n_layers"], dropout=dropout)
    train_model(
        point,
        splits["train"]["x"],
        splits["train"]["y"],
        splits["val"]["x"],
        splits["val"]["y"],
        kind="point",
        **_train_kwargs(cfg),
    )
    pred_val = predict_point(point, splits["val"]["x"])
    pred_test = predict_point(point, splits["test"]["x"])

    split_cp = SplitConformalRegressor(alpha=alpha).calibrate(splits["val"]["y"], pred_val)
    lo_s, hi_s = split_cp.predict_interval(pred_test)
    results["point_lstm_split_conformal"] = summarize_intervals(
        y_test, lo_s, hi_s, target, reg_test, alpha=alpha
    )
    interval_store["point_lstm_split_conformal"] = (lo_s, hi_s, pred_test)

    frozen = FrozenWindowConformal(alpha=alpha, window=cp_window)
    lo_f, hi_f = frozen.predict_interval(splits["val"]["y"], pred_val, pred_test)
    results["point_lstm_frozen_window_cp"] = summarize_intervals(
        y_test, lo_f, hi_f, target, reg_test, alpha=alpha
    )
    interval_store["point_lstm_frozen_window_cp"] = (lo_f, hi_f, pred_test)

    enbpi = EnbPIConformal(alpha=alpha, window=cp_window)
    lo_e, hi_e, enbpi_extra = enbpi.predict_interval(
        splits["val"]["y"], pred_val, pred_test, y_test
    )
    results["point_lstm_enbpi"] = summarize_intervals(
        y_test, lo_e, hi_e, target, reg_test, alpha=alpha
    )
    results["point_lstm_enbpi"]["enbpi"] = {k: v for k, v in enbpi_extra.items() if not isinstance(v, list)}
    interval_store["point_lstm_enbpi"] = (lo_e, hi_e, pred_test)

    aci = ACIConformal(alpha=alpha, gamma=aci_gamma, window=cp_window)
    lo_a, hi_a, aci_extra = aci.predict_interval(
        splits["val"]["y"], pred_val, pred_test, y_test
    )
    results["point_lstm_aci"] = summarize_intervals(
        y_test, lo_a, hi_a, target, reg_test, alpha=alpha
    )
    results["point_lstm_aci"]["aci"] = {
        "mean_alpha": aci_extra["mean_alpha"],
        "final_alpha": aci_extra["final_alpha"],
        "local_coverage": aci_extra["local_coverage"],
    }
    results["_aci_alpha_path"] = aci_extra["alpha_path"]
    interval_store["point_lstm_aci"] = (lo_a, hi_a, pred_test)

    # --- Gaussian LSTM ---
    gauss = GaussianLSTM(lookback, hidden=cfg["hidden"], n_layers=cfg["n_layers"], dropout=dropout)
    train_model(
        gauss,
        splits["train"]["x"],
        splits["train"]["y"],
        splits["val"]["x"],
        splits["val"]["y"],
        kind="gaussian",
        **_train_kwargs(cfg),
    )
    mu, sigma = predict_gaussian(gauss, splits["test"]["x"])
    lo_g, hi_g = gaussian_interval(mu, sigma, alpha=alpha)
    results["gaussian_lstm"] = summarize_intervals(
        y_test, lo_g, hi_g, target, reg_test, alpha=alpha, mu=mu, sigma=sigma
    )
    interval_store["gaussian_lstm"] = (lo_g, hi_g, mu)

    # --- Quantile LSTM + CQR ---
    qmodel = QuantileLSTM(lookback, hidden=cfg["hidden"], n_layers=cfg["n_layers"], dropout=dropout)
    train_model(
        qmodel,
        splits["train"]["x"],
        splits["train"]["y"],
        splits["val"]["x"],
        splits["val"]["y"],
        kind="quantile",
        **_train_kwargs(cfg),
    )
    q_val = predict_quantile(qmodel, splits["val"]["x"])
    q_test = predict_quantile(qmodel, splits["test"]["x"])
    lo_q, hi_q = q_test[:, 0], q_test[:, 2]
    results["quantile_lstm"] = summarize_intervals(
        y_test, lo_q, hi_q, target, reg_test, alpha=alpha, q_mid=q_test[:, 1]
    )
    interval_store["quantile_lstm"] = (lo_q, hi_q, q_test[:, 1])

    cqr = CQR(alpha=alpha).calibrate(splits["val"]["y"], q_val[:, 0], q_val[:, 2])
    lo_c, hi_c = cqr.predict_interval(q_test[:, 0], q_test[:, 2])
    results["quantile_lstm_cqr"] = summarize_intervals(
        y_test, lo_c, hi_c, target, reg_test, alpha=alpha, q_mid=q_test[:, 1]
    )
    results["quantile_lstm_cqr"]["qhat"] = float(cqr.qhat_ or 0.0)
    interval_store["quantile_lstm_cqr"] = (lo_c, hi_c, q_test[:, 1])

    # --- Classical baselines ---
    hist = HistoricalMeanConformal(alpha=alpha, window=cp_window)
    pred_h, lo_h, hi_h = hist.fit_predict_interval(
        splits["train"]["y"], splits["val"]["y"], y_test
    )
    results["historical_mean_cp"] = summarize_intervals(
        y_test, lo_h, hi_h, target, reg_test, alpha=alpha
    )
    interval_store["historical_mean_cp"] = (lo_h, hi_h, pred_h)

    garch = GARCHConformal(alpha=alpha, window=cp_window)
    pred_garch, lo_gc, hi_gc, garch_extra = garch.fit_predict_interval(
        splits["train"]["y"], splits["val"]["y"], y_test
    )
    results["garch_conformal"] = summarize_intervals(
        y_test, lo_gc, hi_gc, target, reg_test, alpha=alpha
    )
    results["garch_conformal"]["backend"] = garch_extra.get("backend")
    interval_store["garch_conformal"] = (lo_gc, hi_gc, pred_garch)

    # --- OOD / crisis / high-vol stress ---
    m = None
    stress_meta: dict = {}
    if reg_test is not None and len(np.unique(reg_test)) > 1:
        stress_id = int(np.max(reg_test))
        cand = reg_test == stress_id
        if np.sum(cand) >= 20:
            m = cand
            stress_meta = {"regime_id": stress_id, "rule": "max_regime_id"}
    if m is None:
        # Fallback: top-quintile |y| days in the test window (always available)
        thr = float(np.quantile(np.abs(y_test), 0.8))
        cand = np.abs(y_test) >= thr
        if np.sum(cand) >= 20:
            m = cand
            stress_meta = {"threshold_abs_y": thr, "rule": "top_quintile_abs_y"}
    if m is not None:
        ood = {**stress_meta, "n": int(np.sum(m))}
        for name, (lo, hi, _) in interval_store.items():
            ood[name] = summarize_intervals(y_test[m], lo[m], hi[m], target, alpha=alpha)
        results["ood_stress_regime"] = ood

    # Figures (primary seed only — caller may gate)
    if cfg.get("_write_figures", True):
        _write_figures(
            bundle.name,
            out_dir,
            y_test,
            reg_test,
            interval_store,
            results.get("_aci_alpha_path"),
            target,
        )

    # drop bulky path from JSON-facing results (kept in figures)
    results.pop("_aci_alpha_path", None)
    return results


def _write_figures(
    name: str,
    out_dir: Path,
    y_test: np.ndarray,
    reg_test: np.ndarray | None,
    interval_store: dict,
    alpha_path: list[float] | None,
    target: float,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(min(300, len(y_test)))
    y_plot = y_test[: len(idx)]

    panel_keys = [
        ("point_lstm_split_conformal", "Split CP"),
        ("point_lstm_enbpi", "EnbPI"),
        ("point_lstm_aci", "ACI"),
        ("quantile_lstm_cqr", "CQR"),
        ("gaussian_lstm", "Gaussian LSTM"),
        ("garch_conformal", "GARCH+CP"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, panel_keys):
        if key not in interval_store:
            ax.axis("off")
            continue
        lo, hi, mid = interval_store[key]
        ax.fill_between(idx, lo[: len(idx)], hi[: len(idx)], alpha=0.25, label="90% interval")
        ax.plot(idx, y_plot, lw=1.0, color="black", label="y")
        if mid is not None:
            ax.plot(idx, mid[: len(idx)], lw=1.0, label="pred")
        ax.set_title(f"{name}: {title}")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_intervals.png", dpi=140)
    plt.close(fig)

    # Coverage–width trade-off
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, (lo, hi, _) in interval_store.items():
        from ua_forecast.eval import coverage, mean_width

        ax.scatter(mean_width(lo, hi), coverage(y_test, lo, hi), s=70, label=key)
    ax.axhline(target, ls="--", color="gray", label=f"target={target:.2f}")
    ax.set_xlabel("Mean interval width")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Calibration–sharpness ({name})")
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_tradeoff.png", dpi=140)
    plt.close(fig)

    # Coverage over time
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for key in ["point_lstm_split_conformal", "point_lstm_enbpi", "point_lstm_aci", "quantile_lstm_cqr"]:
        if key not in interval_store:
            continue
        lo, hi, _ = interval_store[key]
        rc = rolling_coverage(y_test, lo, hi, window=50)
        ax.plot(rc, lw=1.2, label=key)
    ax.axhline(target, ls="--", color="gray")
    if reg_test is not None:
        switches = np.where(np.diff(reg_test) != 0)[0] + 1
        for s in switches:
            ax.axvline(s, color="red", alpha=0.15, lw=0.8)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Rolling PICP (w=50)")
    ax.set_xlabel("Test time")
    ax.set_title(f"Coverage over time ({name})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_coverage_time.png", dpi=140)
    plt.close(fig)

    # ACI alpha path
    if alpha_path:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(alpha_path, lw=1.2, color="tab:purple")
        ax.set_ylabel(r"$\alpha_t$")
        ax.set_xlabel("Test time")
        ax.set_title(f"ACI miscoverage path ({name})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}_aci_alpha.png", dpi=140)
        plt.close(fig)


def build_bundles(cfg: dict, seed: int) -> list:
    bundles = [
        make_regime_switching_ar(
            n=cfg["synthetic"]["n"],
            n_regimes=cfg["synthetic"]["regimes"],
            switch_prob=cfg["synthetic"]["switch_prob"],
            seed=seed,
            gradual=False,
        ),
    ]
    if cfg.get("include_gradual", True):
        bundles.append(
            make_regime_switching_ar(
                n=cfg["synthetic"]["n"],
                n_regimes=cfg["synthetic"]["regimes"],
                switch_prob=cfg["synthetic"]["switch_prob"],
                seed=seed + 11,
                gradual=True,
                transition_len=int(cfg["synthetic"].get("transition_len", 25)),
            )
        )
    bundles.append(
        make_finance_like_returns(
            n=cfg["financial"]["n"],
            vol_regimes=cfg["financial"]["vol_regimes"],
            seed=seed,
            leverage=bool(cfg["financial"].get("leverage", True)),
        )
    )
    if cfg.get("include_real", False):
        real_cfg = cfg.get("real", {})
        try:
            bundles.append(
                make_real_returns_bundle(
                    ticker=real_cfg.get("ticker", "SPY"),
                    start=real_cfg.get("start", "2015-01-01"),
                    end=real_cfg.get("end"),
                    cache_dir=ROOT / "data" / "cache",
                )
            )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] real returns unavailable: {exc}")
    return bundles


def aggregate_results(per_seed: list[dict]) -> dict:
    """Aggregate list of per-seed dataset->method dicts."""
    if not per_seed:
        return {}
    datasets = per_seed[0].keys()
    out: dict = {}
    for ds in datasets:
        methods = [
            k
            for k, v in per_seed[0][ds].items()
            if isinstance(v, dict) and "coverage" in v
        ]
        out[ds] = {
            "n": per_seed[0][ds].get("n"),
            "meta": per_seed[0][ds].get("meta", {}),
        }
        for m in methods:
            seed_metrics = [run[ds][m] for run in per_seed if m in run[ds]]
            out[ds][m] = aggregate_seed_metrics(seed_metrics)
            # also keep last-seed scalar mirror for SUMMARY simplicity
            out[ds][m]["last"] = {k: seed_metrics[-1][k] for k in seed_metrics[-1] if isinstance(seed_metrics[-1][k], (int, float, str))}
        # ood from last seed if present
        if "ood_stress_regime" in per_seed[-1][ds]:
            out[ds]["ood_stress_regime"] = per_seed[-1][ds]["ood_stress_regime"]
    return out


def write_summary(all_results: dict, out_dir: Path, target: float, multi_seed: bool) -> None:
    lines = ["# Experiment summary\n", f"Target coverage: {target:.0%}\n"]
    if multi_seed:
        lines.append("\nMetrics are mean ± stderr across seeds (where available).\n")
    for ds, block in all_results.items():
        lines.append(f"\n## {ds}\n")
        lines.append(
            "| Method | Coverage | Mean width | Winkler | Cal. error |\n"
        )
        lines.append("|---|---:|---:|---:|---:|\n")
        for method, stats in block.items():
            if not isinstance(stats, dict):
                continue
            if multi_seed and "coverage" in stats and isinstance(stats["coverage"], dict):
                cov = stats["coverage"]["mean"]
                cov_e = stats["coverage"]["stderr"]
                width = stats["mean_width"]["mean"]
                wink = stats.get("winkler", {}).get("mean", float("nan"))
                cal = stats["calibration_error"]["mean"]
                lines.append(
                    f"| {method} | {cov:.3f}±{cov_e:.3f} | {width:.3f} | {wink:.3f} | {cal:.3f} |\n"
                )
            elif "coverage" in stats and isinstance(stats["coverage"], (int, float)):
                wink = stats.get("winkler", float("nan"))
                lines.append(
                    f"| {method} | {stats['coverage']:.3f} | {stats['mean_width']:.3f} | "
                    f"{wink:.3f} | {stats['calibration_error']:.3f} |\n"
                )
    summary_path = out_dir / "SUMMARY.md"
    summary_path.write_text("".join(lines))
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--out", type=str, default=str(ROOT / "results"))
    parser.add_argument("--fast", action="store_true", help="Fewer epochs / shorter series")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fast:
        cfg["epochs"] = min(cfg.get("epochs", 40), 8)
        cfg["synthetic"]["n"] = min(cfg["synthetic"]["n"], 1500)
        cfg["financial"]["n"] = min(cfg["financial"]["n"], 1200)
        cfg["seeds"] = cfg.get("seeds", [cfg.get("seed", 42)])[:1]
        cfg["include_gradual"] = False
        cfg["include_real"] = False
        cfg["patience"] = 4

    seeds = cfg.get("seeds") or [cfg.get("seed", 42)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_results: list[dict] = []
    for i, seed in enumerate(seeds):
        print(f"==> Seed {seed} ({i + 1}/{len(seeds)})", flush=True)
        set_seed(int(seed))
        cfg_run = dict(cfg)
        cfg_run["_write_figures"] = i == len(seeds) - 1  # figures from last seed
        seed_block = {}
        for bundle in build_bundles(cfg, int(seed)):
            print(f"    Running on {bundle.name} ...", flush=True)
            seed_block[bundle.name] = run_on_bundle(bundle, cfg_run, out_dir, int(seed))
        per_seed_results.append(seed_block)

    multi = len(seeds) > 1
    if multi:
        all_results = aggregate_results(per_seed_results)
        # also store raw per-seed
        raw_path = out_dir / "metrics_per_seed.json"
        with open(raw_path, "w") as f:
            json.dump(per_seed_results, f, indent=2)
        print(f"Wrote {raw_path}")
    else:
        all_results = per_seed_results[0]

    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out_json}")

    write_summary(all_results, out_dir, 1.0 - cfg["alpha"], multi_seed=multi)


if __name__ == "__main__":
    main()
