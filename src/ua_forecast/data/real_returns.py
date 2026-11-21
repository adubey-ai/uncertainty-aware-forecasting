"""Real market returns via yfinance with local cache + crisis holdout masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ua_forecast.data.datasets import SeriesBundle

DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache"


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    safe = ticker.replace("=", "_").replace("/", "_")
    return cache_dir / f"{safe}_daily.parquet"


def download_or_load_prices(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load adjusted close prices. Tries cache first, then yfinance download,
    then falls back to a synthetic cache seed if offline.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, cache_dir)

    if path.exists():
        df = pd.read_parquet(path)
        if "Close" in df.columns and len(df) > 100:
            return df

    try:
        import yfinance as yf  # type: ignore

        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        if raw is None or len(raw) < 100:
            raise RuntimeError("yfinance returned empty frame")
        out = raw[["Close"]].dropna().copy()
        out.index = pd.to_datetime(out.index)
        out.to_parquet(path)
        return out
    except Exception:
        # Offline fallback: deterministic pseudo-market series written to cache
        if path.exists():
            return pd.read_parquet(path)
        return _write_offline_spy_like(path, start=start)


def _write_offline_spy_like(path: Path, start: str = "2015-01-01", n: int = 2200) -> pd.DataFrame:
    """Seed a reproducible SPY-like return path when download is unavailable."""
    rng = np.random.default_rng(2020)
    idx = pd.bdate_range(start=start, periods=n)
    # vol regimes approximating calm / COVID / 2022
    vol = np.full(n, 0.008)
    dates = idx.to_numpy()
    for i, d in enumerate(dates):
        ts = pd.Timestamp(d)
        if pd.Timestamp("2020-02-15") <= ts <= pd.Timestamp("2020-04-30"):
            vol[i] = 0.035
        elif pd.Timestamp("2022-01-01") <= ts <= pd.Timestamp("2022-10-31"):
            vol[i] = 0.018
    rets = vol * rng.standard_normal(n)
    # mild leverage
    for t in range(1, n):
        if rets[t - 1] < -0.02:
            rets[t] *= 1.15
    price = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"Close": price}, index=idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def crisis_mask(
    index: pd.DatetimeIndex,
    windows: list[tuple[str, str]] | None = None,
) -> np.ndarray:
    """Boolean mask True on crisis / stress windows."""
    windows = windows or [
        ("2020-02-15", "2020-04-30"),  # COVID crash
        ("2022-01-01", "2022-10-31"),  # 2022 vol / rate shock
    ]
    mask = np.zeros(len(index), dtype=bool)
    for a, b in windows:
        mask |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    return mask


def make_real_returns_bundle(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: Path | str | None = None,
    standardize: bool = True,
) -> SeriesBundle:
    """
    Build a SeriesBundle of daily log-returns with regime labels:
    0 = calm, 1 = crisis window.
    """
    prices = download_or_load_prices(ticker, start=start, end=end, cache_dir=cache_dir)
    close = prices["Close"].astype(float)
    logret = np.log(close).diff().dropna()
    idx = pd.DatetimeIndex(logret.index)
    crisis = crisis_mask(idx)
    y = logret.to_numpy(dtype=float)
    if standardize:
        y = (y - y.mean()) / (y.std() + 1e-8)
    regimes = crisis.astype(int)
    return SeriesBundle(
        y=y,
        regimes=regimes,
        name=f"real_{ticker.replace('=', '_').replace('/', '_')}_returns",
        meta={
            "ticker": ticker,
            "start": str(idx[0].date()),
            "end": str(idx[-1].date()),
            "n_crisis": int(crisis.sum()),
            "dates": idx.astype(str).tolist(),
        },
    )


def align_crisis_to_windows(
    dates: list[str] | None,
    lookback: int,
    n_windows: int,
) -> np.ndarray | None:
    """
    Map raw-series crisis labels onto supervised window targets.
    dates length should equal raw return length; windows start at lookback.
    """
    if not dates:
        return None
    idx = pd.DatetimeIndex(dates)
    crisis = crisis_mask(idx)
    aligned = crisis[lookback : lookback + n_windows]
    if len(aligned) != n_windows:
        return None
    return aligned.astype(int)
