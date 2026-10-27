"""Market data: live loaders (Yahoo Finance + FRED + ESG csv) and a synthetic
regime-switching generator that contains 2008-like and 2020-like crises.

The synthetic generator exists so the whole pipeline is runnable/testable
offline.  For the real project, call `load_real_market` and report results on
real data; keep the synthetic one for unit tests and controlled experiments.
"""
from __future__ import annotations
import hashlib
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CRISES = {
    "GFC_2008": ("2008-09-01", "2009-06-30"),
    "COVID_2020": ("2020-02-20", "2020-06-30"),
}


@dataclass
class Market:
    prices: pd.DataFrame      # (T, N) adjusted close
    returns: pd.DataFrame     # (T, N) simple daily returns
    macro: pd.DataFrame       # (T, k) incl. column 'VIX'
    esg: pd.Series            # (N,) score in [0,100], HIGHER = better
    source: str
    meta: dict = field(default_factory=dict)   # audit trail (dropped tickers, coverage, ESG provenance)


def _finish(prices, macro, esg, source, meta=None, min_macro_coverage=0.95):
    """Align everything on the trading calendar WITHOUT any back-fill.

    * prices: forward-fill only (holidays); tickers still containing NaN afterwards
      (listed later than the start date) are dropped and REPORTED in meta['dropped'].
    * macro: forward-fill only.  A column whose coverage over the price calendar is
      below `min_macro_coverage` (e.g. FRED's ICE-BofA series, truncated to the last
      3 years since 2026-04) is dropped and reported; it is never back-filled,
      because back-filling would copy FUTURE values into the past (look-ahead).
    """
    meta = dict(meta or {})
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None) if getattr(prices.index, "tz", None) is not None \
        else pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all").ffill()
    bad = [c for c in prices.columns if prices[c].isna().any()]
    if bad:
        warnings.warn(f"Dropping tickers with missing history (listed after start?): {bad}")
    meta["dropped_tickers"] = bad
    prices = prices.drop(columns=bad)
    if prices.shape[1] == 0:
        raise ValueError("no ticker has a complete price history over the requested range")
    rets = prices.pct_change().dropna()
    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()
    macro = macro[~macro.index.duplicated(keep="last")]
    macro = macro.reindex(macro.index.union(rets.index)).ffill().reindex(rets.index)
    cov = macro.notna().mean()
    meta["macro_coverage"] = {k: float(v) for k, v in cov.items()}
    drop_m = [c for c in macro.columns if cov[c] < min_macro_coverage]
    if drop_m:
        warnings.warn(f"Dropping macro series with insufficient history: {drop_m} (coverage {cov[drop_m].round(3).to_dict()})")
    meta["dropped_macro"] = drop_m
    macro = macro.drop(columns=drop_m)
    if "VIX" not in macro.columns:
        raise ValueError("VIX series missing/insufficient: the time-varying constraints need it")
    macro = macro.dropna()                          # leading rows before the first macro observation
    rets = rets.loc[macro.index]
    esg = esg.reindex(prices.columns)
    n_missing = int(esg.isna().sum())
    meta["esg_missing"] = [c for c in prices.columns if pd.isna(esg[c])]
    if n_missing:
        warnings.warn(f"{n_missing} tickers have no ESG score -> filled with the median (documented in meta).")
    esg = esg.fillna(esg.median())
    return Market(prices.loc[rets.index], rets, macro, esg, source, meta)


# --------------------------------------------------------------------------
# synthetic
# --------------------------------------------------------------------------
def synthetic_market(start="2004-01-01", end="2023-12-31", n_assets=16, seed=0) -> Market:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T = len(idx)
    # regime: 0 calm, 1 stress.  Markov chain + forced crisis windows
    stress = np.zeros(T, dtype=int)
    for i in range(1, T):
        p_stay = 0.985 if stress[i - 1] == 0 else 0.95
        stress[i] = stress[i - 1] if rng.random() < p_stay else 1 - stress[i - 1]
    for a, b in CRISES.values():
        stress[(idx >= a) & (idx <= b)] = 1
    vol_scale = np.where(stress == 1, 2.8, 1.0)
    vol_scale = pd.Series(vol_scale).ewm(span=5).mean().values

    n_sec = 4
    sector = np.arange(n_assets) % n_sec
    beta = rng.uniform(0.6, 1.4, n_assets)
    mkt = rng.standard_t(6, T) * 0.008 * vol_scale + np.where(stress == 1, -0.0010, 0.00045)
    sec = rng.standard_t(6, (T, n_sec)) * 0.004 * vol_scale[:, None]
    alpha = rng.normal(0.0002, 0.0002, n_assets)
    idio = rng.standard_t(6, (T, n_assets)) * rng.uniform(0.006, 0.012, n_assets) * np.sqrt(vol_scale)[:, None]
    rets = mkt[:, None] * beta + sec[:, sector] + idio + alpha
    tickers = [f"S{i:02d}" for i in range(n_assets)]
    r = pd.DataFrame(rets, index=idx, columns=tickers)
    prices = 100 * (1 + r).cumprod()
    vix = 13 + 14 * (vol_scale - 1) / 1.8 * 1.0 + rng.normal(0, 0.8, T)
    macro = pd.DataFrame({"VIX": np.clip(vix, 9, None),
                          "T10Y2Y": 1.0 - 1.5 * stress + rng.normal(0, 0.05, T).cumsum() * 0.02,
                          "HY_OAS": 3.5 + 5 * stress + rng.normal(0, 0.1, T)}, index=idx)
    esg = pd.Series(rng.uniform(20, 95, n_assets), index=tickers)
    return _finish(prices, macro, esg, "synthetic")


# --------------------------------------------------------------------------
# real data.  Two-step so the analysis is reproducible OFFLINE:
#   (1) `python scripts/fetch_data.py`  -> data/raw/{prices,macro}.csv (+ provenance json)   [needs internet]
#   (2) `load_real_market(...)`         -> reads data/raw/*.csv; falls back to the network only if absent
# --------------------------------------------------------------------------
FRED_SERIES = {"VIX": "VIXCLS", "T10Y2Y": "T10Y2Y", "HY_OAS": "BAMLH0A0HYM2"}
RAW_DIR = "data/raw"


def parse_fred_csv(df: pd.DataFrame, series_id: str) -> pd.Series:
    """Robust to FRED's header change (DATE -> observation_date) and '.' for missing."""
    date_col = next((c for c in df.columns if c.lower() in ("observation_date", "date")), df.columns[0])
    val_col = series_id if series_id in df.columns else [c for c in df.columns if c != date_col][0]
    s = pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series_id
    return s.dropna()


def _fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    return parse_fred_csv(pd.read_csv(url), series_id)


def extract_close(raw: pd.DataFrame, tickers) -> pd.DataFrame:
    """Normalise yfinance output (flat / MultiIndex (Price,Ticker) / MultiIndex (Ticker,Price),
    single or multiple tickers, with or without auto_adjust) to a (date x ticker) Close frame."""
    tickers = list(tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        lv0, lv1 = raw.columns.get_level_values(0), raw.columns.get_level_values(1)
        field_name = "Adj Close" if ("Adj Close" in set(lv0) | set(lv1)) else "Close"
        if field_name in set(lv0):
            px = raw[field_name]
        elif field_name in set(lv1):
            px = raw.xs(field_name, axis=1, level=1)
        else:
            raise ValueError(f"no Close column in yfinance output: {list(raw.columns)[:6]}")
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        px = raw[[col]].rename(columns={col: tickers[0]})
    px = px.copy()
    px.index = pd.to_datetime(px.index)
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    return px.reindex(columns=tickers)


def _download_prices(tickers, start, end, retries=3) -> pd.DataFrame:
    import time
    import yfinance as yf
    last = None
    for k in range(retries):
        try:
            raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False,
                              group_by="column", threads=True)
            if raw is not None and len(raw):
                return extract_close(raw, tickers)
        except Exception as e:             # network hiccups / rate limits
            last = e
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"yfinance download failed after {retries} tries: {last}")


def fetch_raw(tickers, start="2004-06-01", end="2023-12-31", raw_dir=RAW_DIR) -> dict:
    """Download prices + FRED macro to CSV (the only network step). Returns provenance dict."""
    import datetime as dt
    import json
    os.makedirs(raw_dir, exist_ok=True)
    prices = _download_prices(tickers, start, end)
    prices.to_csv(os.path.join(raw_dir, "prices.csv"))
    macro = pd.DataFrame({k: _fred(v) for k, v in FRED_SERIES.items()})
    macro.to_csv(os.path.join(raw_dir, "macro.csv"))
    prov = {"fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tickers": list(tickers),
            "start": start, "end": end, "price_field": "Close, auto_adjust=True (split+dividend adjusted)",
            "fred": FRED_SERIES, "n_price_rows": int(len(prices)),
            "price_missing_by_ticker": prices.isna().sum().to_dict(),
            "macro_first_valid": {c: str(macro[c].first_valid_index()) for c in macro}}
    json.dump(prov, open(os.path.join(raw_dir, "provenance.json"), "w"), indent=1)
    return prov


def _synthetic_esg(tickers) -> pd.Series:
    vals = [20 + 75 * (int(hashlib.md5(t.encode()).hexdigest(), 16) % 1000) / 1000 for t in tickers]
    return pd.Series(vals, index=list(tickers))


def load_real_market(tickers, start="2005-01-01", end="2023-12-31",
                     esg_csv: str | None = None, raw_dir: str = RAW_DIR, allow_pseudo_esg: bool = False) -> Market:
    """Prices (Yahoo, split+dividend adjusted) + FRED macro + ESG scores (CSV: ticker,esg; higher = better).

    Reads data/raw/*.csv written by `scripts/fetch_data.py`; only if they are absent does it
    hit the network (and caches them there).  Refuses to run with pseudo-ESG scores unless
    `allow_pseudo_esg=True`, so that fake ESG can never silently end up in "real" results.

    Caveats to discuss in the report: survivorship bias (tickers chosen are all survivors), look-ahead in
    ESG (one static snapshot applied to 2008), adjusted prices embed dividends, FRED ICE-BofA series are
    truncated to 3 years (dropped automatically, see `_finish`).
    """
    tickers = list(tickers)
    pp, mp = os.path.join(raw_dir, "prices.csv"), os.path.join(raw_dir, "macro.csv")
    if not (os.path.exists(pp) and os.path.exists(mp)):
        fetch_raw(tickers, start, end, raw_dir)
    prices = pd.read_csv(pp, index_col=0, parse_dates=True)
    macro = pd.read_csv(mp, index_col=0, parse_dates=True)
    missing_cols = [t for t in tickers if t not in prices.columns]
    meta = {"requested_tickers": tickers, "tickers_not_in_raw": missing_cols, "esg_csv": esg_csv}
    prices = prices.reindex(columns=[t for t in tickers if t in prices.columns])
    prices = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
    if esg_csv and os.path.exists(esg_csv):
        edf = pd.read_csv(esg_csv)
        e = edf.set_index("ticker")["esg"].astype(float)
        meta["esg_source"] = "csv"
        if "source" in edf.columns:
            meta["esg_provenance"] = sorted(edf["source"].dropna().astype(str).unique().tolist())
    elif allow_pseudo_esg:
        warnings.warn("PSEUDO-ESG scores (hash based) in use. NOT for reporting real-data results.")
        e = _synthetic_esg(prices.columns)
        meta["esg_source"] = "PSEUDO"
    else:
        raise FileNotFoundError("ESG csv not found. Build it with `python scripts/build_esg.py` (see README) "
                                "or pass allow_pseudo_esg=True for smoke tests only.")
    src = "yahoo+fred+esg" if meta["esg_source"] != "PSEUDO" else "yahoo+fred+PSEUDO-ESG"
    return _finish(prices, macro, e, src, meta)
