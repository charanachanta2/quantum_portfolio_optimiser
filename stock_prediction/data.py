"""Market-data access: yfinance (network, TTL-cached) and the offline snapshot shipped in data/raw/prices.csv.
All network access goes through `_download`, which tests replace with a mocked payload."""
from __future__ import annotations
import functools, os, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ttl_cache(seconds: float):
    def deco(fn):
        store = {}
        @functools.wraps(fn)
        def wrap(*a, **k):
            key = (a, tuple(sorted(k.items())))
            now = time.time()
            if key in store and now - store[key][0] < seconds:
                return store[key][1]
            v = fn(*a, **k)
            store[key] = (now, v)
            return v
        wrap.cache_clear = store.clear
        return wrap
    return deco


def _download(tickers, **kw) -> pd.DataFrame:
    import yfinance as yf
    return yf.download(tickers, progress=False, auto_adjust=True, threads=False, timeout=8, **kw)


def _field(df: pd.DataFrame, field: str, tickers) -> pd.DataFrame:
    """Normalise yfinance output (MultiIndex (field, ticker) or flat) to a (dates x tickers) frame for one field."""
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        if field in set(lv0):
            out = df[field]
        else:                                           # (ticker, field) ordering
            out = df.xs(field, axis=1, level=1)
    else:
        out = df[[field]]
        out.columns = [tickers[0] if isinstance(tickers, (list, tuple)) else tickers]
    out = out.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


@ttl_cache(300)
def fetch_prices(symbols: tuple, start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    """Adjusted close panel (dates x tickers), forward-filled over holidays only."""
    raw = _download(list(symbols), start=start, **({"end": end} if end else {}))
    if raw is None or len(raw) == 0:
        raise RuntimeError("no data returned (offline or unknown symbol)")
    return _field(raw, "Close", list(symbols)).sort_index().ffill().dropna(how="all")


@ttl_cache(300)
def fetch_ohlcv(symbol: str, start: str = "2018-01-01") -> pd.DataFrame:
    raw = _download(symbol, start=start)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"no data for {symbol}")
    cols = {f: _field(raw, f, [symbol]).iloc[:, 0] for f in ("Open", "High", "Low", "Close", "Volume") if f in set(
        raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns)}
    return pd.DataFrame(cols).dropna()


@ttl_cache(30)
def fetch_quotes(tickers: tuple) -> pd.DataFrame:
    """Latest price and day change for the market strip (1-day-delayed on weekends/holidays; ~15 min delayed intraday)."""
    raw = _download(list(tickers), period="5d", interval="1d")
    c = _field(raw, "Close", list(tickers)).ffill()
    last, prev = c.iloc[-1], c.iloc[-2] if len(c) > 1 else c.iloc[-1]
    return pd.DataFrame({"last": last, "prev": prev, "chg_pct": (last / prev - 1) * 100, "asof": c.index[-1]})


def load_snapshot(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(ROOT, "data", "raw", "prices.csv")
    p = pd.read_csv(path, index_col=0, parse_dates=True)
    return p.sort_index().ffill()


def symbols_table() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbols_valid_meta.csv"))
    return df.dropna(subset=["Symbol", "Security Name"])[["Symbol", "Security Name"]]
