"""Scale-free features and multi-horizon targets.  Every feature at row t uses data up to and including t only."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["ret1", "ret5", "ret10", "vol10", "vol20", "rsi", "macd", "ma20_ratio", "ema50_ratio", "mom10"]
HORIZONS = (1, 2, 3, 4, 5, 6, 7, 21)       # cumulative log-return horizons predicted by the network
H_MAX = max(HORIZONS)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def build_features(close: pd.Series) -> pd.DataFrame:
    """Price-only features (adjusted close).  Normalised by price so one pooled model serves all tickers."""
    c = close.astype(float)
    lr = np.log(c).diff()
    f = pd.DataFrame(index=c.index)
    f["ret1"] = lr
    f["ret5"] = np.log(c).diff(5)
    f["ret10"] = np.log(c).diff(10)
    f["vol10"] = lr.rolling(10).std()
    f["vol20"] = lr.rolling(20).std()
    f["rsi"] = (rsi(c) - 50) / 50
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    f["macd"] = (ema12 - ema26) / c
    f["ma20_ratio"] = c / c.rolling(20).mean() - 1
    f["ema50_ratio"] = c / c.ewm(span=50, adjust=False).mean() - 1
    f["mom10"] = c / c.shift(10) - 1
    return f[FEATURES]


def make_targets(close: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """Cumulative log-return from t to t+h.  NaN where t+h is beyond the data (so never used for training)."""
    lc = np.log(close.astype(float))
    return pd.DataFrame({f"h{h}": lc.shift(-h) - lc for h in horizons}, index=close.index)
