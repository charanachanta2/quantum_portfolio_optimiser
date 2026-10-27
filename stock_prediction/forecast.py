"""forecast_returns(tickers, as_of_date) -> mu_hat   (F1)   -  the interface between the predictor and the optimiser.

No-look-ahead contract
  1. price data used for the features is cut at `index < as_of_date`;
  2. the model used must have been trained on data that ends strictly before `as_of_date` (`Predictor.train_end < as_of_date`),
     otherwise `LookAheadError` is raised.  A `ModelBank` picks the newest model satisfying (2) (walk-forward retraining).
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

from .features import HORIZONS, build_features
from .model import Predictor


class LookAheadError(RuntimeError):
    pass


class ModelBank:
    """Walk-forward model store: files `<folder>/<prefix><YYYY-MM-DD>.keras|json`, the date being the training cut-off."""
    def __init__(self, folder: str, prefix: str = "wf_"):
        self.folder, self.prefix, self._cache = folder, prefix, {}
        self.cutoffs = sorted(pd.Timestamp(os.path.basename(f)[len(prefix):-5]) for f in glob.glob(os.path.join(folder, f"{prefix}*.json")))
        if not self.cutoffs:
            raise FileNotFoundError(f"no models named {prefix}YYYY-MM-DD in {folder}; run scripts/train_predictor.py --walk-forward")

    def select(self, as_of) -> Predictor:
        as_of = pd.Timestamp(as_of)
        ok = [c for c in self.cutoffs if c < as_of]
        if not ok:
            raise LookAheadError(f"no model trained strictly before {as_of.date()} (earliest cut-off {self.cutoffs[0].date()})")
        c = ok[-1]
        if c not in self._cache:
            self._cache[c] = Predictor.load(self.folder, f"{self.prefix}{c.date()}")
        return self._cache[c]


def _resolve(model, as_of) -> Predictor:
    p = model.select(as_of) if isinstance(model, ModelBank) else model
    if pd.Timestamp(as_of) <= p.train_end:
        raise LookAheadError(f"model trained through {p.train_end.date()} cannot be used for as_of={pd.Timestamp(as_of).date()}")
    return p


def confidence_from_metrics(metrics: dict) -> float:
    """Confidence in [0,1] from out-of-sample skill vs the naive baseline: skill<=0 -> 0.25 ('low'), 3 % skill -> 0.5."""
    m = metrics.get("test") or metrics.get("val") or {}
    sk = m.get("skill_vs_naive", {})
    if not sk:
        return 0.25
    s = float(np.mean([sk[k] for k in sk if k in ("h1", "h5", "h7", "h21")]))
    return float(np.clip(0.25 + 8 * s, 0.0, 1.0)) if s > 0 else 0.25


def forecast_returns(tickers, as_of_date, prices: pd.DataFrame, model, horizon: str = "h21", return_log: bool = False) -> np.ndarray:
    """Expected cumulative return over `horizon` (default 21 trading days) for each ticker, known at the OPEN of `as_of_date`.
    `prices`: adjusted-close panel (any length); only rows < as_of_date are read.  NaN if a ticker lacks clean history."""
    p = _resolve(model, as_of_date)
    cols = [t for t in tickers if t in prices.columns]
    out = p.predict_panel(prices[cols], as_of=as_of_date)[horizon].reindex(list(tickers)).values
    return out if return_log else np.expm1(out)


class LSTMForecaster:
    """Callable used by qpo.backtest.make_instance:  f(date, tickers) -> mu_hat (21-day simple return); memoised per date."""
    def __init__(self, model, prices: pd.DataFrame, horizon: str = "h21"):
        self.model, self.prices, self.horizon, self._memo = model, prices, horizon, {}

    def __call__(self, as_of_date, tickers) -> np.ndarray:
        key = (pd.Timestamp(as_of_date), tuple(tickers))
        if key not in self._memo:
            self._memo[key] = forecast_returns(tickers, as_of_date, self.prices, self.model, self.horizon)
        return self._memo[key]

    def confidence(self, as_of_date) -> float:
        return confidence_from_metrics(_resolve(self.model, as_of_date).metrics)


def predict_series(close: pd.Series, predictor: Predictor, last_n: int = 250) -> pd.DataFrame:
    """Predictions made at each of the last `last_n` dates using only data up to that date (rows indexed by that date).
    Column `oos` flags dates after the model's training cut-off (truly out-of-sample)."""
    f = build_features(close.dropna()).dropna()
    L = predictor.hp.seq_len
    ends = np.arange(max(L - 1, len(f) - last_n), len(f))
    X = np.stack([f.values[e - L + 1:e + 1] for e in ends])
    pred = predictor.net.predict(((X - predictor.x_mean) / predictor.x_std).astype(np.float32), verbose=0) * predictor.y_std
    df = pd.DataFrame(pred, index=f.index[ends], columns=[f"h{h}" for h in HORIZONS])
    df["oos"] = df.index > predictor.train_end
    return df
