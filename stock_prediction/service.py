"""Everything the UIs (Flask and Streamlit) show for one ticker, built from the pure functions in this package (E3)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .features import HORIZONS
from .forecast import confidence_from_metrics, predict_series
from .fuzzy import fuzzy_risk, fuzzy_signal
from .model import Predictor


def conf_label(c: float) -> str:
    return "High Confidence" if c >= 0.66 else "Medium Confidence" if c >= 0.4 else "Low Confidence"


def risk_analysis(close: pd.Series, vix_z: float = 0.0, confidence: float = 0.25) -> dict:
    vol = float(close.pct_change().dropna().iloc[-20:].std() * np.sqrt(252))
    score, label = fuzzy_risk(vol, vix_z, confidence)
    return {"ann_vol": vol, "score": score, "label": label}


def analyse_symbol(symbol: str, close: pd.Series, predictor: Predictor, vix_z: float = 0.0, volume: pd.Series | None = None) -> dict:
    close = close.dropna()
    last = float(close.iloc[-1])
    pan = predictor.predict_panel(pd.DataFrame({symbol: close}))
    h = pan.iloc[0]
    if h.isna().any():
        raise ValueError("not enough clean history for this symbol")
    k = [f"h{i}" for i in range(1, 8)]
    sd = predictor.resid_std[:7]
    path = last * np.exp(h[k].values)
    lo, hi = last * np.exp(h[k].values - 1.645 * sd), last * np.exp(h[k].values + 1.645 * sd)
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=7)
    conf = confidence_from_metrics(predictor.metrics)
    risk = risk_analysis(close, vix_z, conf)
    exp21 = float(np.expm1(h["h21"]) * 100)
    trend = float(close.iloc[-1] / close.iloc[-20:].mean() - 1)
    vr = 1.0 if volume is None or len(volume) < 21 else float(volume.iloc[-1] / volume.iloc[-21:-1].mean())
    s, lab = fuzzy_signal(exp21, trend, vr, risk["score"])
    ev = predictor.metrics.get("test") or predictor.metrics.get("val") or {}
    split = "test" if "test" in predictor.metrics else "validation"
    return {"symbol": symbol, "last": last, "dates": dates, "path": path, "lo": lo, "hi": hi, "exp_ret_21d_pct": exp21,
            "exp_ret_7d_pct": float((np.exp(h["h7"]) - 1) * 100), "confidence": conf, "confidence_label": conf_label(conf), "risk": risk,
            "signal_score": s, "signal": lab, "model_train_end": str(predictor.train_end.date()), "metrics_split": split,
            "rmse_h1_model": ev.get("rmse_model", {}).get("h1"), "rmse_h1_naive": ev.get("rmse_naive", {}).get("h1"),
            "skill_h1": ev.get("skill_vs_naive", {}).get("h1"), "dir_acc_h1": ev.get("direction_acc", {}).get("h1"),
            "dir_base_h1": ev.get("direction_base_rate", {}).get("h1"), "metrics": ev}


def prediction_vs_actual(close: pd.Series, predictor: Predictor, n: int = 250) -> pd.DataFrame:
    """Next-day predicted vs realised price, each prediction made with data up to its own date only."""
    ps = predict_series(close, predictor, n + 1)
    c = close.reindex(ps.index)
    nxt = close.shift(-1).reindex(ps.index)
    out = pd.DataFrame({"actual_next": nxt, "pred_next": c * np.exp(ps["h1"]), "naive_next": c, "oos": ps["oos"]}).dropna()
    return out
