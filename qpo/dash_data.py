"""Data loaders for the combined dashboard (kept free of Streamlit so they can be unit-tested with mocked yfinance payloads)."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

from .config import Config
from .noise import NoiseModel
from .data import Market


def market_strip(tickers, fetch=None):
    """(DataFrame[last, prev, chg_pct, asof], label).  Falls back to the offline snapshot when Yahoo is unreachable."""
    if fetch is None:
        from stock_prediction import data as D
        fetch = D.fetch_quotes
    try:
        q = fetch(tuple(tickers))
        if q is None or len(q) == 0 or q["last"].isna().all():
            raise RuntimeError("empty quote payload")
        return q, f"LIVE (Yahoo Finance, ~15 min delayed; daily bar {pd.Timestamp(q['asof'].iloc[0]).date()})"
    except Exception as e:
        from stock_prediction import data as D
        snap = D.load_snapshot()
        cols = [t for t in tickers if t in snap.columns]
        last, prev = snap[cols].iloc[-1], snap[cols].iloc[-2]
        q = pd.DataFrame({"last": last, "prev": prev, "chg_pct": (last / prev - 1) * 100, "asof": snap.index[-1]})
        return q, f"OFFLINE SNAPSHOT (last bar {snap.index[-1].date()}) - live feed unavailable: {type(e).__name__}"


def live_market(base: Market, feed_factory=None):
    """Base (snapshot) market extended with every daily bar published since its last date.  -> (market, label, n_added)."""
    try:
        from .feed import YahooFeed, catch_up
        feed = feed_factory(base) if feed_factory else YahooFeed(list(base.returns.columns), base.returns.index[-1])
        m, n = catch_up(base, feed)
        if n == 0:
            return m, f"snapshot is current (last bar {m.returns.index[-1].date()})", 0
        return m, f"LIVE: snapshot + {n} daily bars from Yahoo (last bar {m.returns.index[-1].date()})", n
    except Exception as e:
        return base, f"OFFLINE SNAPSHOT (last bar {base.returns.index[-1].date()}) - Yahoo unreachable: {type(e).__name__}", 0


def load_results(d: str) -> dict:
    """Everything the Backtest / Ablation / Resources tabs show, read from a results folder (results_real/ or results_lstm/)."""
    rd = lambda f, **k: pd.read_csv(os.path.join(d, f), **k) if os.path.exists(os.path.join(d, f)) else None
    out = {"meta": json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}}
    for p in ("GFC_2008", "COVID_2020"):
        out[f"summary_{p}"] = rd(f"backtest_{p}_summary.csv", index_col=0)
        out[f"signif_{p}"] = rd(f"significance_{p}.csv")
        out[f"fig_{p}"] = os.path.join(d, f"backtest_{p}_cum.png") if os.path.exists(os.path.join(d, f"backtest_{p}_cum.png")) else None
    out["ablation"] = rd("ablation_summary.csv", index_col=0)
    out["scaling"] = rd("scaling.csv", index_col=0)
    out["scaling_fits"] = rd("scaling_fits.csv", index_col=0)
    out["resource_corners"] = rd("resource_corners.csv")
    out["resource_assumptions"] = rd("resource_assumptions.csv")
    out["qec_fit"] = json.load(open(os.path.join(d, "qec_fit.json"))) if os.path.exists(os.path.join(d, "qec_fit.json")) else None
    out["figs"] = {n: os.path.join(d, f) for n, f in (("ablation", "report_figs/ablation.png"), ("scaling", "report_figs/scaling.png"),
                                                       ("resources", "report_figs/resources_vs_n.png"), ("tornado", "resource_tornado.png"),
                                                       ("qec", "qec_threshold.png")) if os.path.exists(os.path.join(d, f))}
    out["is_real"] = out["meta"].get("data_source") == "yahoo+fred+esg"
    return out


def quantum_internals(res: dict, cfg: Config, p_layers: int = 2, results: dict | None = None) -> dict:
    """Qubits, circuit size, effective noise per model and the Stim-fitted logical error rate for the instance `res` returned by solve_now."""
    n, zz, lin = res["n_qubits"], res["n_zz"], res["n_lin"]
    raw = NoiseModel(cfg.p2, cfg.p1, cfg.p_readout, cfg.idle_rate, 0.0)
    dd = NoiseModel(cfg.p2, cfg.p1, cfg.p_readout, cfg.idle_rate, cfg.dd_suppression)
    cn, rz = raw.circuit_stats(n, zz, lin, p_layers)
    out = {"qubits": n, "p_layers": p_layers, "cnots": cn, "rz_rx": rz, "depth_estimate": p_layers * (2 * zz + 2 + 1),
           "q_eff_raw": raw.effective_q(n, zz, lin, p_layers), "q_eff_dd": dd.effective_q(n, zz, lin, p_layers)}
    fit = (results or {}).get("qec_fit")
    if fit:
        from . import qec
        lq = qec.logical_q_per_layer(fit, cfg.logical_distance, cfg.logical_p_phys)
        out.update({"logical_q_per_layer": lq, "q_eff_logical": NoiseModel(logical_q_per_layer=lq).effective_q(n, zz, lin, p_layers),
                    "p_th_fit": fit["p_th"], "distance": cfg.logical_distance, "p_phys_logical": cfg.logical_p_phys})
    rc = (results or {}).get("resource_corners")
    if rc is not None and len(rc):
        out["resource_row"] = rc.iloc[0].to_dict()
    return out


def expected_stats(res: dict) -> pd.DataFrame:
    """Model-implied (NOT realised) expected return / vol / Sharpe of each method's weights over the eligible universe."""
    el = res["elig"]; idx = [res["tickers"].index(t) for t in el]
    mu, S = np.asarray(res["mu_elig"]), np.asarray(res["Sigma_elig"])
    rows = []
    for k, w in res["weights"].items():
        we = np.asarray(w)[idx]
        er, vol = float(we @ mu), float(np.sqrt(max(we @ S @ we, 1e-18)))
        rows.append({"method": k, "exp_return_21d": er, "vol_21d": vol, "exp_sharpe": er / vol, "turnover": res["info"][k].get("turnover"),
                     "solve_seconds": res["seconds"][k], "n_assets_held": int((np.asarray(w) > 1e-6).sum())})
    return pd.DataFrame(rows).set_index("method")
