"""Optimiser-only dashboard: live re-optimisation under a streaming daily-bar feed.   streamlit run dashboard/app.py
(The combined app with the stock-prediction system is the root `app.py`.)
Three data sources: synthetic (offline), real snapshot (STREAMING REPLAY of historical daily bars), and Live (Yahoo Finance): the snapshot is
caught up to the latest published daily bar through the `next_bar()` hook (qpo.feed.YahooFeed) and one as-of-today decision is solved; new bars
are polled as they appear (Yahoo daily bars arrive once per trading day).  Every N ticks the time-dependent QUBO is rebuilt from the data seen
so far and re-solved by every selected method."""
import os, sys, time
import numpy as np, pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo.config import Config
from qpo.data import synthetic_market, load_real_market, CRISES
from qpo.universe import UNIVERSES
from qpo.live import LiveOptimizer
from qpo.backtest import ALL_METHODS

st.set_page_config(page_title="Quantum Portfolio Optimizer", layout="wide")
st.title("Fault-tolerant quantum portfolio optimizer — live re-optimisation (streaming replay / Yahoo feed)")

with st.sidebar:
    st.header("Settings")
    source = st.radio("Market data", ["Synthetic (offline)", "Real (data/raw + data/esg.csv)", "Live (Yahoo Finance feed)"],
                      help="Real needs `python scripts/fetch_data.py` and `python scripts/build_esg.py` first. Live = real snapshot + every daily bar since, polled from Yahoo (needs internet; falls back to the snapshot).")
    start = st.selectbox("Replay start", ["2008-06-02 (GFC)", "2020-01-02 (COVID)", "2015-01-02 (calm)"])
    start = start.split(" ")[0]
    n_assets = st.slider("Assets on qubits (A)", 3, 4, 3, help="qubits = A * 2 bits * 2 periods")
    reb = st.slider("Re-optimise every N ticks", 5, 63, 21)
    shots = st.select_slider("Shots", [256, 512, 1024, 2048], 1024)
    methods = st.multiselect("Methods", ["qaoa_ideal", "qaoa_noisy_raw", "qaoa_noisy_mitigated", "qaoa_logical",
                                          "exact_qubo", "sa_qubo", "miqp_sub", "cvar_sub", "equal_weight_full"],
                             default=["qaoa_noisy_mitigated", "qaoa_logical", "exact_qubo", "cvar_sub"])
    n_ticks = st.slider("Ticks to replay", 20, 250, 80)
    delay = st.slider("Delay per tick (s)", 0.0, 0.5, 0.02)
    p2 = st.select_slider("Physical 2q error", [1e-4, 5e-4, 1e-3, 3e-3, 1e-2], 3e-3)
    go = st.button("▶ Start stream", type="primary")


@st.cache_resource(show_spinner=False)
def _load(real: bool):
    if real:
        return load_real_market(UNIVERSES["extended"], "2004-06-01", "2023-12-31", "data/esg.csv", "data/raw")
    return synthetic_market(n_assets=16)


@st.cache_resource(ttl=900, show_spinner="Catching up with Yahoo Finance ...")
def _live():
    from qpo.dash_data import live_market
    return live_market(_load(True))


def get_market():
    if source.startswith("Live"):
        return _live()[0]
    return _load(source.startswith("Real"))


try:
    _m = get_market()
except Exception as e:                                           # missing raw data / esg csv
    st.error(f"Could not load the selected data: {e}\n\nRun `python scripts/fetch_data.py` and `python scripts/build_esg.py`, "
             "or switch to synthetic data.")
    st.stop()
if _m.source.startswith("synthetic"):
    st.warning("SYNTHETIC DATA - regime-switching simulation with 2008-like and 2020-like crises. Not real market prices.")
elif source.startswith("Live"):
    st.info(f"LIVE FEED: {_live()[1]}. Yahoo daily bars are end-of-day (quotes ~15 min delayed). Research demo, not financial advice.")
else:
    st.success(f"REAL DATA (streaming replay of historical bars): {_m.source} | {_m.returns.shape[1]} assets | {_m.returns.index[0].date()} .. {_m.returns.index[-1].date()} "
               "| survivors only; static ESG snapshot (see report, threats to validity)")


live_mode = source.startswith("Live")
if live_mode and methods:
    from qpo.live import solve_now
    if st.button("Optimise now (as of the latest bar)"):
        cfg = Config(n_assets=n_assets, shots=shots, p_max=2, opt_maxiter=30, p2=p2)
        with st.spinner("Solving ..."):
            r = solve_now(get_market(), cfg, methods)
        st.write(f"Decision date {r['date'].date()} on {r['n_qubits']} qubits, assets {', '.join(r['sub'])}")
        st.bar_chart(pd.DataFrame(r["weights"], index=r["tickers"]).loc[lambda d: (d > 1e-6).any(axis=1)])
        st.dataframe(pd.DataFrame({"seconds": r["seconds"]}))
    st.caption("Streaming from the Yahoo feed: polled bars would be appended one per trading day via `LiveOptimizer(feed=YahooFeed(...))`; use the replay options for a continuous demo.")
elif go and methods:
    cfg = Config(n_assets=n_assets, rebalance_days=reb, shots=shots, p_max=2, opt_maxiter=30, p2=p2)
    lo = LiveOptimizer(get_market(), cfg, start, methods)
    c1, c2 = st.columns([3, 2])
    chart, wbox = c1.empty(), c2.empty()
    status, table = st.empty(), st.empty()
    prog = st.progress(0.0)
    for t in range(n_ticks):
        out = lo.step()
        if out.get("done"):
            break
        df = pd.DataFrame(lo.pnl, index=lo.dates)
        chart.line_chart(df, height=320)
        if out["reoptimised"] or t == 0:
            W = pd.DataFrame({k: lo.target[k] for k in methods}, index=get_market().returns.columns)
            wbox.bar_chart(W[(W > 1e-6).any(axis=1)], height=320)
            rows = []
            for k in methods:
                info = lo.last_info.get(k, {})
                rows.append({"method": k, "E_sel": info.get("E_sel"), "E_opt": info.get("E_opt"),
                             "gap_norm": info.get("gap_norm"), "cap_levels": info.get("stress_cap"),
                             "lambda": info.get("lambda0")})
            table.dataframe(pd.DataFrame(rows))
        status.info(f"{out['date'].date()}  tick {t + 1}/{n_ticks}" + ("   🔁 re-optimised" if out["reoptimised"] else ""))
        prog.progress((t + 1) / n_ticks)
        time.sleep(delay)
    st.success("Replay finished")
else:
    st.write("Pick settings in the sidebar and press **Start stream**. "
             "Weights update at every re-optimisation; the cap tightens automatically when VIX z-score > 1 (time-varying risk constraint).")
