"""Combined live dashboard: quantum portfolio optimiser + stock prediction system.      streamlit run app.py

Tabs:  Live | Backtest (2008, 2020) | Ablation | Resources | About
The LSTM forecasts supply expected returns (mu) and the fuzzy risk score feeds lambda_t of the time-dependent QUBO; QAOA (simulated) and the
classical solvers then choose the weights.  Research demo - not financial advice."""
import os, sys, time, warnings
from dataclasses import replace
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")
from qpo.config import Config
from qpo.data import load_real_market
from qpo.universe import UNIVERSES
from qpo.live import LiveOptimizer, solve_now
from qpo.bgrun import Job, StreamJob
from qpo import dash_data as DD, qec
from qpo import advisor as ADV, advisor_ui
from qpo.backtest import QUANTUM

st.set_page_config(page_title="Quantum Portfolio Optimizer + Stock Prediction", layout="wide")

METHOD_LABELS = {"qaoa_ideal": "QAOA ideal", "qaoa_noisy_raw": "QAOA noisy (raw)", "qaoa_noisy_mitigated": "QAOA noisy + DD/readout mitigation",
                 "qaoa_logical": "QAOA logical qubits (modelled)", "exact_qubo": "Exact QUBO", "sa_qubo": "Simulated annealing",
                 "miqp_sub": "MIQP (Gurobi if installed, else exact DP)", "cvar_sub": "CVaR LP (sub-universe)", "equal_weight_full": "Equal weight"}


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading market snapshot ...")
def base_market():
    return load_real_market(UNIVERSES["extended"], "2004-06-01", "2023-12-31", "data/esg.csv", "data/raw")


@st.cache_resource(ttl=900, show_spinner="Catching up with Yahoo Finance ...")
def live_market_cached():
    return DD.live_market(base_market())


@st.cache_resource(show_spinner="Loading LSTM ...")
def predictor():
    from stock_prediction.model import Predictor
    try:
        return Predictor.load("models", "deployed")
    except Exception:
        return None


@st.cache_resource
def model_bank():
    from stock_prediction.forecast import ModelBank
    try:
        return ModelBank("models", "wf_")
    except Exception:
        return None


@st.cache_resource
def results(folder):
    return DD.load_results(folder)


def logical_q(cfg):
    fit = results("results_real")["qec_fit"]
    return qec.logical_q_per_layer(fit, cfg.logical_distance, cfg.logical_p_phys) if fit else 1e-6


# ----------------------------------------------------------------------------- sidebar (G5)
with st.sidebar:
    st.header("Controls")
    data_mode = st.radio("Market data", ["Live (Yahoo Finance)", "Offline snapshot (2004-2023)"],
                         help="Live extends the shipped snapshot with every daily bar since 2023-12-29 and falls back to the snapshot if offline.")
    st.subheader("Constraints (QUBO rebuilds on change)")
    esg_min = st.slider("Minimum ESG score (percentile)", 0, 80, 40)
    max_w = st.select_slider("Maximum weight per asset", [0.33, 0.5, 0.67], 0.5)
    tc_bps = st.slider("Transaction cost (bps)", 0, 50, 10)
    lam0 = st.slider("Risk aversion lambda0", 1.0, 15.0, 6.0, 0.5)
    st.subheader("Prediction system -> optimiser")
    mu_source = st.radio("Expected returns (mu)", ["historical", "lstm", "blend"], horizontal=True)
    risk_source = st.radio("lambda_t driver", ["vix", "vix+fuzzy"], horizontal=True)
    st.subheader("Quantum solver")
    n_assets = st.slider("Assets on qubits", 3, 4, 3, help="qubits = assets x 2 bits x 2 periods (12 or 16)")
    shots = st.select_slider("Shots", [256, 512, 1024], 512)
    p2 = st.select_slider("Physical 2-qubit error", [1e-4, 5e-4, 1e-3, 3e-3, 1e-2], 3e-3)
    mitigation = st.selectbox("QAOA variant to run", ["qaoa_noisy_mitigated", "qaoa_noisy_raw", "qaoa_logical", "qaoa_ideal"],
                              format_func=lambda m: METHOD_LABELS[m])
    classical = st.multiselect("Classical comparators", ["exact_qubo", "sa_qubo", "miqp_sub", "cvar_sub", "equal_weight_full"],
                               default=["exact_qubo", "miqp_sub", "cvar_sub", "equal_weight_full"], format_func=lambda m: METHOD_LABELS[m])
    refresh = st.slider("Market-strip refresh (s)", 10, 300, 60)
    st.subheader("Gemini advisor")
    gemini_key = st.text_input("Gemini API key (optional)", type="password",
                               help="Kept only in this browser session. Alternatives: env var GEMINI_API_KEY or .streamlit/secrets.toml. "
                                    "Without a key the advisor writes an offline template summary.")
    gemini_model = st.text_input("Gemini model", os.environ.get("GEMINI_MODEL", ADV.DEFAULT_MODEL))


def _gemini_key():
    if gemini_key.strip():
        return gemini_key.strip()
    try:
        return st.secrets.get("GEMINI_API_KEY")      # raises if no secrets file exists
    except Exception:
        return None


def make_cfg(**over):
    return Config(n_assets=n_assets, shots=shots, p_max=2, opt_maxiter=30, p2=p2, esg_min=float(esg_min), tc_bps=float(tc_bps), lambda0=float(lam0),
                  cap_levels_calm=int(round(max_w * 6)), cap_levels_stress=max(1, int(round(max_w * 6)) - 1), mu_source=mu_source, risk_source=risk_source,
                  **over)


# ----------------------------------------------------------------------------- data + banner (G10)
if data_mode.startswith("Live"):
    market, data_label, n_new = live_market_cached()
else:
    market, data_label, n_new = base_market(), f"OFFLINE SNAPSHOT (last bar {base_market().returns.index[-1].date()})", 0
res_real = results("results_real")
st.title("Quantum portfolio optimizer + stock prediction system")
st.warning("**Research demo - not financial advice.** Quantum solvers run on a classical simulator; logical-qubit results use a modelled noise channel. "
           f"**Data:** {data_label}.  **Backtest results:** {'REAL data (Yahoo + FRED + S&P-sourced ESG snapshot)' if res_real['is_real'] else 'SYNTHETIC - not real markets'}. "
           "Forecasts are noisy; see the About tab for what the evidence does and does not show.")
tabs = st.tabs(["Live", "Backtest (2008, 2020)", "Ablation", "Resources", "About"])
tick_all = list(market.returns.columns)

# ============================================================================= LIVE
with tabs[0]:
    # ---- G2 market strip
    strip_t = tick_all[:7] + ["^VIX"]

    def _strip():
        q, label = DD.market_strip(strip_t)
        st.caption(f"Market strip - {label} - refreshed {time.strftime('%H:%M:%S')}")
        cols = st.columns(len(q))
        for c, (t, r) in zip(cols, q.iterrows()):
            c.metric(t.replace("^", ""), f"{r['last']:.2f}", f"{r['chg_pct']:+.2f}%")
    st.fragment(run_every=f"{refresh}s")(_strip)()

    # ---- G3 prediction panel
    st.subheader("Prediction panel")
    pc1, pc2 = st.columns([1, 3])
    sym = pc1.selectbox("Ticker", tick_all)
    other = pc1.text_input("...or any other symbol (needs internet)", "")
    sym = other.strip().upper() or sym
    pr = predictor()
    if pr is None:
        pc2.error("No trained model in models/. Run `python scripts/train_predictor.py` (offline training - the app never trains on request).")
    else:
        from stock_prediction import data as D
        from stock_prediction.service import analyse_symbol
        try:
            px = D.fetch_prices((sym,), start="2022-01-01")[sym] if data_mode.startswith("Live") or other.strip() else D.load_snapshot()[sym]
            src = "Yahoo Finance"
        except Exception:
            px = market.prices[sym] if sym in market.prices.columns else None
            src = "offline snapshot"
        if px is None:
            pc2.error("No data for this symbol offline.")
        else:
            vz = float((market.macro["VIX"].iloc[-1] - market.macro["VIX"].iloc[-252:].mean()) / (market.macro["VIX"].iloc[-252:].std() + 1e-9))
            try:
                a = analyse_symbol(sym, px, pr, vix_z=vz)
                fig = go.Figure(go.Scatter(x=px.index[-180:], y=px.iloc[-180:], name="Close"))
                fig.add_trace(go.Scatter(x=a["dates"], y=a["path"], name="7-day forecast", line=dict(dash="dot")))
                fig.add_trace(go.Scatter(x=list(a["dates"]) + list(a["dates"])[::-1], y=list(a["hi"]) + list(a["lo"])[::-1], fill="toself", mode="lines", opacity=0.25,
                                         line=dict(width=0), name="90% band (validation residuals)"))
                fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), title=f"{sym}: price ({src}) and LSTM 7-day forecast")
                pc2.plotly_chart(fig, width="stretch")
                g1, g2, g3, g4 = st.columns(4)
                gauge = go.Figure(go.Indicator(mode="gauge+number", value=a["risk"]["score"] * 100, title={"text": f"Fuzzy risk - {a['risk']['label']}"},
                                               gauge={"axis": {"range": [0, 100]}, "steps": [{"range": [0, 33], "color": "#cfe8cf"}, {"range": [33, 66], "color": "#f3e7b3"},
                                                                                              {"range": [66, 100], "color": "#f1c0bb"}]}))
                gauge.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
                g1.plotly_chart(gauge, width="stretch")
                g2.metric("Trading signal (fuzzy)", a["signal"], f"{a['exp_ret_21d_pct']:+.2f}% expected 21d")
                g3.metric("Confidence", a["confidence_label"], f"{a['confidence']:.2f}")
                g4.metric(f"1-day skill vs naive ({a['metrics_split']})", f"{a['skill_h1']:+.3f}" if a["skill_h1"] is not None else "n/a",
                          "negative = worse than 'tomorrow = today'", delta_color="off")
                st.caption(f"Model trained through {a['model_train_end']} (offline). Direction accuracy {a['dir_acc_h1']:.3f} vs base rate {a['dir_base_h1']:.3f}. "
                           "Skill near zero means the forecast adds little beyond the naive baseline.")
            except Exception as e:
                pc2.error(f"Could not analyse {sym}: {e}")

    # ---- G4 optimiser panel (background worker, G9)
    st.subheader("Optimizer panel")
    st.caption(f"Weights chosen on the data up to the latest bar ({market.returns.index[-1].date()}).  mu source: **{mu_source}**, lambda_t driver: **{risk_source}**.")
    methods = [mitigation] + classical

    def _forecaster():
        if mu_source == "historical":
            return None
        from stock_prediction.forecast import LSTMForecaster
        return LSTMForecaster(predictor(), market.prices) if predictor() is not None else None

    if st.button("Solve now", type="primary"):
        cfg = make_cfg()
        fc = _forecaster()
        if mu_source != "historical" and fc is None:
            st.error("mu source needs the LSTM model (models/deployed.*).")
        else:
            st.session_state["job"] = Job(solve_now, market, cfg, methods, logical_q=logical_q(cfg), forecaster=fc)
            st.session_state["job_cfg"] = cfg
            st.session_state.pop("quotes_at_solve", None)

    def _opt_panel():
        job = st.session_state.get("job")
        if job is None:
            st.info("Press **Solve now**. The solve runs in a background thread; this panel updates by itself.")
            return
        if job.state == "running":
            st.info(f"Solving ... {job.seconds:.0f}s (QAOA on a statevector simulator)"); return
        if job.state == "error":
            st.error(job.error); return
        r = job.result
        W = pd.DataFrame({METHOD_LABELS[k]: v for k, v in r["weights"].items()}, index=r["tickers"])
        W = W[(W > 1e-6).any(axis=1)]
        fig = go.Figure([go.Bar(name=c, x=W.index, y=W[c]) for c in W.columns]); fig.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=30, b=10),
                                                                                                    yaxis_title="weight", title=f"Target weights as of {r['date'].date()}")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(DD.expected_stats(r).rename(index=METHOD_LABELS).style.format({"exp_return_21d": "{:.2%}", "vol_21d": "{:.2%}", "exp_sharpe": "{:.2f}", "turnover": "{:.2f}",
                                                                                      "solve_seconds": "{:.2f}"}), width="stretch")
        st.caption("Expected return/vol/Sharpe are MODEL-IMPLIED (mu and Sigma of the instance), not realised performance. "
                   f"QUBO on {r['n_qubits']} qubits over assets {', '.join(r['sub'])}; solve time {job.seconds:.1f}s total.")
        if "quotes_at_solve" not in st.session_state:
            try:
                from stock_prediction import data as D
                st.session_state["quotes_at_solve"] = D.fetch_quotes(tuple(r["tickers"]))["last"]
            except Exception:
                st.session_state["quotes_at_solve"] = None
    st.fragment(run_every="1s" if (st.session_state.get("job") is not None and st.session_state["job"].state == "running") else None)(_opt_panel)()

    # ---- G6 streaming portfolio value
    st.subheader("Streaming portfolio value")
    s1, s2 = st.columns(2)
    start = s1.selectbox("Replay start (historical bars, streamed tick by tick)", ["2008-06-02 (GFC)", "2020-01-02 (COVID)", "2015-01-02 (calm)"]).split(" ")[0]
    n_ticks = s2.slider("Ticks to stream", 20, 200, 60)
    reb = s2.slider("Re-optimise every N ticks", 10, 63, 21)
    if st.button("Start streaming replay"):
        cfg = make_cfg(rebalance_days=reb)
        fc = None
        if mu_source != "historical":
            from stock_prediction.forecast import LSTMForecaster
            mb = model_bank()
            fc = LSTMForecaster(mb, base_market().prices) if mb is not None else None
        lo = LiveOptimizer(base_market(), cfg, start, methods, logical_q(cfg), forecaster=fc)
        st.session_state["stream"] = StreamJob(lo, n_ticks, delay=0.05)

    def _stream():
        sj = st.session_state.get("stream")
        if sj is None:
            st.info("Streaming replay of historical daily bars (the live Yahoo feed delivers one bar per trading day, see About).")
            return
        sn = sj.snapshot()
        if sn["error"]:
            st.error(sn["error"]); return
        if len(sn["pnl"]) > 1:
            fig = go.Figure([go.Scatter(x=sn["pnl"].index, y=sn["pnl"][c], name=METHOD_LABELS.get(c, c)) for c in sn["pnl"].columns])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="portfolio value (start = 1)")
            st.plotly_chart(fig, width="stretch")
        st.progress(min(sn["t"] / sn["n"], 1.0), text=f"{sn['state']}  tick {sn['t']}/{sn['n']}")
    st.fragment(run_every="1s")(_stream)()

    # live mark-to-market of the last 'Solve now' weights
    def _mtm():
        job, q0 = st.session_state.get("job"), st.session_state.get("quotes_at_solve")
        if job is None or job.state != "done" or q0 is None:
            return
        try:
            from stock_prediction import data as D
            q1 = D.fetch_quotes(tuple(job.result["tickers"]))["last"]
        except Exception:
            st.caption("Live mark-to-market unavailable (offline)."); return
        rel = (q1 / q0 - 1).fillna(0)
        rows = {METHOD_LABELS[k]: float(np.asarray(w) @ rel.values) for k, w in job.result["weights"].items()}
        st.caption(f"Mark-to-market since the solve (live quotes, no costs): " + "; ".join(f"{k}: {v:+.3%}" for k, v in rows.items()))
    st.fragment(run_every=f"{refresh}s")(_mtm)()

    # ---- G7 quantum internals
    st.subheader("Quantum internals")
    job = st.session_state.get("job")
    if job is not None and job.state == "done":
        cfg = st.session_state.get("job_cfg", make_cfg())
        p_used = int(next((v.get("p") for v in job.result["info"].values() if v.get("p")), cfg.p_max))
        qi = DD.quantum_internals(job.result, cfg, p_used, res_real)
        c = st.columns(5)
        c[0].metric("Logical qubits (QAOA)", qi["qubits"]); c[1].metric("QAOA layers p", qi["p_layers"]); c[2].metric("CNOTs / depth (est.)", f"{qi['cnots']} / {qi['depth_estimate']}")
        c[3].metric("Effective flip prob. (raw / DD)", f"{qi['q_eff_raw']:.3f} / {qi['q_eff_dd']:.3f}")
        if "logical_q_per_layer" in qi:
            c[4].metric(f"Logical error / layer (d={qi['distance']}, p={qi['p_phys_logical']:g})", f"{qi['logical_q_per_layer']:.1e}", f"Stim fit p_th={qi['p_th_fit']:.4f}", delta_color="off")
        if "resource_row" in qi:
            r0 = qi["resource_row"]
            st.caption(f"Resource estimate (first corner of results_real/resource_corners.csv): {r0}")
        st.caption("Logical error rates come from Stim/PyMatching surface-code MEMORY simulations; QAOA receives them as an effective bit-flip rate (modelled logical qubits). Resource-estimator constants are assumptions.")
    else:
        st.info("Run **Solve now** to see the circuit and noise numbers of the current instance.")

    # ---- Gemini advisor: stock pick (computed by the optimiser code) + written summary (Gemini or offline template)
    st.subheader("AI advisor (Gemini)")
    st.caption("The stock ranking is computed from the optimisers' weights; Gemini only writes the explanation. Research demo - not financial advice.")
    def _advisor_panel():
        job = st.session_state.get("job")
        if job is not None and job.state in ("done", "error") and st.session_state.get("_adv_rerun_for") != id(job):
            st.session_state["_adv_rerun_for"] = id(job)
            st.rerun()      # one full rerun when the solve finishes, so every panel (internals, advisor) sees the result
        advisor_ui.render_panel(st, job, res_real, st.session_state.get("job_cfg", make_cfg()), _gemini_key(), gemini_model.strip(), esg=market.esg)
    _jb = st.session_state.get("job")
    st.fragment(run_every="1s" if (_jb is not None and _jb.state == "running") else None)(_advisor_panel)()

# ============================================================================= BACKTEST
with tabs[1]:
    st.subheader("Out-of-sample backtests (results_real/)")
    for p in ("GFC_2008", "COVID_2020"):
        s = res_real[f"summary_{p}"]
        if s is None:
            continue
        st.markdown(f"**{p}**")
        st.dataframe(s.style.format("{:.3f}"), width="stretch")
        t = res_real[f"signif_{p}"]
        if t is not None:
            st.caption(f"Significant quantum-vs-classical Sharpe differences (Holm-adjusted bootstrap p<0.05): {int((t.p_boot_holm < 0.05).sum())} / {len(t)}")
        if res_real[f"fig_{p}"]:
            st.image(res_real[f"fig_{p}"])
    lstm = results("results_lstm")
    st.subheader("Prediction-system variants (results_lstm/)")
    import glob
    sums = sorted(glob.glob("results_lstm/summary_*.csv"))
    if not sums:
        st.info("Run `python scripts/backtest_lstm.py --real` to produce these.")
    for f in sums:
        w = os.path.basename(f)[8:-4]
        d = pd.read_csv(f)
        st.markdown(f"**{w}: Sharpe by method and variant**")
        st.dataframe(d.pivot(index="method", columns="variant", values="sharpe").style.format("{:.2f}"), width="stretch")
        for g, title in ((f"results_lstm/significance_vs_historical_{w}.csv", "variant vs historical (paired bootstrap, Holm)"),
                         (f"results_lstm/forecast_quality_{w}.csv", "forecast quality per rebalance")):
            if os.path.exists(g):
                x = pd.read_csv(g)
                st.caption(title)
                st.dataframe(x.describe().T[["mean", "min", "max"]].round(4) if "forecast" in g else x.round(4), width="stretch")

# ============================================================================= ABLATION
with tabs[2]:
    st.subheader("Error-mitigation ablation")
    if res_real["ablation"] is not None:
        st.dataframe(res_real["ablation"].style.format("{:.4f}"), width="stretch")
    if "ablation" in res_real["figs"]:
        st.image(res_real["figs"]["ablation"])
    if res_real["scaling"] is not None:
        st.subheader("Scaling: shots/evaluations to the optimum vs qubits")
        st.dataframe(res_real["scaling"].style.format("{:.2f}"), width="stretch")
        if res_real["scaling_fits"] is not None:
            st.caption("Fitted exponents (cost ~ 2^(slope*n), n=6..16 only - indicative): " + ", ".join(f"{i}: {v:.2f}" for i, v in res_real["scaling_fits"].iloc[:, 0].items()))
        if "scaling" in res_real["figs"]:
            st.image(res_real["figs"]["scaling"])

# ============================================================================= RESOURCES
with tabs[3]:
    st.subheader("Fault-tolerant resource estimate")
    if res_real["resource_corners"] is not None:
        st.dataframe(res_real["resource_corners"].round(3), width="stretch")
    for k in ("resources", "tornado", "qec"):
        if k in res_real["figs"]:
            st.image(res_real["figs"][k])
    if res_real["resource_assumptions"] is not None:
        with st.expander("Assumptions (constants are assumptions, swept in results_real/resource_*)"):
            st.dataframe(res_real["resource_assumptions"], width="stretch")

# ============================================================================= ABOUT
with tabs[4]:
    st.markdown(open("docs/ABOUT_DASHBOARD.md").read() if os.path.exists("docs/ABOUT_DASHBOARD.md") else "See README.md")
