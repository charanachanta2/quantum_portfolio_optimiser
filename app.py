"""QC-01 Fault-Tolerant Quantum Portfolio Optimizer with Dynamic Constraints - dashboard.      streamlit run app.py

Three SEPARATE data modes (sidebar, Step 1):
  Synthetic  - simulated market, fully offline.
  Real       - saved real snapshot: data/raw (prices, VIX) + data/esg.csv. No internet calls at all.
  Live feed  - snapshot extended with the latest Yahoo Finance bars; market strip, live re-optimisation, mark-to-market.
Flow: choose data -> AI forecasts returns & risk -> (quantum) optimiser picks weights -> advisor names the best stock.
Research demo - not financial advice."""
import os, sys, time, warnings, glob
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

st.set_page_config(page_title="Quantum Portfolio Optimizer", page_icon="⚛️", layout="wide")

METHOD_LABELS = {"qaoa_ideal": "QAOA ideal", "qaoa_noisy_raw": "QAOA noisy (raw)", "qaoa_noisy_mitigated": "QAOA noisy + error mitigation",
                 "qaoa_logical": "QAOA logical qubits (modelled)", "exact_qubo": "Exact QUBO", "sa_qubo": "Simulated annealing",
                 "miqp_sub": "MIQP (Gurobi if installed, else exact DP)", "cvar_sub": "CVaR LP (sub-universe)", "equal_weight_full": "Equal weight"}
COLORS = ["#7c3aed", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#3b82f6", "#ec4899", "#84cc16"]

# ----------------------------------------------------------------------------- look & feel
st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1280px}
footer,#MainMenu{visibility:hidden}
.hero{background:linear-gradient(120deg,#4f46e5 0%,#7c3aed 50%,#06b6d4 100%);padding:1.7rem 2rem;border-radius:20px;color:#fff;margin-bottom:1rem;box-shadow:0 8px 28px rgba(79,70,229,.25)}
.hero.live{background:linear-gradient(120deg,#047857 0%,#0891b2 60%,#2563eb 100%)}
.hero.synth{background:linear-gradient(120deg,#b45309 0%,#d97706 50%,#7c3aed 100%)}
.hero h1{margin:0;font-size:2.1rem;color:#fff;line-height:1.2}
.hero p{margin:.4rem 0 0;color:#fff;opacity:.93;font-size:1.02rem}
.pill{display:inline-block;padding:.22rem .75rem;border-radius:999px;background:rgba(255,255,255,.22);font-size:.8rem;margin:.8rem .4rem 0 0;color:#fff}
.step{border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:.85rem 1rem;height:100%;background:rgba(128,128,128,.05)}
.step .n{display:inline-block;width:1.7rem;height:1.7rem;line-height:1.7rem;text-align:center;border-radius:50%;background:#7c3aed;color:#fff;font-weight:700;margin-right:.5rem}
.step .t{font-weight:700}
.step .d{opacity:.75;font-size:.88rem;margin-top:.3rem}
.sec{font-size:1.3rem;font-weight:800;margin:1.8rem 0 .15rem}
.hint{opacity:.72;font-size:.93rem;margin-bottom:.7rem}
.pick{background:linear-gradient(135deg,rgba(16,185,129,.20),rgba(6,182,212,.16));border:1px solid rgba(16,185,129,.5);border-radius:18px;padding:1.1rem 1.4rem}
.pick .lbl{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;opacity:.75}
.pick .big{font-size:2.6rem;font-weight:800;line-height:1.15}
.pick .sub{opacity:.8;font-size:.92rem}
.chip{display:inline-block;padding:.12rem .6rem;border-radius:999px;border:1px solid rgba(128,128,128,.4);font-size:.78rem;margin-right:.3rem}
.livebox{border:1px solid rgba(16,185,129,.5);background:rgba(16,185,129,.08);border-radius:16px;padding:.9rem 1.1rem;margin-bottom:.8rem}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:.65rem .95rem;background:rgba(128,128,128,.06)}
section[data-testid="stSidebar"] .stButton>button{border-radius:12px;font-weight:700;padding:.65rem .5rem}
div.stButton>button[kind="primary"]{border-radius:12px;font-weight:700}
button[data-baseweb="tab"]{font-weight:600}
</style>
""", unsafe_allow_html=True)


def section(title, hint=""):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="hint">{hint}</div>' if hint else ""), unsafe_allow_html=True)


def style_fig(fig, height=330, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), colorway=COLORS, legend=dict(orientation="h", y=-0.2), **kw)
    return fig


def top_avg(r):
    """Best stock = highest average weight across the optimisers (equal-weight baseline excluded)."""
    ws = {k: np.asarray(v, float) for k, v in r["weights"].items() if k != "equal_weight_full"} or {k: np.asarray(v, float) for k, v in r["weights"].items()}
    return pd.Series(np.mean(list(ws.values()), axis=0), index=r["tickers"]).sort_values(ascending=False), len(ws)


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading real market data (data/raw + data/esg.csv) ...")
def base_market():
    return load_real_market(UNIVERSES["extended"], "2004-06-01", "2023-12-31", "data/esg.csv", "data/raw")


@st.cache_resource(ttl=900, show_spinner="Catching up with Yahoo Finance ...")
def live_market_cached():
    return DD.live_market(base_market())


@st.cache_resource(show_spinner="Generating synthetic market ...")
def synthetic_market():
    """Synthetic regime-switching market from qpo/data.py. The exact function name was not visible when this file was written, so likely
    names/call styles are tried. If none matches, an error tells you what to fix. Find the real name with:
        python -c "import qpo.data as d; print([n for n in dir(d) if not n.startswith('_')])" """
    import qpo.data as QD
    names = ("synthetic_market", "make_synthetic_market", "generate_synthetic_market", "make_synthetic", "synth_market", "synthetic", "generate_synthetic")
    attempts = (lambda f: f(UNIVERSES["extended"], "2004-06-01", "2023-12-31"), lambda f: f(UNIVERSES["extended"]), lambda f: f(Config()), lambda f: f())
    last_err = None
    for n in names:
        fn = getattr(QD, n, None)
        if not callable(fn):
            continue
        for call in attempts:
            try:
                out = call(fn)
            except TypeError as e:
                last_err = e
                continue
            return out[0] if isinstance(out, tuple) and hasattr(out[0], "returns") else out
    raise RuntimeError(f"Could not find the synthetic-market function in qpo/data.py (tried {', '.join(names)}). Last error: {last_err}. "
                       "Edit synthetic_market() in app.py with the right name.")


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


# ----------------------------------------------------------------------------- dataset buttons (callbacks run BEFORE the rerun)
if "dataset" not in st.session_state:
    st.session_state["dataset"] = "real"


def _choose_dataset(src):
    st.session_state["dataset"] = src
    st.session_state["autostart"] = True            # the dashboard starts the optimisation automatically
    for k in ("job", "job_cfg", "quotes_at_solve", "stream", "_adv_rerun_for", "live_hist", "_logged", "last_live_solve"):
        st.session_state.pop(k, None)               # drop results that belong to the previous dataset


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## ⚛️ Controls")
    st.markdown("**Step 1 - choose your data** (click to run)")
    _active = st.session_state["dataset"]
    st.button("🧪  Synthetic data", on_click=_choose_dataset, args=("synthetic",), use_container_width=True,
              type="primary" if _active == "synthetic" else "secondary", help="Simulated market with 2008-like and 2020-like crises. Offline.")
    st.button("📈  Real data (esg.csv + raw)", on_click=_choose_dataset, args=("real",), use_container_width=True,
              type="primary" if _active == "real" else "secondary", help="Saved real prices (data/raw) + ESG scores (data/esg.csv). Offline - no Yahoo calls.")
    st.button("📡  Live feed (Yahoo)", on_click=_choose_dataset, args=("live",), use_container_width=True,
              type="primary" if _active == "live" else "secondary", help="Real snapshot + the newest Yahoo Finance bars, live quotes and live re-optimisation. Needs internet.")
    st.caption("Active: **" + {"synthetic": "Synthetic (simulated)", "real": "Real (data/raw + data/esg.csv)", "live": "Live feed (Yahoo Finance)"}[_active] + "**")

    st.markdown("**Step 2 - your preferences**")
    lam0 = st.slider("Risk aversion", 1.0, 15.0, 6.0, 0.5, help="Higher = the optimiser prefers safer, steadier portfolios.")
    esg_min = st.slider("Minimum ESG score (percentile)", 0, 80, 40, help="Hard ESG exclusion: removes companies with lower sustainability scores.")
    max_w = st.select_slider("Max share per stock", [0.33, 0.5, 0.67], 0.5, format_func=lambda x: f"{int(x * 100)}%")

    with st.expander("⚙️ Advanced settings"):
        tc_bps = st.slider("Transaction cost (bps)", 0, 50, 10)
        mu_source = st.radio("Expected returns come from", ["historical", "lstm", "blend"], horizontal=True, help="historical = past averages, lstm = AI forecast, blend = both.")
        risk_source = st.radio("Risk level driven by", ["vix", "vix+fuzzy"], horizontal=True)
        n_assets = st.slider("Stocks on qubits", 3, 4, 3, help="qubits = stocks x 2 bits x 2 periods (12 or 16)")
        shots = st.select_slider("Shots", [256, 512, 1024], 512)
        p2 = st.select_slider("Physical 2-qubit error", [1e-4, 5e-4, 1e-3, 3e-3, 1e-2], 3e-3)
        mitigation = st.selectbox("Quantum variant", ["qaoa_noisy_mitigated", "qaoa_noisy_raw", "qaoa_logical", "qaoa_ideal"], format_func=lambda m: METHOD_LABELS[m])
        classical = st.multiselect("Compare against", ["exact_qubo", "sa_qubo", "miqp_sub", "cvar_sub", "equal_weight_full"],
                                   default=["exact_qubo", "miqp_sub", "cvar_sub", "equal_weight_full"], format_func=lambda m: METHOD_LABELS[m])
        refresh = st.slider("Live quote refresh (s)", 10, 300, 60)

    with st.expander("🤖 Gemini advisor"):
        gemini_key = st.text_input("Gemini API key", type="password",
                                   help="Kept only in this browser session. Or set env GEMINI_API_KEY / .streamlit/secrets.toml. No key = offline summary.")
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


# ----------------------------------------------------------------------------- load data (three separate paths)
dataset = st.session_state["dataset"]
is_synth, is_real, is_live = dataset == "synthetic", dataset == "real", dataset == "live"
n_new = 0
try:
    if is_synth:
        market = synthetic_market()
        data_label = "SYNTHETIC (simulated market - not real stocks)"
        replay_market = market
    elif is_real:
        market = base_market()
        data_label = f"REAL SNAPSHOT (data/raw + data/esg.csv, last bar {market.returns.index[-1].date()})"
        replay_market = market
    else:
        market, data_label, n_new = live_market_cached()
        replay_market = base_market()
except Exception as e:
    st.error(f"Could not load the **{dataset}** dataset: {e}")
    if is_real or is_live:
        st.info("Real/Live data needs files in `data/raw` and `data/esg.csv`. Run `python scripts/fetch_data.py` then `python scripts/build_esg.py` "
                "(if the Yahoo download fails: `pip install -U yfinance`). Meanwhile click **Synthetic data** in the sidebar.")
    st.stop()

res_real = results("results_real")
tick_all = list(market.returns.columns)
last_bar = market.returns.index[-1].date()

# ----------------------------------------------------------------------------- hero + how it works
mode_name = {"synthetic": "Synthetic", "real": "Real", "live": "Live feed"}[dataset]
hero_cls = {"synthetic": "synth", "real": "", "live": "live"}[dataset]
hero_txt = {"synthetic": "Simulated market, fully offline - good for testing the pipeline.",
            "real": "Saved real prices and ESG scores. Offline - no Yahoo calls.",
            "live": "Real snapshot extended with the latest Yahoo Finance bars, with live re-optimisation."}[dataset]
st.markdown(f"""
<div class="hero {hero_cls}">
  <h1>⚛️ Quantum Portfolio Optimizer</h1>
  <p>Fault-tolerant hybrid quantum-classical portfolio optimiser with dynamic constraints. {hero_txt}</p>
  <span class="pill">Mode: {mode_name}</span>
  <span class="pill">Last bar: {last_bar}</span>
  <span class="pill">{len(tick_all)} stocks in universe</span>
  {'<span class="pill">+' + str(n_new) + ' new bars</span>' if is_live else ''}
  <span class="pill">Research demo - not financial advice</span>
</div>
""", unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns(4)
for col, n, t, d in ((h1, 1, "Choose data", "Synthetic, Real (esg.csv) or Live Yahoo feed."),
                     (h2, 2, "AI forecast", "An LSTM predicts returns; a fuzzy score rates risk."),
                     (h3, 3, "Optimise", "Time-dependent QUBO solved by adaptive QAOA + classical solvers."),
                     (h4, 4, "Advisor", "Best stock is picked from the weights; Gemini explains it.")):
    col.markdown(f'<div class="step"><span class="n">{n}</span><span class="t">{t}</span><div class="d">{d}</div></div>', unsafe_allow_html=True)

with st.expander("ℹ️ Data sources and honest limitations"):
    st.markdown(f"- **Data in use:** {data_label}.\n"
                f"- **Backtest results shown in other tabs:** {'REAL data (Yahoo + FRED + S&P-sourced ESG snapshot)' if res_real['is_real'] else 'SYNTHETIC - not real markets'}.\n"
                "- Quantum solvers run on a **classical simulator**; logical-qubit results use a **modelled** noise channel.\n"
                "- Forecasts are noisy and often add little over a naive baseline. 'Best stock' is a model output, **not financial advice**.\n"
                "- Real mode uses the shipped snapshot (ends 2023-12-29), so the advisor labels its pick STALE. Use **Live feed** for recent bars.")
if is_synth:
    st.info("Synthetic data: stock names and prices are simulated, so the 'best stock' is a demonstration only.")

tabs = st.tabs(["📡 Live feed" if is_live else "📊 Optimiser", "🕰️ Backtests (2008 & 2020)", "🧪 Error-mitigation tests", "🖥️ Hardware needs", "✅ Challenge checklist", "📘 About"])

methods = [mitigation] + classical


def _forecaster(mkt):
    if mu_source == "historical":
        return None
    from stock_prediction.forecast import LSTMForecaster
    return LSTMForecaster(predictor(), mkt.prices) if predictor() is not None else None


def start_job(mkt):
    cfg = make_cfg()
    fc = _forecaster(mkt)
    if mu_source != "historical" and fc is None:
        return False
    st.session_state["job"] = Job(solve_now, mkt, cfg, methods, logical_q=logical_q(cfg), forecaster=fc)
    st.session_state["job_cfg"] = cfg
    st.session_state["last_live_solve"] = time.time()
    st.session_state.pop("quotes_at_solve", None)
    return True


# ============================================================================= MAIN TAB
with tabs[0]:
    # ------------------------------------------------------------------ LIVE-ONLY: feed status, strip, auto re-optimisation
    if is_live:
        section("📡 Live market feed", "Daily bars from Yahoo Finance extend the saved snapshot; quotes are ~15 min delayed. The feed falls back to the snapshot if Yahoo is unreachable.")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Feed status", "Live" if n_new > 0 else "Snapshot only", "new bars received" if n_new > 0 else "no new bars / offline", delta_color="off")
        f2.metric("New bars since snapshot", n_new)
        f3.metric("Last bar", str(last_bar))
        if f4.button("🔄 Refresh feed now", use_container_width=True):
            live_market_cached.clear()
            st.rerun()
        if n_new == 0:
            st.warning("No new Yahoo bars were received, so this is the offline snapshot. Check your internet or update yfinance (`pip install -U yfinance`).")

        strip_t = tick_all[:7] + ["^VIX"]

        def _strip():
            try:
                q, label = DD.market_strip(strip_t)
                q = q.dropna(subset=["last", "chg_pct"])
                if q.empty:
                    raise ValueError("no quotes returned")
                st.caption(f"Market strip - {label} - refreshed {time.strftime('%H:%M:%S')}")
                cols = st.columns(len(q))
                for c, (t, r) in zip(cols, q.iterrows()):
                    c.metric(str(t).replace("^", ""), f"{r['last']:.2f}", f"{r['chg_pct']:+.2f}%")
            except Exception:
                st.caption("Live market strip unavailable right now (Yahoo Finance did not return quotes). The rest of the dashboard still works.")
        st.fragment(run_every=f"{refresh}s")(_strip)()

        st.markdown('<div class="livebox"><b>Streaming re-optimisation</b> - when switched on, the dashboard pulls the newest bars and re-solves the portfolio '
                    'on a timer, and logs each result below.</div>', unsafe_allow_html=True)
        a1, a2 = st.columns([1, 2])
        auto_on = a1.toggle("Auto re-optimise", value=False)
        every_min = a2.select_slider("Every", [1, 5, 15, 30, 60], 5, format_func=lambda m: f"{m} min", disabled=not auto_on)

        def _live_loop():
            if not auto_on:
                return
            job = st.session_state.get("job")
            running = job is not None and job.state == "running"
            if not running and time.time() - st.session_state.get("last_live_solve", 0) >= every_min * 60:
                live_market_cached.clear()
                m, _, _ = live_market_cached()
                if not start_job(m):
                    st.error("Expected returns from LSTM need the model (models/deployed.*). Use 'historical' in Advanced settings.")
            nxt = max(0, int(every_min * 60 - (time.time() - st.session_state.get("last_live_solve", 0))))
            st.caption(f"Auto re-optimisation ON - next run in about {nxt}s")
        st.fragment(run_every="5s" if auto_on else None)(_live_loop)()

    # ------------------------------------------------------------------ optimiser (all modes)
    section("🎯 Step 3 - Your optimised portfolio",
            "Press the button, or click a data mode in the sidebar - the optimisation then starts by itself." if not is_live else
            "Solves on the latest data above. Use 'Re-optimise now' any time, or switch on auto re-optimisation.")
    b1, b2 = st.columns([1, 3])
    solve_clicked = b1.button("▶ Re-optimise now" if is_live else "▶ Run optimisation", type="primary", use_container_width=True)
    b2.caption(f"Mode: **{mode_name}** | expected returns: **{mu_source}** | risk driver: **{risk_source}** | data up to **{last_bar}**")
    auto_start = st.session_state.pop("autostart", False)
    if solve_clicked or auto_start:
        if not start_job(market):
            st.error("Expected returns from LSTM need the model (models/deployed.*). Switch 'Expected returns come from' to historical in Advanced settings.")

    def _opt_panel():
        job = st.session_state.get("job")
        if job is None:
            st.info("No result yet. Click the run button, or choose a data mode in the sidebar.")
            return
        if job.state == "running":
            st.info(f"⏳ Solving ... {job.seconds:.0f}s (the quantum circuit is simulated on your computer - this can take a minute)")
            st.progress(min(job.seconds / 90.0, 0.95))
            return
        if job.state == "error":
            st.error(job.error); return
        r = job.result
        W = pd.DataFrame({METHOD_LABELS[k]: v for k, v in r["weights"].items()}, index=r["tickers"])
        W = W[(W > 1e-6).any(axis=1)]
        avg, n_solvers = top_avg(r)
        top = avg.index[0]
        k1, k2 = st.columns([1.2, 2])
        k1.markdown(f'<div class="pick"><div class="lbl">Top pick (highest average weight)</div><div class="big">{top}</div>'
                    f'<div class="sub">{avg.iloc[0]:.0%} average share across {n_solvers} solvers</div>'
                    f'<div style="margin-top:.5rem">' + "".join(f'<span class="chip">{t} {v:.0%}</span>' for t, v in avg.iloc[1:4].items()) + '</div></div>',
                    unsafe_allow_html=True)
        m1, m2, m3 = k2.columns(3)
        m1.metric("Stocks chosen", int((avg > 1e-6).sum()))
        m2.metric("Qubits used", r["n_qubits"])
        m3.metric("Solve time", f"{job.seconds:.1f}s")
        k2.caption(f"Optimised on assets: {', '.join(r['sub'])}. Weights as of {r['date'].date()}.")

        c1, c2 = st.columns([3, 2])
        fig = go.Figure([go.Bar(name=c, x=W.index, y=W[c]) for c in W.columns])
        style_fig(fig, 340, barmode="group", yaxis_title="share of portfolio", yaxis_tickformat=".0%", title="How each method splits the money")
        c1.plotly_chart(fig, width="stretch")
        fig2 = go.Figure(go.Bar(x=avg.values[::-1], y=avg.index[::-1], orientation="h", marker_color=COLORS[0]))
        style_fig(fig2, 340, xaxis_tickformat=".0%", title="Average share across methods")
        c2.plotly_chart(fig2, width="stretch")

        with st.expander("📋 Detailed numbers per method"):
            st.dataframe(DD.expected_stats(r).rename(index=METHOD_LABELS).style.format(
                {"exp_return_21d": "{:.2%}", "vol_21d": "{:.2%}", "exp_sharpe": "{:.2f}", "turnover": "{:.2f}", "solve_seconds": "{:.2f}"}), width="stretch")
            st.caption("Expected return / volatility / Sharpe are MODEL-IMPLIED for this instance, not realised performance.")

        if is_live:                                                   # log every live re-optimisation + remember quotes for mark-to-market
            logged = st.session_state.setdefault("_logged", set())
            if id(job) not in logged:
                logged.add(id(job))
                st.session_state.setdefault("live_hist", []).append(
                    {"solved at": time.strftime("%H:%M:%S"), "last bar": str(r["date"].date()), "top pick": top, "top share": f"{avg.iloc[0]:.0%}",
                     "runner-up": avg.index[1] if len(avg) > 1 else "-", "solve s": round(job.seconds, 1)})
            if "quotes_at_solve" not in st.session_state:
                try:
                    from stock_prediction import data as D
                    st.session_state["quotes_at_solve"] = D.fetch_quotes(tuple(r["tickers"]))["last"]
                except Exception:
                    st.session_state["quotes_at_solve"] = None
    st.fragment(run_every="1s" if (st.session_state.get("job") is not None and st.session_state["job"].state == "running") else None)(_opt_panel)()

    if is_live:
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
            st.caption("Change since the last optimisation (live quotes, no costs): " + "; ".join(f"{k}: {v:+.3%}" for k, v in rows.items()))
        st.fragment(run_every=f"{refresh}s")(_mtm)()

        hist = st.session_state.get("live_hist", [])
        if hist:
            with st.expander(f"🗂️ Re-optimisation log ({len(hist)} runs)", expanded=True):
                st.dataframe(pd.DataFrame(hist[::-1]), width="stretch", hide_index=True)

    # ------------------------------------------------------------------ advisor (all modes)
    section("🤖 Step 4 - AI advisor: best stock and plain-English summary",
            "The best stock is computed from the optimiser weights; Gemini only writes the explanation (or an offline template if there is no key / Gemini is busy).")

    def _advisor_panel():
        job = st.session_state.get("job")
        if job is not None and job.state in ("done", "error") and st.session_state.get("_adv_rerun_for") != id(job):
            st.session_state["_adv_rerun_for"] = id(job)
            st.rerun()      # one full rerun when the solve finishes, so every panel sees the result
        advisor_ui.render_panel(st, job, res_real, st.session_state.get("job_cfg", make_cfg()), _gemini_key(), gemini_model.strip(), esg=market.esg)
    _jb = st.session_state.get("job")
    st.fragment(run_every="1s" if (_jb is not None and _jb.state == "running") else None)(_advisor_panel)()

    # ------------------------------------------------------------------ prediction panel (data source depends on mode)
    section("🔮 Stock outlook - AI forecast for one stock", "Pick a ticker to see its recent price, a 7-day LSTM forecast, a risk gauge and a signal.")
    pc1, pc2 = st.columns([1, 3])
    sym = pc1.selectbox("Ticker", tick_all)
    other = pc1.text_input("...or another symbol (Live feed only)", "", disabled=not is_live)
    sym = other.strip().upper() if (is_live and other.strip()) else sym
    pr = predictor()
    if pr is None:
        pc2.error("No trained model in models/. Run `python scripts/train_predictor.py` (training is offline - the app never trains on request).")
    else:
        from stock_prediction import data as D
        from stock_prediction.service import analyse_symbol
        px, src = None, ""
        if is_live:
            try:
                px, src = D.fetch_prices((sym,), start="2022-01-01")[sym], "Yahoo Finance (live)"
            except Exception:
                px, src = (market.prices[sym] if sym in market.prices.columns else None), "snapshot (Yahoo unavailable)"
        elif is_real:
            try:
                px, src = D.load_snapshot()[sym], "saved real snapshot"
            except Exception:
                px, src = (market.prices[sym] if sym in market.prices.columns else None), "saved real snapshot"
        else:
            px, src = (market.prices[sym] if sym in market.prices.columns else None), "synthetic data"
        if px is None:
            pc2.error("No data for this symbol in the current mode.")
        else:
            vz = float((market.macro["VIX"].iloc[-1] - market.macro["VIX"].iloc[-252:].mean()) / (market.macro["VIX"].iloc[-252:].std() + 1e-9))
            try:
                a = analyse_symbol(sym, px, pr, vix_z=vz)
                fig = go.Figure(go.Scatter(x=px.index[-180:], y=px.iloc[-180:], name="Close", line=dict(color=COLORS[0], width=2)))
                fig.add_trace(go.Scatter(x=a["dates"], y=a["path"], name="7-day forecast", line=dict(dash="dot", color=COLORS[1], width=3)))
                fig.add_trace(go.Scatter(x=list(a["dates"]) + list(a["dates"])[::-1], y=list(a["hi"]) + list(a["lo"])[::-1], fill="toself", mode="lines", opacity=0.25,
                                         line=dict(width=0), name="90% band", fillcolor="rgba(6,182,212,.25)"))
                style_fig(fig, 340, title=f"{sym}: price ({src}) and 7-day forecast")
                pc2.plotly_chart(fig, width="stretch")
                g1, g2, g3, g4 = st.columns(4)
                gauge = go.Figure(go.Indicator(mode="gauge+number", value=a["risk"]["score"] * 100, title={"text": f"Risk - {a['risk']['label']}"},
                                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS[0]},
                                                      "steps": [{"range": [0, 33], "color": "#bbf7d0"}, {"range": [33, 66], "color": "#fde68a"}, {"range": [66, 100], "color": "#fecaca"}]}))
                gauge.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
                g1.plotly_chart(gauge, width="stretch")
                g2.metric("Signal", a["signal"], f"{a['exp_ret_21d_pct']:+.2f}% expected 21d")
                g3.metric("Confidence", a["confidence_label"], f"{a['confidence']:.2f}")
                g4.metric(f"Skill vs naive ({a['metrics_split']})", f"{a['skill_h1']:+.3f}" if a["skill_h1"] is not None else "n/a",
                          "negative = worse than 'tomorrow = today'", delta_color="off")
                st.caption(f"Model trained through {a['model_train_end']} (offline). Direction accuracy {a['dir_acc_h1']:.3f} vs base rate {a['dir_base_h1']:.3f}. "
                           "Skill near zero means the forecast adds little beyond the naive baseline.")
            except Exception as e:
                pc2.error(f"Could not analyse {sym}: {e}")

    # ------------------------------------------------------------------ Synthetic / Real: historical replay (NOT the live feed)
    if not is_live:
        with st.expander("⏯️ Historical replay (simulated stream of past bars)"):
            st.caption("Replays past daily bars tick by tick and re-optimises as it goes - a simulated stream, not a live feed. For the real live feed choose **Live feed** in the sidebar.")
            s1, s2 = st.columns(2)
            start = s1.selectbox("Replay start", ["2008-06-02 (GFC)", "2020-01-02 (COVID)", "2015-01-02 (calm)"]).split(" ")[0]
            n_ticks = s2.slider("Ticks to stream", 20, 200, 60)
            reb = s2.slider("Re-optimise every N ticks", 10, 63, 21)
            if st.button("Start replay"):
                cfg = make_cfg(rebalance_days=reb)
                fc = None
                if mu_source != "historical":
                    from stock_prediction.forecast import LSTMForecaster
                    mb = model_bank()
                    fc = LSTMForecaster(mb, replay_market.prices) if mb is not None else None
                lo = LiveOptimizer(replay_market, cfg, start, methods, logical_q(cfg), forecaster=fc)
                st.session_state["stream"] = StreamJob(lo, n_ticks, delay=0.05)

            def _stream():
                sj = st.session_state.get("stream")
                if sj is None:
                    st.info("Press Start replay.")
                    return
                sn = sj.snapshot()
                if sn["error"]:
                    st.error(sn["error"]); return
                if len(sn["pnl"]) > 1:
                    fig = go.Figure([go.Scatter(x=sn["pnl"].index, y=sn["pnl"][c], name=METHOD_LABELS.get(c, c)) for c in sn["pnl"].columns])
                    style_fig(fig, 320, yaxis_title="portfolio value (start = 1)")
                    st.plotly_chart(fig, width="stretch")
                st.progress(min(sn["t"] / sn["n"], 1.0), text=f"{sn['state']}  tick {sn['t']}/{sn['n']}")
            st.fragment(run_every="1s")(_stream)()

    # ------------------------------------------------------------------ quantum internals (all modes)
    with st.expander("🔬 Behind the scenes: quantum circuit and noise numbers"):
        job = st.session_state.get("job")
        if job is not None and job.state == "done":
            cfg = st.session_state.get("job_cfg", make_cfg())
            p_used = int(next((v.get("p") for v in job.result["info"].values() if v.get("p")), cfg.p_max))
            qi = DD.quantum_internals(job.result, cfg, p_used, res_real)
            c = st.columns(5)
            c[0].metric("Logical qubits", qi["qubits"]); c[1].metric("QAOA layers p", qi["p_layers"]); c[2].metric("CNOTs / depth (est.)", f"{qi['cnots']} / {qi['depth_estimate']}")
            c[3].metric("Flip prob. (raw / DD)", f"{qi['q_eff_raw']:.3f} / {qi['q_eff_dd']:.3f}")
            if "logical_q_per_layer" in qi:
                c[4].metric(f"Logical error/layer (d={qi['distance']})", f"{qi['logical_q_per_layer']:.1e}", f"Stim fit p_th={qi['p_th_fit']:.4f}", delta_color="off")
            if "resource_row" in qi:
                st.caption(f"Resource estimate (first corner of results_real/resource_corners.csv): {qi['resource_row']}")
            st.caption("Logical error rates come from Stim/PyMatching surface-code MEMORY simulations; QAOA receives them as an effective bit-flip rate (modelled). "
                       "Resource-estimator constants are assumptions.")
        else:
            st.info("Run the optimisation first to see circuit and noise numbers.")

# ============================================================================= BACKTEST
with tabs[1]:
    section("🕰️ Out-of-sample crisis backtests", "How quantum and classical (Gurobi/MIQP, CVaR, RL) solvers behave in the 2008 crisis and the 2020 COVID crash (results_real/).")
    for p in ("GFC_2008", "COVID_2020"):
        s = res_real[f"summary_{p}"]
        if s is None:
            continue
        st.markdown(f"**{p.replace('_', ' ')}**")
        st.dataframe(s.style.format("{:.3f}"), width="stretch")
        t = res_real[f"signif_{p}"]
        if t is not None:
            st.caption(f"Significant quantum-vs-classical Sharpe differences (Holm-adjusted bootstrap p<0.05): {int((t.p_boot_holm < 0.05).sum())} / {len(t)}")
        if res_real[f"fig_{p}"]:
            st.image(res_real[f"fig_{p}"])
    lstm = results("results_lstm")
    section("Prediction-system variants", "Does adding the LSTM forecast help? (results_lstm/)")
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
    section("🧪 Ablation of error-mitigation techniques", "Which of dynamical decoupling (DD), zero-noise extrapolation (ZNE) and probabilistic error cancellation (PEC) help the quantum solver, and how cost grows with size.")
    if res_real["ablation"] is not None:
        st.dataframe(res_real["ablation"].style.format("{:.4f}"), width="stretch")
    if "ablation" in res_real["figs"]:
        st.image(res_real["figs"]["ablation"])
    if res_real["scaling"] is not None:
        st.markdown("**Scaling: shots/evaluations to reach the optimum vs qubits**")
        st.dataframe(res_real["scaling"].style.format("{:.2f}"), width="stretch")
        if res_real["scaling_fits"] is not None:
            st.caption("Fitted exponents (cost ~ 2^(slope*n), n=6..16 only - indicative): " + ", ".join(f"{i}: {v:.2f}" for i, v in res_real["scaling_fits"].iloc[:, 0].items()))
        if "scaling" in res_real["figs"]:
            st.image(res_real["figs"]["scaling"])

# ============================================================================= RESOURCES
with tabs[3]:
    section("🖥️ Logical-qubit resource estimation", "Rough estimate of the error-corrected quantum computer this problem would need (assumption-based).")
    if res_real["resource_corners"] is not None:
        st.dataframe(res_real["resource_corners"].round(3), width="stretch")
    for k in ("resources", "tornado", "qec"):
        if k in res_real["figs"]:
            st.image(res_real["figs"][k])
    if res_real["resource_assumptions"] is not None:
        with st.expander("Assumptions (constants are assumptions, swept in results_real/resource_*)"):
            st.dataframe(res_real["resource_assumptions"], width="stretch")

# ============================================================================= CHALLENGE CHECKLIST
with tabs[4]:
    section("✅ QC-01 challenge checklist", "Where each requirement of the problem statement lives in this project.")
    rows = [
        ("Multi-period, time-dependent QUBO/Ising with soft + hard constraints", "Optimiser / Live feed tab; sidebar Step 2",
         "Time-varying risk aversion and weight caps driven by VIX, transaction-cost term, hard ESG exclusion + soft ESG tilt (qpo/qubo.py)"),
        ("Adaptive QAOA on logical qubits (simulated error-corrected backend)", "Quantum variant = 'QAOA logical qubits (modelled)'; Behind the scenes",
         "Logical error rates from Stim + PyMatching surface-code memory runs; fed to QAOA as an effective bit-flip rate (modelled, not full code-level)"),
        ("Dynamical decoupling, ZNE and PEC", "Error-mitigation tests tab", "Ablation of each technique (qpo/qaoa.py, qpo/noise.py)"),
        ("Compare against Gurobi / classical / RL solvers", "Backtests tab; 'Compare against' in Advanced settings",
         "Exact search, SA, CVaR (HiGHS/Gurobi), MIQP (Gurobi, else exact DP), RL agent. Gurobi branches only run if gurobipy is installed"),
        ("Out-of-sample crisis periods", "Backtests tab", "2008 GFC and 2020 COVID walk-forward windows (no look-ahead)"),
        ("Statistical significance tests", "Backtests tab", "Paired bootstrap on Sharpe, Wilcoxon, Holm correction (qpo/stats.py)"),
        ("Resource estimation report for logical qubits", "Hardware needs tab", "Code distance, physical qubits, T-count, runtime; constants are assumptions with sweeps"),
        ("Interactive dashboard with live re-optimisation under streaming feeds", "Live feed mode", "Yahoo daily bars + delayed quotes; auto re-optimisation timer and log. 'Live' = daily-bar granularity"),
        ("Datasets: Yahoo Finance, FRED, public ESG scores", "Real data / Live feed modes", "Prices + VIX/FRED in data/raw, ESG in data/esg.csv (static snapshot)"),
        ("Stack: Qiskit/Cirq, Stim, Ocean SDK, Gurobi/CPLEX, PyTorch, Streamlit", "Whole project", "Stim, PyTorch RL, Streamlit, Gurobi (optional) and Ocean/Qiskit demos (scripts/optional_*.py)"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Requirement", "Where in the dashboard", "How it is covered"]), width="stretch", hide_index=True)
    st.caption("Be honest in your report: the quantum parts run on a classical simulator and QAOA does not beat exact search or SA at these sizes; see the README limitations.")

# ============================================================================= ABOUT
with tabs[5]:
    st.markdown(open("docs/ABOUT_DASHBOARD.md").read() if os.path.exists("docs/ABOUT_DASHBOARD.md") else "See README.md")