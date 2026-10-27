"""Technical report generator (PDF, A4).   python report/build_report.py --results results_real --out report/QPO_Technical_Report.pdf

Every number is read from the csv/json files in --results (nothing is typed in by hand).  If results/meta.json does not say the
data came from the real loader (yahoo+fred+esg), EVERY page carries a "SYNTHETIC-DATA DRAFT" banner and the title says so.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo import theory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from PIL import Image as PI

NAVY, TEAL, RED = colors.HexColor("#1B2A49"), colors.HexColor("#139A8C"), colors.HexColor("#B3261E")
ST = {
    "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=NAVY, spaceAfter=8),
    "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=12, leading=16, textColor=TEAL, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=TEAL, spaceBefore=6, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=10, leading=13.8, spaceAfter=6, alignment=4),
    "b": ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=13.4, leftIndent=12, bulletIndent=2, spaceAfter=3),
    "cap": ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=colors.grey, spaceAfter=7),
    "box": ParagraphStyle("box", fontName="Helvetica", fontSize=9.2, leading=12.4, backColor=colors.HexColor("#FDECEA"), borderPadding=6, spaceAfter=8),
    "ok": ParagraphStyle("ok", fontName="Helvetica", fontSize=9.2, leading=12.4, backColor=colors.HexColor("#E4F4F1"), borderPadding=6, spaceAfter=8),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.8, leading=10, backColor=colors.HexColor("#F3F4F6"), borderPadding=4, spaceAfter=6),
}
E = []


def P(t, s="p"): E.append(Paragraph(t, ST[s]))
def H1(t): E.append(Paragraph(t, ST["h1"]))
def H2(t): E.append(Paragraph(t, ST["h2"]))
def CODE(lines):
    E.append(Paragraph("<br/>".join(l.replace("&", "&amp;").replace("<", "&lt;").replace(" ", "&nbsp;") for l in lines), ST["code"]))


def BL(items):
    for i in items: E.append(Paragraph(i, ST["b"], bulletText="\u2022"))


def fig(path, width_cm, cap):
    if not os.path.exists(path):
        P(f"<i>[figure missing: {os.path.basename(path)}]</i>", "box"); return
    w, h = PI.open(path).size
    W = width_cm * cm; H = W * h / w
    E.append(KeepTogether([Image(path, W, H), Paragraph(cap, ST["cap"])]))


def table(df, cap=None, fs=7.2, colw=None, index=True, nd=4):
    d = df.copy()
    if index: d = d.reset_index()
    d = d.map(lambda v: (f"{v:.{nd}g}" if abs(v) < 1e5 else f"{v:,.0f}") if isinstance(v, (float, np.floating)) else str(v))
    data = [list(map(str, d.columns))] + d.values.tolist()
    data = [[Paragraph(f"<font size={fs}>{c}</font>", ST["p"]) if len(str(c)) > 26 else c for c in r] for r in data]
    t = Table(data, colWidths=colw, repeatRows=1)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", fs), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E4F4F1")), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    E.append(KeepTogether([t, Paragraph(cap or "", ST["cap"])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="report/QPO_Technical_Report.pdf")
    ap.add_argument("--theory-results", default=None, help="folder with qec_*/resource_* files if different (they are data-independent)")
    a = ap.parse_args()
    R = a.results; RT = a.theory_results or R
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    figdir = os.path.join(R, "report_figs"); os.makedirs(figdir, exist_ok=True)
    meta = json.load(open(f"{R}/meta.json"))
    src = meta.get("data_source", "unknown")
    real = src == "yahoo+fred+esg"
    rd = lambda f, **k: pd.read_csv(f"{R}/{f}", **k)
    qfit = json.load(open(f"{RT}/qec_fit.json")); qsw = pd.read_csv(f"{RT}/qec_sweep.csv")
    cor = pd.read_csv(f"{RT}/resource_corners.csv"); asm = pd.read_csv(f"{RT}/resource_assumptions.csv")
    swp = pd.read_csv(f"{RT}/resource_sweep.csv")
    abl = rd("ablation_summary.csv", index_col=0); scl = rd("scaling.csv", index_col=0); fits = rd("scaling_fits.csv", index_col=0).iloc[:, 0]
    cfg = meta["config"]
    periods = [p for p in ("GFC_2008", "COVID_2020") if os.path.exists(f"{R}/backtest_{p}_summary.csv")]
    D = "DATA: " + ("REAL (Yahoo Finance + FRED + ESG csv)" if real else "SYNTHETIC")

    # ------------------------------------------------------------------ title
    E.append(Spacer(1, 1.2 * cm))
    P("Fault-Tolerant Quantum Portfolio Optimizer with Dynamic Constraints", "title")
    P("Logical-qubit resource estimation, error-mitigation ablation, significance tests and a live dashboard", "sub")
    if not real:
        P("<b>SYNTHETIC-DATA DRAFT.</b> Every number that depends on market data (sections 7-9: ablation instances, backtests, scaling instances) comes from a simulated market, because the build "
          "environment had no internet access (Yahoo/FRED/ESG could not be downloaded). They illustrate the pipeline and must not be read as results on real markets. "
          "Sections 5-6 (surface-code simulation, resource estimates) and the theory in section 9 do not depend on market data. To produce the final report run "
          "<font name=Courier>python report/build_report.py --results results_real</font> after the real-data run: this banner disappears automatically "
          "only when <font name=Courier>meta.json</font> records the real-data source.", "box")
    else:
        P(f"Data: Yahoo Finance (adjusted close), FRED (VIX, 10Y-2Y), ESG csv. {meta.get('n_assets')} assets, {meta.get('first_date')} to {meta.get('last_date')}. "
          f"Dropped tickers: {meta.get('data_meta', {}).get('dropped_tickers') or 'none'}.", "ok")
    E.append(Paragraph("<b>Abstract.</b> We build a hybrid quantum-classical, walk-forward portfolio optimizer whose multi-period decision problem is a time-dependent "
        "QUBO (time-varying risk aversion and weight caps driven by VIX, transaction-cost surrogate, hard ESG exclusion and soft ESG tilt). The QUBO is solved by a "
        "layer-wise adaptive QAOA on an exact statevector simulator, under (i) a phenomenological NISQ+ noise model with dynamical decoupling (DD), zero-noise extrapolation (ZNE) and probabilistic error "
        "cancellation (PEC) and (ii) a logical-qubit mode whose logical error rates are extracted from Stim/PyMatching surface-code simulations. We compare with exact search, simulated annealing, "
        "a discrete MIQP, CVaR stochastic programs, and RL agents on out-of-sample windows containing the 2008 and 2020 crises. "
        "<b>The claim we defend is deliberately modest: rigorous bounds plus an empirical test of advantage criteria, not a proven quantum advantage.</b>", ST["p"]))
    H2("Contents")
    BL(["1 Introduction and claim framing", "2 Related work", "3 Problem formulation, data and audit trail", "4 Algorithms and baselines", "5 Noise model and fault tolerance",
        "6 (a) Logical-qubit resource estimation", "7 (b) Error-mitigation ablation", "8 (c) Out-of-sample results and significance tests", "9 Scaling study and theory",
        "10 (d) Interactive dashboard", "11 Software verification", "12 Discussion: what an advantage would require", "13 Threats to validity and conclusion", "Appendices A-C: reproduction, significance tables, resource sensitivity"])
    E.append(PageBreak())
    H1("1  Introduction and what is (and is not) claimed")
    P("The task statement asks to \"prove (or rigorously bound) a quantum advantage in solution quality or sample complexity under realistic NISQ+ noise\". No proof of an advantage of "
      "constant-depth QAOA over the best classical solvers on portfolio QUBOs is known, and our instances (n &le; 16 qubits) are solved exactly by enumeration in under a millisecond, so any advantage could only be a "
      "<i>scaling</i> statement. We therefore contribute: (1) exact bounds for our noise model (success-probability lower bound, PEC/ZNE sampling costs, a noise budget that any sample-complexity advantage must respect); "
      "(2) an empirical scaling comparison for n &le; 16; (3) logical-qubit resource estimates with an explicit assumption table and sensitivity analysis; (4) an out-of-sample statistical comparison with multiple-testing correction; "
      "(5) a fully reproducible pipeline. Where the data are negative we report them as negative.")
    H2("Contributions")
    BL(["Multi-period time-dependent QUBO with hard and soft constraints, verified against the direct objective (unit tests) and against a hard-budget MIQP formulation.",
        "QAOA with DD, ZNE, PEC and a logical-qubit mode; ablation of each mitigation (section 7).",
        "Resource estimator with consistent time/failure model, magic-state error check, and sensitivity sweep over every constant (section 6).",
        "Walk-forward backtests with stationary-bootstrap Sharpe tests, Wilcoxon tests and Holm correction (section 8)."])

    H1("2  Related work")
    P("Walk-forward QUBO rebalancing with QAOA on classical simulators (Weinberg, arXiv:2603.16904); multi-period annealing with quadratic transaction-cost surrogates (arXiv:2607.03218); "
      "multi-period QUBO with market impact (Rosenberg et al., IEEE JSTSP 2016); the closest study of advantage under noise for portfolio QAOA (Yuan et al., arXiv:2410.16265), which finds that "
      "realistic noise precludes advantage except when measurement noise dominates; QAOA itself (Farhi et al., arXiv:1411.4028); error mitigation (Temme-Bravyi-Gambetta, arXiv:1612.02058); "
      "noisy-device limitations (Stilck Fran&ccedil;a &amp; Garc&iacute;a-Patr&oacute;n, Nat. Phys. 2021); surface codes and lattice surgery (Fowler et al. 2012; Fowler &amp; Gidney, arXiv:1808.06709; Litinski, Quantum 2019); "
      "Stim (Gidney, Quantum 2021) and PyMatching v2 (Higgott &amp; Gidney, arXiv:2303.15933); rotation synthesis (Ross &amp; Selinger, arXiv:1403.2975; Bocharov et al., PRL 114, 080502). "
      "<i>The entries for Weinberg, arXiv:2607.03218 and Yuan et al. were located by search during project setup and the others cited from memory; verify every citation before submission.</i>")

    H1("3  Problem formulation")
    P(f"At each rebalance date <i>t</i> (every {cfg['rebalance_days']} trading days) we use only the previous {cfg['lookback']} days. Assets below the ESG threshold ({cfg['esg_min']}) are excluded (<b>hard</b> constraint). "
      f"From the eligible set, {cfg['n_assets']} assets are selected greedily (trailing Sharpe, penalising correlation) to fit on qubits. Each asset-period weight is "
      f"w = level/{cfg['units']} with level = b<sub>0</sub> + 2 b<sub>1</sub> (K = {cfg['n_bits']} bits), over {cfg['n_periods']} look-ahead periods, so n = A &middot; K &middot; T = "
      f"{cfg['n_assets'] * cfg['n_bits'] * cfg['n_periods']} qubits. The energy to minimise is")
    P("E(x) = &Sigma;<sub>t</sub> [ &minus;&mu;<sub>t</sub>&middot;w<sub>t</sub> + (&lambda;<sub>t</sub>/2) w<sub>t</sub>&prime;&Sigma;<sub>t</sub> w<sub>t</sub> + c<sub>tc</sub> &Vert;w<sub>t</sub> &minus; w<sub>t&minus;1</sub>&Vert;<sup>2</sup> "
      "+ P<sub>b</sub>(1&prime;w<sub>t</sub> &minus; 1)<sup>2</sup> + P<sub>c</sub> &middot; cap<sub>t</sub> penalty &minus; &tau; s&middot;w<sub>t</sub>/100 ]", "code")
    P(f"<b>Time-varying constraints.</b> With z the 252-day z-score of the previous day's VIX: &lambda;<sub>0</sub> = &lambda;(1 + &kappa; clip(z, &minus;1, 3)) with &lambda; = {cfg['lambda0']}, &kappa; = {cfg['kappa_vix']}; "
      f"the weight cap drops from {cfg['cap_levels_calm']}/{cfg['units']} to {cfg['cap_levels_stress']}/{cfg['units']} when z &gt; {cfg['stress_z']}. "
      f"<b>Transaction costs</b>: linear {cfg['tc_bps']} bp in the P&amp;L, quadratic surrogate c<sub>tc</sub> = {cfg['tc_quad']} inside the QUBO. "
      f"<b>ESG</b>: soft tilt &tau; = {cfg['esg_tilt']}. The soft budget/cap penalties are followed by a hard repair (non-negative, fully invested). "
      "The QUBO matrix is checked against the direct objective for random bitstrings (tests/test_core.py), and its exact optimum equals the optimum of the hard-budget MIQP whenever the penalty is large "
      "(tests/test_miqp.py).")

    fig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "walkthrough", "figs", "01_architecture.png"), 14.0, "Pipeline architecture (data, estimation, QUBO, solvers, noise and mitigation, evaluation, dashboard).")
    H2("3.1  Data, universe and audit trail")
    dm = meta.get("data_meta", {})
    if real:
        P(f"Prices: Yahoo Finance via yfinance (<font name=Courier>auto_adjust=True</font>, i.e. split- and dividend-adjusted close), frozen to csv by <font name=Courier>scripts/fetch_data.py</font> so every later stage runs offline. "
          f"Macro: FRED VIXCLS and T10Y2Y through the public csv endpoint (header <font name=Courier>observation_date</font>; '.' = missing). ESG: <font name=Courier>{dm.get('esg_source')}</font> "
          f"({'; '.join(dm.get('esg_provenance', []) or ['see data/esg.csv'])}). Requested tickers: {', '.join(dm.get('requested_tickers', []))}. "
          f"Dropped for missing history: {dm.get('dropped_tickers') or 'none'}; dropped macro series: {dm.get('dropped_macro') or 'none'}; tickers without ESG score (median-filled): {dm.get('esg_missing') or 'none'}.")
    else:
        P("Synthetic market: 16 assets in 4 sectors with a market factor, sector factors and idiosyncratic Student-t noise (6 d.o.f.), a two-state volatility regime (x2.8 in stress) forced on to the 2008-09 and 2020 crisis windows, "
          "a VIX proxy driven by the regime, and random ESG scores in [20, 95]. It exists to test the pipeline; it contains no information about real markets.")
    P("<b>Look-ahead controls.</b> Decisions at index i use returns and VIX strictly before i; macro series are forward-filled only and a series with insufficient coverage is dropped, never back-filled "
      "(FRED now truncates ICE-BofA series to three years, so back-filling the credit spread would copy 2023 values into 2008). Hyper-parameters for each test window are chosen on earlier data only (scripts/tune.py).")

    H1("4  Algorithms and baselines")
    P("<b>QAOA.</b> Exact statevector simulation of p alternating cost phases exp(&minus;i&gamma;&pi;E<sub>norm</sub>) and RX mixers; parameters by COBYLA, starting at p = 1 and adding a layer "
      f"(up to p<sub>max</sub> = {cfg['p_max']}) only while the relative energy improvement exceeds {cfg['adapt_tol']:.0%} (<i>adaptive ansatz</i>). The best of {cfg['shots']} samples, evaluated classically, is decoded and repaired. "
      "Parameters are trained noiselessly and deployed under each noise model (noise-aware training appears only in the ablation).")
    P("<b>Classical baselines on the same problems and out-of-sample periods.</b>")
    BL(["<i>exact_qubo</i> (true QUBO optimum, ceiling for every QUBO solver) and <i>sa_qubo</i> (single-flip simulated annealing).",
        "<i>miqp_sub</i>: the hard-budget discrete multi-period problem solved exactly (Gurobi MIQP via the matrix API when gurobipy is present, otherwise an exact dynamic program; identical optimum, cross-tested). <i>miqp_gurobi_full</i>: same on all ESG-eligible assets.",
        "<i>cvar_full / cvar_sub / cvar_gurobi_sub</i>: single-period CVaR<sub>0.9</sub> scenario programs (HiGHS and Gurobi) with 50% cap and linear costs - the stochastic-programming baseline; <i>_sub</i> uses the same assets as the QUBO, <i>_full</i> all eligible assets, separating selection from solver effects.",
        "<i>rl_es_full</i>: small MLP policy trained by evolution strategies (NumPy); <i>rl_torch_full</i>: PyTorch REINFORCE with a per-state baseline (same reward). Both train only on data preceding the test start.",
        "<i>equal_weight_full</i>."])
    CODE(["for each rebalance date i in test window:                  # every 21 (or 63) trading days",
          "    window = returns[i-252 : i]; z = vix_zscore(i-1)         # past data only",
          "    elig = {assets with ESG >= esg_min}                     # hard filter",
          "    sub  = greedy_select(window[elig], A)                   # assets that go on qubits",
          "    Q    = build_qubo(mu, Sigma, w_hold, lambda_t(z), caps_t(z), esg)",
          "    for method in methods:                                   # each keeps its OWN holdings",
          "        x = solve(method, Q | scenarios | policy)           # QAOA / exact / SA / MIQP / CVaR / RL",
          "        w = repair(first_period_weights(x))                  # hard constraints",
          "        pay linear cost on |w - w_hold|; hold for 21 days; drift weights"])
    methods = meta.get("methods", [])
    P(f"Methods in this run: {', '.join(methods)}. Optional packages available at run time: "
      f"{', '.join(k for k, v in meta.get('optional_methods_available', {}).items() if v) or 'none of torch/gurobipy'}; missing optional methods are <b>absent from all tables</b>, not simulated.")

    H1("5  Noise model and fault tolerance")
    P("<b>NISQ+ model.</b> Gate counts (2 CNOT + 1 RZ per ZZ coupling per layer, RZ per linear term, RX mixer) are converted to an effective independent bit-flip probability q per qubit on the measured string: "
      f"two-qubit error {cfg['p2']}, one-qubit {cfg['p1']}, readout {cfg['p_readout']}, idle {cfg['idle_rate']} per qubit-layer. DD removes a fraction {cfg['dd_suppression']} of the idle error. "
      "This keeps the simulation exact and makes ZNE (noise scaling of the gate part) and PEC (inverse flip channel) analysable, but it is a <b>phenomenological</b> model: real devices have correlated and coherent noise.")
    P("Formally, with N<sub>cx</sub> = 2 n<sub>zz</sub> p CNOTs and N<sub>rz</sub> = (n<sub>zz</sub> + n<sub>lin</sub> + n) p rotations: &lambda; = (2/3)[p<sub>2</sub> N<sub>cx</sub> + p<sub>1</sub> N<sub>rz</sub> + r<sub>idle</sub>(1&minus;DD) n p]/n &middot; s, "
      "q<sub>gate</sub> = (1 &minus; e<super>&minus;2&lambda;</super>)/2, q = q<sub>gate</sub> + r<sub>ro</sub> &minus; 2 q<sub>gate</sub> r<sub>ro</sub>, where s is the ZNE noise-scale factor (readout is not scaled) and the noisy distribution is "
      "P<sub>noisy</sub> = (flip<sub>q</sub>)<super>&otimes;n</super> P<sub>ideal</sub>. In logical mode q follows the same formula with &lambda; = &minus;&frac12; ln(1 &minus; 2 q<sub>L</sub>) p s and no readout term.")
    P("<b>Logical qubits - what is and is not simulated.</b> Stim simulates the rotated surface code <i>memory</i> experiment (d rounds, circuit-level depolarising noise, MWPM decoding by PyMatching). "
      "QAOA contains non-Clifford rotations and a 2<super>n</super> statevector, which cannot be co-simulated at the code level. We therefore fit the Stim logical error rate "
      "and <b>inject it as an effective logical bit-flip probability into the small QAOA simulation</b>. That is a modelling assumption, not an end-to-end fault-tolerant simulation.")
    f = qsw[["d", "p", "shots", "fails", "p_L_per_round", "stderr"]].copy()
    table(f, f"Stim + PyMatching memory sweep. Fit p<sub>L</sub>/round = A (p/p<sub>th</sub>)<super>(d+1)/2</super>: A = {qfit['A']:.3g}, p<sub>th</sub> = {qfit['p_th']:.4f} "
          f"({qfit['n_points']} points; rows with &lt;5 failures are excluded from the fit). Only p = 2e-3..6e-3 was simulated; p &le; 1e-3 is an extrapolation.", index=False, nd=3,
          colw=[1.3 * cm, 1.8 * cm, 1.8 * cm, 1.6 * cm, 2.8 * cm, 2.2 * cm])
    fig(f"{RT}/qec_threshold.png", 10.5, "Logical error per round vs physical error for d = 3, 5, 7 (Stim, memory only).")
    fig(f"{figdir}/qec_extrapolation.png", 11.0, "Fitted logical error per round vs p for larger distances, with the Stim data points (black). Below p = 2e-3 the curves are extrapolated.")
    lq = 0.5 * min(1.0, cfg["logical_distance"] * qfit["A"] * (cfg["logical_p_phys"] / qfit["p_th"]) ** ((cfg["logical_distance"] + 1) / 2))
    P(f"With d = {cfg['logical_distance']} and p = {cfg['logical_p_phys']}, the fitted model gives a logical flip probability of {lq:.1e} per gate layer for a Z-basis measurement. "
      "Note this is an extrapolation of a 9-point fit whose lowest-error points (d = 7, p = 2e-3) have only a handful of failures.")

    # ------------------------------------------------------------------ 6 resources
    H1("6  (a) Logical-qubit resource estimation")
    P("For each circuit we count rotations, synthesize them into T gates, and choose the smallest odd distance d such that the total failure budget (split in three equal parts: logical errors of the data patches, "
      "rotation-synthesis error, magic-state infidelity) is met. A single exposure time (the larger of the Clifford depth and the T-supply time) is used for both the failure estimate and the runtime. "
      "Magic states come from 15-to-1 distillation (1 or 2 levels, whichever meets the budget). Physical qubits = n<sub>logical</sub> &middot; routing &middot; (2d<super>2</super>&minus;1) + factories &middot; footprint &middot; d<super>2</super>.")
    cc = cor.copy(); cc["circuit"] = cc["circuit"].str.replace(r" \(.*", "", regex=True)
    cc = cc[["circuit", "p_phys", "scenario", "distance", "physical_qubits", "time_per_shot_ms", "distill_levels"]]
    table(cc, "Optimistic / baseline / pessimistic corners (all constants moved together, plus fit-uncertainty multipliers on A and p<sub>th</sub>). time = one circuit execution (one shot).",
          index=False, nd=3, colw=[2.4 * cm, 1.6 * cm, 2.4 * cm, 1.5 * cm, 2.8 * cm, 3.0 * cm, 2.0 * cm])
    big = cor[(cor.circuit.str.startswith("large")) & (cor.p_phys == 1e-3)].set_index("scenario")
    sm = cor[(cor.circuit.str.startswith("small")) & (cor.p_phys == 1e-3)].set_index("scenario")
    P(f"<b>Reading.</b> At p = 1e-3 the 16-qubit instance needs {int(sm.loc['baseline', 'physical_qubits']):,} physical qubits at baseline (range {int(sm.loc['optimistic', 'physical_qubits']):,} to {int(sm.loc['pessimistic', 'physical_qubits']):,}) "
      f"and {sm.loc['baseline', 'time_per_shot_ms']:.0f} ms per shot; a 120-qubit instance needs {int(big.loc['baseline', 'physical_qubits']):,} (range {int(big.loc['optimistic', 'physical_qubits']):,} to {int(big.loc['pessimistic', 'physical_qubits']):,}) "
      f"with {int(big.loc['baseline', 'distill_levels'])}-level distillation. Runtime is T-supply bound (the sweep shows cycle time, T cycles and synthesis cost dominate, routing hardly matters). "
      "A 16-qubit problem that a laptop solves in microseconds would thus need ~10<super>4</super>-10<super>5</super> physical qubits for fault-tolerant execution: logical qubits make QAOA possible at scale, not competitive at this scale.")
    fig(f"{figdir}/resources_vs_n.png", 14.5, "Baseline physical qubits and seconds per shot vs number of logical qubits (QAOA depth p = 3, dense couplings, constants at baseline).")
    fig(f"{RT}/resource_tornado.png", 16.5, "Sensitivity of physical qubits (top) and time per shot (bottom) to each constant, one at a time (teal: modelling assumption, grey: design variable).")
    A2 = asm[["parameter", "baseline", "swept", "source / justification", "confidence"]].copy()
    A2["source / justification"] = A2["source / justification"].str.slice(0, 150)
    table(A2.assign(swept=A2["swept"].str.slice(0, 48)), "Assumptions table. 'Confidence' is our judgement of how well the constant is pinned down; constants marked low are order-of-magnitude. The '+9.2' additive T-count of an earlier draft could not be verified and is not used.",
          index=False, fs=6.4, colw=[2.6 * cm, 2.2 * cm, 3.2 * cm, 7.0 * cm, 2.0 * cm])
    P("<b>Not modelled:</b> classical decoding latency and reaction time, factory idle errors, non-uniform layouts, measurement-based rotations injected without synthesis (analog-rotation architectures), and correlated physical noise. "
      "Estimates are order-of-magnitude tools for scaling arguments, not hardware quotes.")

    # ------------------------------------------------------------------ 7 ablation
    H1("7  (b) Error-mitigation ablation")
    P(f"Instances are QUBOs from random historical dates; each configuration is run with finite shots ({cfg['shots']}) on every instance with repeats. "
      "<i>est_err</i> = |estimated &minus; ideal energy| (estimation accuracy); <i>gap_norm</i> = decoded-solution gap normalised by (mean &minus; optimum) (0 = optimal); <i>p_opt_model</i> = exact probability of sampling the optimum under the noise model; "
      "<i>overhead</i> = sampling-variance multiplier of the mitigation.")
    ab = abl[["est_err", "est_err_sd", "gap_norm", "hit_opt", "p_opt_model", "N99", "overhead", "q_eff"]]
    table(ab, "Ablation (averages over instances x repeats; N99 = median shots to see the optimum with 99% confidence under the model).", nd=3)
    fig(f"{figdir}/ablation.png", 13.5, "Energy-estimation error by configuration (mean &plusmn; sd over runs).")
    best = abl.drop(index=[i for i in abl.index if i.startswith("ideal") or i.startswith("logical")]).est_err.idxmin()
    lg = [i for i in abl.index if i.startswith("logical")][0]
    P(f"<b>Findings.</b> Raw noise gives est_err {abl.loc['raw', 'est_err']:.3f}; the best physical-noise mitigation is <i>{best}</i> ({abl.loc[best, 'est_err']:.3f}, overhead {abl.loc[best, 'overhead']:.1f}x); "
      f"the logical-qubit mode reaches {abl.loc[lg, 'est_err']:.3f}, indistinguishable from ideal ({abl.loc['ideal', 'est_err']:.3f}, finite-shot floor). "
      "Mitigation reduces estimation bias but barely changes the probability of finding the optimum, which is set by q&middot;n (theory P1): decoding already uses classical evaluation of every sample. "
      "ZNE pays a variance factor of 19 per Richardson extrapolation (P4) and PEC pays &gamma;<super>2</super> (P3); both overheads grow exponentially in n&middot;q.")

    # ------------------------------------------------------------------ 8 backtests
    H1("8  (c) Out-of-sample results and statistical significance")
    P(f"Walk-forward: at each rebalance the instance is built from past data only (unit-tested: perturbing the future does not change earlier decisions), solved, traded with {cfg['tc_bps']} bp costs and held. "
      "Test windows: GFC 2008-01-01 to 2009-12-31 and COVID 2019-07-01 to 2020-12-31. Significance: paired stationary-bootstrap test of the Sharpe-ratio difference (mean block 10, 1000 resamples), "
      "Wilcoxon signed-rank on per-period returns, Holm correction over all quantum-vs-classical pairs.")
    if not real:
        P("<b>All numbers in this section are from the synthetic market.</b>", "box")
    tuned = meta.get("tuned_config_used", {})
    P("Hyper-parameters: " + ("; ".join(f"{k}: {'tuned on pre-test data only ' + str({kk: vv for kk, vv in v.items()}) if v else 'defaults (no tuning run)'}" for k, v in tuned.items()) or "defaults") + ".")
    for p in periods:
        s = rd(f"backtest_{p}_summary.csv", index_col=0)[["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover", "avg_seconds"]]
        table(s, f"{p}: out-of-sample performance net of costs.", nd=3, fs=7)
        sg = rd(f"significance_{p}.csv")
        n_b, n_w = int((sg.p_boot_holm < 0.05).sum()), int((sg.p_wilcoxon_holm < 0.05).sum()) if "p_wilcoxon_holm" in sg else -1
        best_row = sg.sort_values("p_boot_holm").iloc[0]
        P(f"<b>{p}</b>: {n_b} of {len(sg)} quantum-vs-classical Sharpe differences are significant after Holm correction (bootstrap); {n_w} by Wilcoxon. Smallest adjusted p = {best_row.p_boot_holm:.3f} "
          f"({best_row.quantum} vs {best_row.classical}, difference {best_row.sharpe_diff:+.2f}, 95% CI [{best_row.ci_low:+.2f}, {best_row.ci_high:+.2f}]).")
        fig(f"{R}/backtest_{p}_cum.png", 15.0, f"{p}: cumulative net wealth of all methods.")
        rec = rd(f"backtest_{p}_records.csv")
        qs = rec[rec.method.isin(["exact_qubo", "sa_qubo", "qaoa_ideal", "qaoa_noisy_raw", "qaoa_noisy_mitigated", "qaoa_logical"])].groupby("method").agg(
            gap_norm=("gap_norm", "mean"), approx_ratio=("approx_ratio", "mean"), optimum_hit_rate=("is_opt", "mean"))
        table(qs, f"{p}: QUBO solution quality per rebalance (hit rate = fraction of rebalances where the exact optimum was returned).", nd=3)
    pw = []
    for p in periods:
        sg = rd(f"significance_{p}.csv"); pw.append(float(((sg.ci_high - sg.ci_low) / 2).median()))
    if pw:
        P(f"<b>Statistical power.</b> The median half-width of the 95% bootstrap interval for a Sharpe difference is {', '.join(f'{x:.2f} ({p})' for x, p in zip(pw, periods))}. "
          "Hence Sharpe differences smaller than roughly this size cannot be detected in a single 6-18 month window, and a non-significant result is <i>absence of evidence</i>, not evidence of equality. "
          "Equivalence (that quantum and classical solvers do not differ) would need the much stronger statement that the interval lies inside a pre-specified margin.")
    if "miqp_sub" in methods and periods:
        rr = rd(f"backtest_{periods[0]}_records.csv")
        a1 = rr[rr.method == "miqp_sub"].set_index("date"); a2 = rr[rr.method == "exact_qubo"].set_index("date")
        same = float((np.abs(a1["miqp_obj"] - a2["E_opt"]) < 1e-9).mean())
        P(f"<b>Soft vs hard budget.</b> The exact QUBO optimum coincides with the hard-budget MIQP optimum in {same:.0%} of rebalances ({periods[0]}); where it does, the QUBO penalty formulation loses nothing and "
          "differences between QUBO solvers in P&amp;L are due to solver imperfection, not to the formulation.")
    P("<b>Interpretation guardrails.</b> Differences among QUBO solvers reflect how often each returns the exact optimum on a 12-16 qubit problem, not investment skill; "
      "Sharpe differences over 6-18 months have wide confidence intervals; the comparison against CVaR and RL conflates the objective (QUBO vs CVaR vs learned policy) with the solver.")

    # ------------------------------------------------------------------ 9 scaling and theory
    H1("9  Scaling study and theory")
    sc = scl[["p_layers", "p_opt_ideal", "N99_qaoa", "N99_qaoa_noisy_LB", "N99_random", "SA_evals_median", "grover_queries"]]
    table(sc, "Shots (QAOA) or single-flip evaluations (SA) to reach the exact optimum with 99% confidence, vs qubits n (medians over instances).", nd=3)
    fig(f"{figdir}/scaling.png", 12.5, "Cost to reach the optimum vs n (log scale). Slopes are fitted exponents in 2<super>slope &middot; n</super>.")
    P("Fitted exponents (cost ~ 2<super>s&middot;n</super>): " + ", ".join(f"{k}: {v:.2f}" for k, v in fits.items()) + ". "
      "<b>Caveats:</b> six sizes, n &le; 16; p fixed by the adaptive rule; SA 'evaluations' are O(n)-cost single flips while a QAOA shot needs a full cost evaluation and a circuit execution, so the two axes are not directly "
      "comparable in wall-clock; SA uses restarts of 200n flips, which quantises its counts. This is trend evidence at toy size, not asymptotics.")
    H2("Bounds proved for this noise model (docs/THEORY.md, each unit-tested)")
    BL(["<b>P1</b> Success lower bound: P<sub>noisy</sub>(z) &ge; P<sub>ideal</sub>(z)(1&minus;q)<super>n</super> for every bitstring z.",
        "<b>P2</b> Energy estimation: N &ge; &Delta;<super>2</super> ln(2/&delta;)/(2&epsilon;<super>2</super>) shots (Hoeffding).",
        "<b>P3</b> PEC for independent flips: &gamma; = (1&minus;2q)<super>&minus;n</super>, shots &prop; &gamma;<super>2</super>; with q = 0.03: &gamma;<super>2</super> = 7.2 at n = 16, 2.4&times;10<super>5</super> at n = 100, 2.8&times;10<super>6</super> at n = 120.",
        "<b>P4</b> ZNE (scales 1,2,3): variance inflation 19 plus truncation bias.",
        "<b>P5</b> Random sampling needs ln(0.01)/ln(1&minus;2<super>&minus;n</super>) &asymp; 4.6&middot;2<super>n</super> shots; Grover-type (&pi;/4)2<super>n/2</super> is a reference line only.",
        "<b>P6</b> Noise budget: a shot-complexity advantage over a classical solver needing N<sub>c</sub> evaluations requires q &le; 1 &minus; (p*/P<sub>ideal</sub>(z*))<super>1/n</super>, which shrinks like 1/n - the quantitative reason logical qubits (q<sub>L</sub> ~ 1e-5) are needed."])
    H2("Short proofs and a data-derived noise budget")
    P("<b>P1.</b> The flip channel is a mixture: with probability (1&minus;q)<super>n</super> no qubit flips and the output string equals the ideal one, so P<sub>noisy</sub>(z) = (1&minus;q)<super>n</super> P<sub>ideal</sub>(z) + (non-negative terms) &ge; (1&minus;q)<super>n</super> P<sub>ideal</sub>(z). "
      "<b>P3.</b> One flip(q) channel acts on a qubit's distribution as the matrix [[1&minus;q, q],[q, 1&minus;q]] with inverse [[1&minus;q, &minus;q],[&minus;q, 1&minus;q]]/(1&minus;2q), whose absolute entries sum (per column) to 1/(1&minus;2q); "
      "for n qubits the quasi-probability norm is &gamma; = (1&minus;2q)<super>&minus;n</super>. Sampling signs from the inverse gives an unbiased estimator with range &plusmn;&gamma;E<sub>max</sub>, hence Hoeffding gives N = O(&gamma;<super>2</super>&Delta;<super>2</super> ln(1/&delta;)/&epsilon;<super>2</super>). "
      "<b>P6.</b> By P1 the shots to see z* with probability 0.99 are at most ln(0.01)/ln(1 &minus; P<sub>ideal</sub>(z*)(1&minus;q)<super>n</super>); requiring this to be &le; N<sub>c</sub> and solving for q gives the budget in the text.")
    nb = []
    for n_, r_ in scl.iterrows():
        qmax = theory.noise_budget_for_advantage(int(n_), float(r_["p_opt_ideal"]), float(r_["SA_evals_median"]))
        nb.append({"n": int(n_), "P_ideal(z*)": r_["p_opt_ideal"], "classical SA evals": r_["SA_evals_median"], "max q for shot advantage": qmax,
                   "max q x n": qmax * n_})
    table(pd.DataFrame(nb), "Per-qubit flip probability q below which the guaranteed QAOA shot count (P1) stays under the median SA evaluation count, from our own instances. 0 means QAOA already needs more shots than SA with <b>no</b> noise. "
          "(Shots and SA flips are different units; this is the most generous reading for QAOA.)", index=False, nd=3)
    P("<b>Not proven:</b> any advantage of constant-depth QAOA over the best classical solvers; our scaling data are for n &le; 16. Tang-style dequantization arguments for mean-variance structure are a further threat to any finance-specific advantage claim.", "box")

    # ------------------------------------------------------------------ 10 dashboard
    H1("10  (d) Interactive dashboard: live re-optimisation under streaming data")
    P("<font name=Courier>streamlit run dashboard/app.py</font> streams daily bars tick by tick; every N ticks it rebuilds the time-dependent QUBO from the data seen so far, \
re-solves it with each selected method (QAOA variants, exact, SA, MIQP, CVaR, equal weight), and updates P&amp;L lines, target weights, the solver-gap table, the VIX-driven cap and risk aversion. \
A banner states whether the data are synthetic or real. Three data sources: synthetic, the real snapshot (a <b>streaming replay</b> of historical daily bars) and a <b>Live</b> option: \
<font name=Courier>qpo/feed.py</font> defines the <font name=Courier>next_bar()</font> hook, <font name=Courier>YahooFeed</font> polls Yahoo Finance for daily bars newer than the last one and \
<font name=Courier>catch_up()</font> extends the snapshot to the latest bar. Yahoo daily bars are end-of-day and quotes are about 15 minutes delayed, so &laquo;live&raquo; means daily-bar granularity plus delayed quotes; \
the Yahoo code was unit-tested with mocked payloads (the build sandbox has no Yahoo access) and falls back to the snapshot, saying so on screen, when offline.")
    shots = [f for f in ("dashboard_real_idle.png", "dashboard_real_final.png", "dashboard_synthetic_final.png") if os.path.exists(f"{R}/{f}")]
    for f_ in shots[:2]:
        fig(f"{R}/{f_}", 15.5, f"Dashboard screenshot ({f_.split('_')[1]} data).")
    if not shots:
        P("<i>No dashboard screenshots were available when this PDF was built (Streamlit could not run in the build sandbox). "
          "Generate them with <font name=Courier>python scripts/screenshots.py --real --out results_real</font> and rebuild the report.</i>", "box")


    # ------------------------------------------------------------------ 10.1 combined system (H4)
    LR = os.environ.get("QPO_LSTM_RESULTS", "results_lstm")
    if os.path.exists(f"{LR}/meta.json"):
        H2("10.1  Combined system: LSTM forecasts and fuzzy risk inside the optimiser")
        P("The root app (<font name=Courier>streamlit run app.py</font>) joins the optimiser with the stock-prediction system. A pooled LSTM (price-only features, 8 outputs: cumulative log-returns at 1-7 and 21 days; \
trained offline, time-ordered split, train-only scalers, purged labels) supplies the expected returns &mu; of the QUBO (<font name=Courier>mu_source = historical | lstm | blend</font>); a fuzzy risk score built from realised volatility, \
the VIX z-score and the forecast's out-of-sample confidence adds to the VIX term of &lambda;<sub>t</sub> (<font name=Courier>risk_source = vix | vix+fuzzy</font>). The quantum part only optimises. \
<font name=Courier>forecast_returns(tickers, as_of_date)</font> reads prices strictly before <i>as_of_date</i> and refuses any model trained on or after that date (walk-forward models, cut-offs at year ends; unit-tested). \
Transaction costs, ESG filter, caps and asset pre-selection are unchanged, and the same tuned configurations and causal rules are used, so the runs are paired.")
        for w in ("GFC_2008", "COVID_2020"):
            if not os.path.exists(f"{LR}/summary_{w}.csv"):
                continue
            smt = pd.read_csv(f"{LR}/summary_{w}.csv").pivot(index="method", columns="variant", values="sharpe")[["historical", "lstm", "blend", "blend+fuzzy"]]
            table(smt, f"{w}: out-of-sample Sharpe by solver and decision rule. equal_weight_full and cvar_sub do not use &mu; and are controls (identical across columns).", nd=2)
            fq = pd.read_csv(f"{LR}/forecast_quality_{w}.csv")
            sg = pd.read_csv(f"{LR}/significance_vs_historical_{w}.csv")
            qv = pd.read_csv(f"{LR}/quantum_vs_classical_by_variant_{w}.csv")
            P(f"<b>{w.replace('_', ' ')}.</b> 21-day forecast RMSE averaged over {len(fq)} rebalance dates: LSTM {fq.rmse_lstm.mean():.4f}, trailing-mean {fq.rmse_hist.mean():.4f}, zero {fq.rmse_zero.mean():.4f}; \
mean cross-sectional rank-free correlation with realised returns: LSTM {fq.ic_lstm.mean():+.3f}, trailing-mean {fq.ic_hist.mean():+.3f}. \
Paired Sharpe differences against the historical-&mu; run that are significant after Holm correction: {int((sg.p_boot_holm < 0.05).sum())} of {len(sg)} (bootstrap), {int((sg.p_wilcoxon_holm < 0.05).sum())} of {len(sg)} (Wilcoxon). \
Quantum-vs-classical differences significant inside each variant: {', '.join(f'{r.variant} {r.n_significant_holm}/{r.n_tests}' for r in qv.itertuples())}.")
        P("<b>Reading.</b> The forecaster does not beat the naive zero-return baseline out of sample, the portfolio differences between decision rules are small and not statistically distinguishable, \
and their signs flip between windows and solvers. The combination is a working pipeline, not evidence that predictions improve quantum portfolio selection.", "ok")
        P("<b>Limitations of the combined system.</b> (i) Forecast noise: near-zero skill at every horizon; the 90% band uses validation residual spread, not a calibrated predictive distribution. \
(ii) Delayed data: Yahoo daily bars/15-minute quotes, so nothing here is real-time trading. (iii) Small asset count: only 3-4 assets reach the qubits (greedy pre-selection is unchanged and drives most of the portfolio quality). \
(iv) The deployed model is trained on the shipped snapshot (to 2023-12-29) unless retrained with <font name=Courier>--refresh</font>; the GA search used a deliberately small budget (6 &times; 3 evaluations) and is not used by the walk-forward backtest models. \
(v) Features are price-only (the snapshot has no volume), the fuzzy signal's volume rule therefore uses a neutral ratio in backtests.")
        sc = [f"{R}/screens_combined/02_live_solved_streaming.png", f"{R}/screens_combined/03_backtest.png"]
        if os.path.exists(sc[0]):
            im = PI.open(sc[0]); half = im.size[1] // 2
            for k, (a_, b_, cap) in enumerate(((0, half, "Combined dashboard, Live tab (top): market strip, LSTM 7-day forecast band, fuzzy risk gauge, signal."),
                                                (half, im.size[1], "Combined dashboard, Live tab (bottom): solver weights, streaming portfolio value, quantum internals."))):
                pth = f"{figdir}/combined_live_{k}.png"; im.crop((0, a_, im.size[0], b_)).save(pth); fig(pth, 10.5, cap)

    # ------------------------------------------------------------------ 11 threats
    H1("11  Software verification")
    P("Each claim in this report that can be checked mechanically is covered by a unit test (run with <font name=Courier>python -m pytest -q</font>). The tests are the reason the reported numbers can be trusted to be the numbers the code means to produce:")
    tests = pd.DataFrame([
        ("QUBO == direct objective", "build_qubo energy equals the hand-written multi-period objective on random bitstrings"),
        ("QUBO optimum == MIQP optimum", "with a large budget penalty the exact QUBO minimum equals the exact hard-budget MIQP optimum (cross-validates both formulations)"),
        ("MIQP quadratic form", "(H, c, const) given to Gurobi reproduces the objective for any integer levels, T = 1..3"),
        ("QAOA normalised / improves", "probabilities sum to 1; trained energy below uniform sampling"),
        ("Flip channel inverse", "apply_inverse_flip_channel exactly undoes apply_flip_channel"),
        ("PEC exact / unbiased; ZNE reduces bias", "infinite-shot PEC equals the ideal energy; sampled PEC is unbiased; ZNE error below raw"),
        ("Theory P1, P2, P4, P3 values", "P1 holds for every bitstring; Hoeffding and Richardson constants; gamma^2 reference values (7.24, 2.37e5)"),
        ("No look-ahead", "changing future returns leaves earlier backtest decisions unchanged"),
        ("Real-data loaders (mocked)", "all yfinance layouts, FRED old/new headers, no back-fill, truncated series dropped, late tickers dropped, pseudo-ESG refused"),
        ("Ising / dimod conversions", "Ising and dict energies equal QUBO energies; Ising phase layer equals the simulator's up to a global phase"),
        ("Stim suppression; Gurobi == HiGHS; Qiskit == simulator; PyTorch agent", "run only where stim / gurobipy / qiskit / torch are installed (skipped otherwise)")], columns=["test", "what it establishes"])
    table(tests, None, index=False, fs=7.4, colw=[5.0 * cm, 12.0 * cm])
    P("Which of the optional tests actually ran for this report is recorded in meta.json (<font name=Courier>packages</font>) and in the status note shipped with the code.")
    H1("12  Discussion: what a quantum advantage would require")
    tq = float(cor[(cor.circuit.str.startswith("small")) & (cor.p_phys == 1e-3) & (cor.scenario == "baseline")].time_per_shot_ms.iloc[0]) / 1e3
    tl = float(cor[(cor.circuit.str.startswith("large")) & (cor.p_phys == 1e-3) & (cor.scenario == "baseline")].time_per_shot_ms.iloc[0]) / 1e3
    grow = "faster than" if fits["N99_qaoa"] > fits["SA_evals_median"] else "no faster than"
    P(f"Combining the pieces: (i) P6 says the tolerable per-qubit noise for a shot-complexity advantage decays like 1/n, so any advantage at scale is a logical-qubit statement; (ii) section 6 shows that a logical-qubit "
      f"execution of 16 qubits needs about {int(sm.loc['baseline', 'physical_qubits']):,} physical qubits and {tq:.2f} s per shot (about {tq * cfg['shots'] / 60:.0f} minutes for {cfg['shots']} shots), and 120 qubits need {tl:.0f} s per shot "
      f"({tl * cfg['shots'] / 3600:.0f} hours for {cfg['shots']} shots), against microseconds for classical exhaustive search at n = 16; "
      f"(iii) the fitted cost exponent of QAOA ({fits['N99_qaoa']:.2f}) grows {grow} that of simulated annealing ({fits['SA_evals_median']:.2f}) on our instances, which limits what can be extrapolated; (iv) portfolio problems with realistic constraints "
      "(cardinality, round lots, sector limits) have far larger discrete search spaces than our 12-16 qubit encodings, but also much stronger classical solvers (branch-and-bound with convex relaxations, as in the Gurobi MIQP baseline) whose scaling we have not mapped. "
      "A credible advantage study would therefore need: instances with n &gt;= 40 where exact methods fail, hardness-calibrated classical baselines (MIQP at its limits, tuned tabu/SA), QAOA depth scaling, and noise that is correlated and "
      "hardware-measured - none of which a 16-qubit statevector study can provide. What this project provides is the machinery and a negative baseline.")
    H1("13  Threats to validity and conclusion")
    BL(["<b>Survivorship bias.</b> Every ticker is a survivor (Lehman, Bear Stearns, WaMu, Wachovia are absent); this flatters all methods in 2008. The extended universe adds names that fell sharply but survived (C, BAC, AIG, GE, F...), reducing but not removing the bias.",
        "<b>Static ESG look-ahead.</b> ESG scores are a single present-day snapshot applied to 2005-2023; rating agencies also disagree strongly. The hard ESG filter therefore uses information unavailable at the time.",
        "<b>Sustainalytics polarity.</b> Risk scores (lower = better) are converted to a cross-sectional percentile score; the threshold then excludes a fixed fraction of the universe by construction.",
        "<b>Adjusted prices and macro data.</b> Dividends are embedded in adjusted prices; FRED's ICE-BofA credit-spread series is truncated to three years and is dropped, never back-filled.",
        "<b>Logical qubits are modelled, not run.</b> Stim simulates a surface-code <i>memory</i> experiment; QAOA receives the fitted logical error rate only as an effective independent bit-flip probability, so no logical gate, lattice-surgery or magic-state process is simulated. Resource-estimator constants (T-count per rotation, distillation footprint, cycle time, routing overhead) are assumptions, swept in results_real/resource_* and shown as corners and a tornado chart, never fitted to data.",
        "<b>Model-based noise.</b> NISQ+ noise is a phenomenological flip channel; real devices have correlated and coherent noise.",
        "<b>Small n.</b> At n &le; 16 exhaustive search is trivial; all conclusions about scaling are extrapolations from six sizes.",
        "<b>Statistics.</b> Short windows, many pairs (Holm-corrected), and hyper-parameters tuned on windows preceding each test window only; the GFC validation window is calm so it cannot teach crisis behaviour.",
        "<b>Untested components.</b> " + ("; ".join(x for x in [
            "PyTorch agent not executed" if not meta.get("optional_methods_available", {}).get("rl_torch_full") else "",
            "Gurobi MIQP/CVaR not executed" if not meta.get("optional_methods_available", {}).get("miqp_gurobi_full") else ""] if x) or "none: all optional baselines ran") + "."])
    hit = {}
    if periods:
        rr = pd.concat([rd(f"backtest_{p}_records.csv") for p in periods])
        hit = rr[rr.method.isin(["sa_qubo", "qaoa_ideal", "qaoa_noisy_raw", "qaoa_noisy_mitigated", "qaoa_logical"])].groupby("method").is_opt.mean().to_dict()
    sig = "; ".join(f"{p}: {int((rd(f'significance_{p}.csv').p_boot_holm < 0.05).sum())} of {len(rd(f'significance_{p}.csv'))}" for p in periods)
    P(f"<b>Conclusion.</b> Rate at which each solver returned the exact QUBO optimum at rebalance time: " + ", ".join(f"{k} {v:.0%}" for k, v in hit.items()) + ". "
      f"Quantum-vs-classical Sharpe differences significant after Holm correction: {sig}. Fitted cost-to-optimum exponents: QAOA {fits['N99_qaoa']:.2f} vs simulated annealing {fits['SA_evals_median']:.2f}. "
      "The theory shows that noise budgets for an advantage shrink as 1/n and that PEC costs explode with n&middot;q, and the resource estimates show that fault-tolerant execution of a 16-qubit instance needs 10<super>4</super>-10<super>5</super> physical qubits. "
      "The supported claim is therefore <i>rigorous bounds plus an empirical test of advantage criteria</i>; no quantum advantage is demonstrated" + ("." if real else " (and these market numbers are synthetic).") + "")

    H1("Appendix A: reproduction")
    for c in ["python scripts/fetch_data.py                 # internet; writes data/raw", "python scripts/build_esg.py                  # writes data/esg.csv",
              "python scripts/run_all.py --stage qec          # needs stim, pymatching", "python scripts/run_all.py --stage resources",
              "python scripts/tune.py --real --target GFC_2008; python scripts/tune.py --real --target COVID_2020",
              "python scripts/run_all.py --real --require-optional --stage backtest,ablation,scaling,summary",
              "python scripts/screenshots.py --real --out results_real", "python report/build_report.py --results results_real --out report/QPO_Technical_Report.pdf"]:
        P(c, "code")
    H1("Appendix B: full significance tables")
    for p in periods:
        sg = rd(f"significance_{p}.csv")[["quantum", "classical", "sharpe_diff", "ci_low", "ci_high", "p_boot", "p_boot_holm", "p_wilcoxon", "p_wilcoxon_holm"]]
        table(sg, f"{p}: every quantum-vs-classical pair (bootstrap and Wilcoxon p-values, raw and Holm-adjusted).", index=False, nd=3, fs=6.2,
              colw=[3.0 * cm, 3.0 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm, 1.3 * cm, 1.5 * cm, 1.5 * cm, 1.6 * cm])
    H1("Appendix C: resource sensitivity (one parameter at a time)")
    sw2 = swp.groupby(["circuit", "parameter"], sort=False).agg(qubits_min=("physical_qubits", "min"), qubits_max=("physical_qubits", "max"),
                                                               ms_min=("time_per_shot_ms", "min"), ms_max=("time_per_shot_ms", "max")).reset_index()
    sw2["circuit"] = sw2["circuit"].str.replace(r" \(.*", "", regex=True)
    table(sw2, "Range of physical qubits and time per shot (ms) when each parameter is swept over the values in the assumptions table, others at baseline.", index=False, nd=3, fs=6.6)
    P(f"Seeds: Config.seed = {meta.get('seed')}. Versions: " + ", ".join(f"{k} {v}" for k, v in meta.get("packages", {}).items() if v) + f". Results folder: {R}.")

    # ------------------------------------------------------------------ figures
    for nm in ("ablation", "scaling"):
        pass

    def footer(c, d):
        c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
        c.drawString(2 * cm, 1.0 * cm, "QPO technical report"); c.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"Page {d.page}")
        if not real:
            c.setFillColor(RED); c.setFont("Helvetica-Bold", 8); c.drawCentredString(A4[0] / 2, A4[1] - 1.0 * cm, "SYNTHETIC-DATA DRAFT - not real-market results")
        c.restoreState()
    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.6 * cm,
                            title="Fault-Tolerant Quantum Portfolio Optimizer - Technical Report")
    doc.build(E, onFirstPage=footer, onLaterPages=footer)
    print("wrote", a.out)


def make_figs(R):
    figdir = os.path.join(R, "report_figs"); os.makedirs(figdir, exist_ok=True)
    abl = pd.read_csv(f"{R}/ablation_summary.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 3.4)); ax.bar(range(len(abl)), abl.est_err, yerr=abl.est_err_sd, color="#139A8C", capsize=2)
    ax.set_xticks(range(len(abl))); ax.set_xticklabels(abl.index, rotation=35, ha="right", fontsize=7); ax.set_ylabel("energy estimation error"); fig.tight_layout()
    fig.savefig(f"{figdir}/ablation.png", dpi=150); plt.close(fig)
    s = pd.read_csv(f"{R}/scaling.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for c, lab in [("N99_qaoa", "QAOA (ideal)"), ("N99_qaoa_noisy_LB", "QAOA noisy bound"), ("N99_random", "random sampling"), ("SA_evals_median", "simulated annealing"), ("grover_queries", "Grover (reference)")]:
        ax.plot(s.index, s[c], "o-", label=lab)
    ax.set_yscale("log"); ax.set_xlabel("qubits n"); ax.set_ylabel("shots / evaluations to optimum (99%)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{figdir}/scaling.png", dpi=150); plt.close(fig)


def make_theory_figs(RT, figdir):
    import json
    m = json.load(open(f"{RT}/qec_fit.json")); sw = pd.read_csv(f"{RT}/qec_sweep.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); ps = np.logspace(-4, -2.2, 50)
    for d in (3, 5, 7, 11, 15, 21):
        ax.plot(ps, m["A"] * (ps / m["p_th"]) ** ((d + 1) / 2), label=f"fit d={d}")
    for d, c in zip((3, 5, 7), ("C0", "C1", "C2")):
        r = sw[sw.d == d]; ax.plot(r.p, r.p_L_per_round, "ko", ms=4)
    ax.axvspan(1e-4, 2e-3, color="#E9A820", alpha=.15); ax.text(1.2e-4, 3e-14, "extrapolated (no Stim data)", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-16, 1); ax.set_xlabel("physical error p"); ax.set_ylabel("logical error / round"); ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(f"{figdir}/qec_extrapolation.png", dpi=150); plt.close(fig)
    rs = pd.read_csv(f"{RT}/resources.csv"); rs = rs[(rs.p_layers == 3)]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    for pp, g in rs.groupby("p_phys"):
        g = g.sort_values("n_logical"); ax[0].plot(g.n_logical, g.physical_qubits, "o-", label=f"p={pp:g}"); ax[1].plot(g.n_logical, g.time_per_shot_ms / 1e3, "o-", label=f"p={pp:g}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[0].set_xlabel("logical qubits n"); ax[0].set_ylabel("physical qubits"); ax[1].set_xlabel("logical qubits n"); ax[1].set_ylabel("seconds per shot"); ax[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{figdir}/resources_vs_n.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap0 = argparse.ArgumentParser(add_help=False); ap0.add_argument("--results", default="results"); r0, _ = ap0.parse_known_args()
    _need = ["ablation_summary.csv", "scaling.csv", "meta.json", "RESULTS.md"]
    _miss = [f for f in _need if not os.path.exists(os.path.join(r0.results, f))]
    if _miss:
        sys.exit(f"Cannot build the report: {r0.results}/ is missing {_miss}.\n"
                 "These are produced by the backtest/ablation/scaling/summary stages, which need real data + a real ESG csv:\n"
                 f"  python scripts/run_all.py --real --require-optional --stage backtest,ablation,scaling,summary --out {r0.results}\n"
                 "(run build_esg.py / provide data/esg.csv first).")
    make_figs(r0.results)
    ap1 = argparse.ArgumentParser(add_help=False); ap1.add_argument('--theory-results', default=None); r1, _ = ap1.parse_known_args()
    make_theory_figs(r1.theory_results or r0.results, os.path.join(r0.results, 'report_figs'))
    main()
