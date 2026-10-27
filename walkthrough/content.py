"""Builds steps.json for mkpdf.py / mkppt.js.  Every number comes from the results folder (QPO_RESULTS, default ../qpo_project/results)
and figs/facts.json (written by figs.py).  Data-source label is taken from meta.json: nothing says 'real' unless the run was real."""
import json, os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("QPO_RESULTS", os.path.join(HERE, "..", "qpo_project", "results"))
if not os.path.isabs(RES) and not os.path.exists(RES):
    RES = os.path.join(HERE, "..", "qpo_project", RES)
F = "figs/"
meta = json.load(open(f"{RES}/meta.json")); facts = json.load(open(os.path.join(HERE, "figs", "facts.json")))
real = meta.get("data_source") == "yahoo+fred+esg"
TAG = "REAL DATA" if real else "SYNTHETIC DATA"
rd = lambda f, **k: pd.read_csv(f"{RES}/{f}", **k)
abl = rd("ablation_summary.csv", index_col=0); scl = rd("scaling.csv", index_col=0); fits = rd("scaling_fits.csv", index_col=0).iloc[:, 0]
cor = rd("resource_corners.csv"); fit = json.load(open(f"{RES}/qec_fit.json"))
bt = {p: rd(f"backtest_{p}_summary.csv", index_col=0) for p in ("GFC_2008", "COVID_2020")}
sg = {p: rd(f"significance_{p}.csv") for p in bt}
cls = lambda p: bt[p].drop(index=[i for i in bt[p].index if i.startswith("qaoa")]).sharpe
sm = cor[(cor.circuit.str.startswith("small")) & (cor.p_phys == 1e-3)].set_index("scenario")
lg = cor[(cor.circuit.str.startswith("large")) & (cor.p_phys == 1e-3)].set_index("scenario")
lgq = [i for i in abl.index if i.startswith("logical")][0]
def bsum(p):
    b = bt[p].sharpe; c = cls(p)
    return f"{p}: best classical {c.idxmax()} Sharpe {c.max():.2f}; QAOA logical {b['qaoa_logical']:.2f}; equal weight {b['equal_weight_full']:.2f}."
nsig = {p: int((sg[p].p_boot_holm < 0.05).sum()) for p in sg}
ex = sg["GFC_2008"][(sg["GFC_2008"].quantum == "qaoa_ideal") & (sg["GFC_2008"].classical == "cvar_sub")].iloc[0]
tor = "torch" if meta.get("optional_methods_available", {}).get("rl_torch_full") else None
gur = meta.get("optional_methods_available", {}).get("miqp_gurobi_full")

steps = [
("The big picture", "How the whole code runs, end to end",
 ["Data comes in: prices, VIX, ESG scores.", "Every month we estimate returns (mu) and risk (Sigma).", "We turn 'choose the best weights' into a QUBO: a yes/no puzzle over bits.",
  "QAOA (a quantum circuit) tries to find the lowest-energy bit pattern.", "Noise is added, then error mitigation or logical qubits are applied.",
  "Classical solvers (exact, SA, MIQP, CVaR, RL) solve the same problem; a walk-forward backtest compares all of them."],
 "01_architecture.png", "run_all.py calls each box in this order, one command per stage.", f"Each box is one file in qpo/. Numbers in this deck: {TAG}."),
("Step 1 - Data", "qpo/data.py, scripts/fetch_data.py, scripts/build_esg.py",
 ["Real mode: fetch_data.py freezes Yahoo prices and FRED series to csv; every later stage runs offline from that snapshot.",
  "ESG: build_esg.py converts Sustainalytics risk scores (lower = better) into a percentile score where higher = better.",
  "Safeguards: macro series are never back-filled, truncated FRED series are dropped, pseudo-ESG is refused.",
  "Offline mode: a synthetic market with calm and stress regimes and forced 2008-like and 2020-like crises (figure).",
  f"This deck was built from: {TAG} (meta.json data_source = {meta.get('data_source')})."],
 "02_market.png", "Figure is the synthetic market used for tests and demos.", "Real runs must report survivorship bias and static-ESG look-ahead."),
("Step 2 - Estimate return and risk", "qpo/backtest.py make_instance()",
 ["Use the last 252 trading days only (no look-ahead).", "Expected return mu = half the asset's own mean + half the universe mean (shrinkage).",
  "Covariance Sigma is shrunk 20% toward its diagonal.", "ESG hard filter: assets with ESG < 40 are removed first.", "Greedy pick of 3-4 assets (best Sharpe, low correlation) to fit on qubits.",
  "VIX z-score sets risk aversion lambda_t and the weight cap (50% calm, 33% stress)."],
 None, "lambda_t = 6 x (1 + 0.5 x VIX_z).  Stress: cap drops from 3/6 to 2/6.", "Time-varying constraints come from this one step."),
("Step 3 - Build the QUBO", "qpo/qubo.py build_qubo()",
 ["A QUBO is: minimise x^T Q x with every x a bit.", "Each weight = level/6, level = b0 + 2*b1: two bits per asset per period.",
  f"3 assets x 2 bits x 2 periods = {facts['n']} qubits (4096 candidate answers).", "Q mixes: -return, +risk, +transaction cost, +budget penalty, +cap penalty, -ESG tilt.",
  "Tests: Q matches the direct objective, and its optimum equals the hard-budget MIQP optimum."],
 "03_qubo.png", f"Heatmap of Q for the {facts['date']} instance (synthetic market).", "Red/blue cells are the couplings the circuit must encode."),
("Step 4 - QAOA (the quantum part)", "qpo/qaoa.py",
 ["Start in an equal superposition of all bitstrings.", "Repeat p times: a cost phase then an RX mixer.", "Two angles per layer are tuned by COBYLA.",
  "Adaptive depth: add layers (up to 3) only while the improvement exceeds 1%.", "Sample bitstrings, evaluate their true energy classically, keep the best."],
 "04_qaoa.png", f"Optimum energy {facts['Emin']:.3f}, random mean {facts['Emean']:.3f}, QAOA (p={facts['p']}) mean {facts['E_qaoa']:.3f}.", f"P(optimum) = {facts['popt']:.4f} vs uniform {1/2**facts['n']:.5f}: boosted, but the optimum stays unlikely."),
("Step 5 - Noise (NISQ+ model)", "qpo/noise.py",
 ["Gate, idle and readout errors are turned into an independent bit-flip probability q per qubit on the output string.",
  f"On this instance q is {facts['q_eff'][0]:.3f} (raw), {facts['q_eff'][1]:.3f} (with DD) and {facts['q_eff'][2]:.1e} (logical).", "Phenomenological model, not hardware truth."],
 "05_noise.png", f"Energy {facts['noise_vals'][0]:.3f} (ideal) -> {facts['noise_vals'][1]:.3f} (raw).", "Dynamical decoupling only trims idle error, so it helps a little."),
("Step 6a - Zero-noise extrapolation (ZNE)", "qpo/qaoa.py estimate_energy(method='zne')",
 ["Run at 1x, 2x and 3x noise.", "Fit a curve and read it at 0x.", "Cost: more runs and variance inflated by 19x (Richardson weights 3, -3, 1)."],
 "06_zne.png", f"Raw {facts['raw']:.4f} -> ZNE {facts['zne']:.4f} (ideal {facts['E_qaoa']:.4f}).", "ZNE reduces the bias but not completely."),
("Step 6b - Probabilistic error cancellation (PEC)", "qpo/qaoa.py + qpo/theory.py",
 ["The channel is known, so we apply its inverse with signed random samples.", f"On this instance PEC gives {facts['pec']:.4f} (ideal {facts['E_qaoa']:.4f}): unbiased.",
  "Price: gamma = (1-2q)^(-n); shots grow as gamma^2."],
 "07_pec.png", f"n=12 at this q: gamma^2 = {facts['pec_gamma2_n12']:.1f}.  n=100, q=0.03: gamma^2 = {facts['pec_gamma2_n100_q03']:.2e}.", "This is why logical qubits are needed at scale."),
("Step 7 - Logical qubits with Stim", "qpo/qec.py",
 ["Stim simulates the rotated surface code memory experiment, PyMatching decodes (d = 3, 5, 7).",
  f"Fit: p_L/round = A (p/p_th)^((d+1)/2) with A = {fit['A']:.3f}, p_th = {fit['p_th']:.4f} (9 points).", "Logical mode injects the fitted logical flip rate into the small QAOA simulation.",
  "Stim covers memory only; QAOA's non-Clifford gates are not simulated at code level."],
 "12_qec.png", "Data exist for p = 2e-3..6e-3 only; lower p is an extrapolation.", "Say this clearly in the report."),
("Step 8 - Resource estimation", "qpo/qec.py estimate_resources(), scripts/resource_sweep.py",
 ["Count rotations -> T gates, choose distance d, add T-factory area; failure budget split three ways (logical, synthesis, magic states).",
  f"n=16, p=1e-3: {int(sm.loc['baseline','physical_qubits']):,} physical qubits (range {int(sm.loc['optimistic','physical_qubits']):,} - {int(sm.loc['pessimistic','physical_qubits']):,}).",
  f"n=120, p=1e-3: {int(lg.loc['baseline','physical_qubits']):,} (range {int(lg.loc['optimistic','physical_qubits']):,} - {int(lg.loc['pessimistic','physical_qubits']):,}); needs {int(lg.loc['baseline','distill_levels'])}-level distillation.",
  "All constants are swept (assumptions table, tornado plot). Runtime is T-supply bound."],
 "08_resources.png", "Report item (a).", "Better hardware (p=1e-4) cuts the qubit count several-fold."),
("Step 9 - Error-mitigation ablation", "scripts/ablation.py",
 [f"Each mitigation is switched on separately and in combination ({int(len(abl))} configurations).", f"Raw error {abl.loc['raw','est_err']:.3f}; ZNE {abl.loc['ZNE','est_err']:.3f}; PEC {abl.loc['PEC','est_err']:.3f}.",
  f"DD+ZNE {abl.loc['DD+ZNE','est_err']:.3f}; DD+PEC {abl.loc['DD+PEC','est_err']:.3f}; {lgq} {abl.loc[lgq,'est_err']:.3f}.",
  f"Sampling overhead: ZNE {abl.loc['ZNE','overhead']:.0f}x, PEC {abl.loc['PEC','overhead']:.0f}x."],
 "09_ablation.png", "Report item (b).", "Mitigation helps estimation; logical qubits help most."),
("Step 10 - Classical baselines", "qpo/classical.py, qpo/miqp.py, qpo/rl_torch.py",
 ["exact_qubo: brute force = true QUBO optimum (ceiling for QUBO solvers). sa_qubo: simulated annealing.",
  "miqp_sub: hard-budget discrete problem, Gurobi MIQP if installed, else an exact DP (same optimum, cross-tested)" + ("; miqp_gurobi_full on all eligible assets." if gur else "."),
  "CVaR stochastic programs in HiGHS (and Gurobi when installed); equal weight.",
  "RL: NumPy evolution-strategy agent" + (" and PyTorch REINFORCE agent (ran in this build)." if tor else "; PyTorch REINFORCE agent written but NOT run in this build.")],
 None, "CVaR minimises average loss in the worst 10% of scenarios.", "The exact optimum lets us measure how far QAOA is from perfect."),
("Step 11 - Walk-forward backtest", "qpo/backtest.py run_backtest()",
 ["Every rebalance: build the QUBO from past data only, solve, trade, pay costs, hold.", f"Methods: {len(meta.get('methods', []))}.", bsum("GFC_2008"), bsum("COVID_2020"),
  "QUBO solvers differ little: the problem is tiny and the exact optimum is the ceiling."],
 "11_backtest.png", f"{TAG}." + ("" if real else " These numbers say nothing about real markets."), "Tests confirm there is no look-ahead."),
("Step 12 - Statistical significance", "qpo/stats.py",
 ["Paired stationary bootstrap on the Sharpe difference; Wilcoxon on per-period returns; Holm correction.",
  f"Result: {nsig['GFC_2008']} of {len(sg['GFC_2008'])} (GFC) and {nsig['COVID_2020']} of {len(sg['COVID_2020'])} (COVID) differences are significant.",
  f"Example: qaoa_ideal vs cvar_sub in 2008: diff {ex.sharpe_diff:+.2f}, 95% CI [{ex.ci_low:+.2f}, {ex.ci_high:+.2f}].", "Wide intervals mean low power: not significant is not the same as equal."],
 None, "Report item (c).", "No claim of quantum outperformance is supported by this table."),
("Step 13 - Scaling and theory bounds", "qpo/theory.py, docs/THEORY.md, scripts/scaling.py",
 ["P1 noisy success >= ideal x (1-q)^n.  P2 Hoeffding shots.  P3 PEC cost gamma^2 (n=100, q=0.03: 2.4e5).  P4 ZNE variance x19.  P5 random baseline 2^n.  P6 noise budget shrinks like 1/n.",
  f"Empirical exponents (cost ~ 2^(s n)): QAOA {fits['N99_qaoa']:.2f}, SA {fits['SA_evals_median']:.2f}, random {fits['N99_random']:.2f}, Grover {fits['grover_queries']:.2f}.",
  "Six sizes, n <= 16: trend evidence only."],
 "10_scaling.png", "Honest result: the bounds are rigorous; an advantage is not shown.", "Grover's sqrt(2^n) curve is a reference line only."),
("Step 14 - Live dashboard", "dashboard/app.py, qpo/live.py, scripts/screenshots.py",
 ["Streamlit app replays market data tick by tick (synthetic or the real snapshot, with a banner saying which).", "Every N ticks it rebuilds the QUBO from data seen so far and re-solves with each chosen method.",
  "Shows P&L lines, weight bars and a table of solver gaps.", "Screenshots below are from an EARLIER build on synthetic data; regenerate with scripts/screenshots.py."],
 "13_dash_live.png", "Screenshot: 2008 replay, synthetic data, earlier build.", "Report item (d)."),
]
json.dump(dict(data_tag=TAG, status=[
    "Built and tested offline: QUBO, QAOA, noise, DD/ZNE/PEC, resource estimator + sweeps, MIQP (exact DP), stats, theory, data-loader safeguards (mocked)",
    "Needs your machine: real Yahoo/FRED/ESG data, Stim re-run, PyTorch agent test, Gurobi MIQP test, dashboard screenshots",
    "Honest claim: rigorous bounds plus an empirical test; no quantum advantage shown at n up to 16"], steps=[dict(title=a, sub=b, bullets=c, img=(F + d if d else None), cap=e, take=f) for a, b, c, d, e, f in steps]),
    open(os.path.join(HERE, "steps.json"), "w"), indent=1)
print(len(steps), TAG)
