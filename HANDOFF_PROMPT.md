# Paste this into a NEW chat, and attach QPO_Handoff.zip

I am building "Fault-Tolerant Quantum Portfolio Optimizer with Dynamic Constraints" (problem statement in `reference/PROBLEM_STATEMENT.md`; concepts in `reference/Quantum_Portfolio_Optimizer_Learning_Guide.pdf`). I have about 4 days left. The attached zip holds the work so far. Unzip it, read `qpo_project/README.md` and `qpo_project/docs/*`, run `python -m pytest -q` (12 tests should pass), then continue from the state below. Do the work in order, run tests after each change, and tell me honestly about anything that fails.

## Current state (about 70% complete)
Built and tested: multi-period time-dependent QUBO (`qpo/qubo.py`); adaptive-depth statevector QAOA, ZNE and PEC (`qpo/qaoa.py`); NISQ+ and logical noise model with dynamical decoupling (`qpo/noise.py`); Stim + PyMatching surface-code sweeps, logical-error fit and resource estimator (`qpo/qec.py`); classical baselines: exact, simulated annealing, CVaR LP (HiGHS and Gurobi), equal weight, evolution-strategy agent (`qpo/classical.py`); walk-forward backtest (`qpo/backtest.py`); bootstrap, Wilcoxon and Holm tests (`qpo/stats.py`); theory P1-P6 (`qpo/theory.py`, `docs/THEORY.md`); Streamlit dashboard (`dashboard/app.py`, `qpo/live.py`); scripts `run_all.py`, `ablation.py`, `scaling.py`. A walkthrough PPT/PDF and the figure scripts are in `walkthrough/`.

All numbers in `results/` come from SYNTHETIC data. Findings so far: QAOA does not beat exact search or simulated annealing (cost about 2^(0.86n) vs 2^(0.43n) for SA); 0 of 28 quantum-vs-classical Sharpe differences are significant after Holm correction.

## Remaining work (priority order)
1. **Real data.** Run `qpo/data.py::load_real_market` (yfinance + FRED CSV endpoint). Fix format and column problems. Build `data/esg.csv` (ticker,esg) from an MSCI or Sustainalytics open subset for the tickers in `scripts/run_all.py`. Use tickers that existed in 2004-2008. Document survivorship bias and static-ESG look-ahead. If the network is blocked, tell me and give me exact commands to run on my laptop.
2. **Re-run on real data:** `python scripts/run_all.py --real --esg-csv data/esg.csv` (and `--fast` first). Include the 2008 and 2020 windows. Tune `Config` on non-test periods only.
3. **PyTorch RL baseline.** Test and fix `qpo/rl_torch.py`, wire it into the backtest as a method (`rl_torch_full`), and compare it with the existing NumPy evolution-strategy agent.
4. **Gurobi MIQP baseline** on the discrete portfolio problem, alongside the CVaR stochastic program.
5. **Logical-qubit rigor.** Sweep the resource-estimator constants (T per rotation, factory footprint, routing overhead) and put the assumptions in a table. State clearly that Stim simulates surface-code memory only and logical noise is injected into the small QAOA simulation.
6. **Fix a doc error:** `docs/THEORY.md` P3 says gamma^2 is about 3.6e4 at n=100, q=0.03; the formula gives about 2.4e5.
7. **Optional:** Ocean SDK simulated annealing, and a Qiskit circuit-level QAOA demo.
8. **Dashboard:** take screenshots on real data (Playwright can drive `streamlit run dashboard/app.py`).
9. **Technical report, 15-20 pages**, following `docs/REPORT_SKELETON.md`: (a) logical-qubit resource estimation, (b) mitigation ablation, (c) significance tests, (d) dashboard. Frame the claim honestly as "rigorous bounds plus an empirical test of advantage criteria", not "proven advantage". Use only numbers from the real-data run.
10. Regenerate `walkthrough/` (PPT and PDF) with the real-data numbers using `walkthrough/figs.py`, `content.py`, `mkpdf.py`, `mkppt.js`.

## Rules
- Never present synthetic-data results as real-market results.
- Report negative results honestly.
- Keep the pipeline reproducible: one command per stage, seeds fixed.
- Give me a short status note after each numbered item.
