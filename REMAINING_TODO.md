# QPO - Remaining work (updated)

Status: about 90-95% of the problem-statement deliverables are done (real-data run, 26 Oct 2026).
Part 1 finishes the submission. Part 2 builds the combined live dashboard (quantum optimizer + stock prediction system).
Tick the boxes as you go.

---

# PART 1 - Finish the project

## A. Real gaps (do these first)

- [ ] **A1. Live data feed for the dashboard.**
  The dashboard currently *replays* historical daily bars (`qpo/live.py`, `dashboard/app.py`). The statement says "streaming market feeds".
  - Best: add a small poller (e.g. yfinance latest bars) behind the `next_bar` hook, with a "Live" option in the data-source switch.
  - Minimum: rename it "streaming replay" everywhere (dashboard title, README, report).
- [ ] **A2. State clearly that logical qubits are modelled.**
  Stim simulates surface-code memory; QAOA receives the result as an effective logical bit-flip rate. Say so in the report (method + limitations) and the README. Say that resource-estimator constants are assumptions, swept in `results_real/resource_*`.

## B. Wording and honesty in the report

- [ ] **B1. Present the null result as the headline.**
  - 0 of 40 significant quantum-vs-classical Sharpe differences per window (bootstrap and Wilcoxon, Holm-corrected).
  - No scaling advantage: cost ~ 2^(0.85 n) for QAOA vs 2^(0.42 n) for SA; near-equal at n~10, SA ahead from n~12.
  - Frame the report as a test of advantage criteria and noise thresholds, not a claimed advantage.
  - Note the exact QUBO optimum scored worse than noisy QAOA in GFC (Sharpe -0.28 vs -0.16): solver differences reflect discretisation and noise, not alpha.
- [ ] **B2. Name the ESG source correctly.** The file is `sp_esg.csv` (S&P-sourced); the statement lists MSCI or Sustainalytics open subsets. State the source, the percentile conversion, and that it is a static snapshot.
- [ ] **B3. Threats to validity section:**
  - survivors-only universe (survivorship bias flatters 2008)
  - static ESG snapshot (look-ahead)
  - only 4 of 30 assets encoded on qubits (greedy selection drives much of portfolio quality)
  - n = 16 qubits: exhaustive search solves each instance in < 1 ms
  - logical noise is a phenomenological bit-flip model
  - JNJ has no ESG score and receives the median
  - HY_OAS dropped (FRED now gives < 3 years)
- [ ] **B4. Scaling caveat.** The exponent fit uses n = 6..16 only; add confidence intervals or call it "indicative". An SA evaluation is a cheap function call; a QAOA shot is a full circuit run.
- [ ] **B5. Cite closest literature** (read each before quoting): Stopfer and Wagner benchmark (arXiv:2509.17876), Yuan et al. (arXiv:2410.16265), Stilck Franca and Garcia-Patron (Nat. Phys. 2021).

## C. Optional polish

- [ ] **C1. VQE.** Statement says "QAOA or VQE", so QAOA suffices. Add VQE only if time allows.
- [ ] **C2. Qiskit and Ocean are only in optional demos.** In the report, say the main pipeline uses its own exact statevector simulator and Qiskit confirms it (max |dP| = 6.07e-18; 168 cx matches the model). Note the rz mismatch (204 transpiled vs 132 modelled).
- [ ] **C3. Ocean SA comparison has only 10 instances** (hit rate 0.3 vs 0.5). Use 50+ instances or label it a sanity check.
- [ ] **C4. Tuning sensitivity.** GFC tuning picked kappa_vix = 0.0 and the grid is nearly flat at the top (1.914-1.929). Add a robustness table across the top configurations.
- [ ] **C5. Optional scale-up.** Run the scaling study to larger n (structured or sparse problems) with confidence intervals.
- [ ] **C6. Merge the references.** Run `merge_papers.py` on your laptop to build one PDF with your results plus the downloaded papers.
- [ ] **C7. Refresh the walkthrough** (`make walkthrough`) and regenerate the report (`python report/build_report.py --results results_real`) after edits.

## D. Final checks before submission

- [ ] `python -m pytest -q` passes.
- [ ] Re-run `scripts/run_all.py --real --require-optional ...` if any code changed; `meta.json` must say real data.
- [ ] No "SYNTHETIC" banner on the final report.
- [ ] Report is 15-20 pages: resource estimation, ablation, significance tests, dashboard.
- [ ] README commands work from a clean venv.

---

# PART 2 - Combined live dashboard (quantum optimizer + Stock Prediction System)

Idea: LSTM forecasts supply the expected returns (mu) and a risk score supplies the risk aversion (lambda_t) for the time-dependent QUBO; the quantum and classical solvers then choose the weights. The quantum part does not "detect" anything, it optimizes.

## E. Understand and repair the Stock Prediction System

- [ ] **E1. Check the fuzzy logic and genetic algorithm.** The README describes them, but `app.py` only contains the LSTM and `ta` indicators. Either add them or remove them from the README.
- [ ] **E2. Stop training on every request.** `train_model` currently runs on each POST. Train offline, save the model (`.keras`) and scaler, and load them with a cache.
- [ ] **E3. Turn `app.py` into a module.** Separate functions (fetch, indicators, sequences, forecast, confidence, risk) from the Flask routes so Streamlit can import them.
- [ ] **E4. Fix evaluation honesty.** Use a time-ordered train/validation/test split, no shuffling and no leakage from the scaler; report RMSE against a naive "tomorrow = today" baseline.
- [ ] **E5. Pin versions** for TensorFlow, numpy and pandas so they coexist with torch and Gurobi in one environment (or run the predictor in a separate process).

## F. Integrate prediction into the optimizer

- [ ] **F1. Interface.** A function `forecast_returns(tickers, as_of_date) -> mu_hat` using only data before `as_of_date`.
- [ ] **F2. Plug into the QUBO.** Add a config switch `mu_source = "historical" | "lstm" | "blend"` in `qpo/backtest.make_instance`.
- [ ] **F3. Risk score into lambda_t.** Combine the VIX z-score with the prediction system's risk rating (or a fuzzy risk output) to set lambda_t.
- [ ] **F4. Backtest the combination** on GFC 2008 and COVID 2020 with the same causal rules and significance tests. Report honestly if LSTM mu does not beat historical mu.
- [ ] **F5. Reuse costs.** Keep transaction costs, ESG filter and caps unchanged so results are comparable.

## G. Build the single dashboard (Streamlit)

- [ ] **G1. Skeleton.** One Streamlit app, tabs: Live | Backtest (2008, 2020) | Ablation | Resources | About.
- [ ] **G2. Market strip.** Live ticker prices, daily change and VIX (yfinance polling, refresh every N seconds, cache with TTL).
- [ ] **G3. Prediction panel.** Price chart with the LSTM 7-day forecast band, confidence score, risk gauge, trading signal.
- [ ] **G4. Optimizer panel.** Weights from QAOA, exact QUBO, Gurobi MIQP, CVaR and equal weight side by side; turnover, expected Sharpe, solve time.
- [ ] **G5. Constraint controls.** Sidebar sliders for ESG minimum, maximum weight, transaction cost, risk aversion, noise level, mitigation method; the QUBO rebuilds on change.
- [ ] **G6. Streaming portfolio value.** Live P&L of each solver vs equal weight.
- [ ] **G7. Quantum internals panel.** Qubits, circuit depth, effective noise, logical error rate (Stim fit), resource estimate.
- [ ] **G8. Results tabs.** Load `results_real/` tables and figures into the Backtest, Ablation and Resources tabs.
- [ ] **G9. Performance.** Cache the market data and the LSTM; limit QAOA to 3-4 assets and 512-1024 shots in live mode; run the solve in a background thread so the UI never freezes.
- [ ] **G10. Safety labels.** Show "research demo, not financial advice", data source, data delay and whether results are real or synthetic.

## H. Testing and delivery of the combined app

- [ ] **H1. Tests.** Unit tests for the `forecast_returns` interface (no look-ahead) and for the dashboard data loaders (use mocked yfinance payloads).
- [ ] **H2. One-command run.** `streamlit run app.py` from a clean venv; update `requirements.txt` and the README quick start.
- [ ] **H3. Screenshots** of every tab for the report.
- [ ] **H4. Update the report** with a short section on the combined system and its limitations (forecast noise, delayed data, small asset count).