# Fault-Tolerant Quantum Portfolio Optimizer with Dynamic Constraints

Hybrid quantum-classical, walk-forward portfolio optimizer. Multi-period, time-dependent QUBO (time-varying risk aversion and
weight caps driven by VIX, transaction-cost surrogate, hard ESG exclusion + soft ESG tilt) solved by a layer-wise adaptive QAOA
on an exact statevector simulator with (i) a NISQ+ noise model + DD / ZNE / PEC, and (ii) a *logical-qubit* mode whose noise comes from
**Stim + PyMatching surface-code simulations**. Compared with exact search, simulated annealing, CVaR stochastic programs (HiGHS **and Gurobi**),
equal weight and an evolution-strategy "deep RL" agent, on out-of-sample windows containing 2008 and 2020 crises.

## Quick start
```bash
# Windows PowerShell: python -m venv .venv ; .venv\Scripts\Activate.ps1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                       # ~60 tests, ~1-2 min (no PYTHONPATH needed)
python scripts/run_all.py --fast --out results_smoke   # ~1 min, 3 assets / 12 qubits, synthetic data
python scripts/run_all.py                # ~6 min, 4 assets / 16 qubits, synthetic data (writes results/)
streamlit run app.py                     # COMBINED live dashboard: quantum optimiser + stock prediction (LSTM, fuzzy, GA)
streamlit run dashboard/app.py           # optimiser-only dashboard (synthetic / real replay / Live Yahoo feed)
```
Real data - **one command per stage, see `STATUS.md` and the `Makefile`**:
`python scripts/fetch_data.py` -> `python scripts/build_esg.py` -> `python scripts/tune.py --real --target ...` -> `python scripts/run_all.py --real --require-optional` -> `python report/build_report.py --results results_real`.
Universes live in `qpo/universe.py`; real runs write to `results_real/` so they can never be mixed up with the synthetic `results/`.

## Combined system: optimiser + stock prediction (Part 2)
```
prices -> LSTM (stock_prediction/) -> mu_hat (21-day expected returns) ----+
VIX z-score + fuzzy risk score (volatility, VIX stress, forecast confidence) -> lambda_t --+-> time-dependent QUBO -> QAOA / exact / SA / MIQP / CVaR -> weights
```
The quantum part only *optimises*; it does not predict or detect anything.
* `stock_prediction/` is the old Flask prototype turned into a module: `features` (price-only, scale-free), `model` (pooled LSTM, time-ordered train/val/test split with purging, train-only scalers, RMSE vs the naive "tomorrow = today" baseline), `fuzzy` (risk + signal rules from the old README), `ga` (real-valued GA: tournament / arithmetic crossover / Gaussian mutation), `forecast` (`forecast_returns(tickers, as_of_date, prices, model)` - raises `LookAheadError` if the model saw `as_of_date`), `service`, `flask_app` (routes only; **never trains on a request**).
* Train offline: `python scripts/train_predictor.py` (deployed model, 70/15/15 time-ordered split), `--walk-forward 2007-12-31 2008-12-31 2018-12-31 2019-12-31` (models for the backtests), `python scripts/ga_tune.py` (GA hyper-parameters, small default budget). Models land in `models/` (`.keras` + `.json`).
* Optimiser switches in `Config`: `mu_source = "historical" | "lstm" | "blend"`, `risk_source = "vix" | "vix+fuzzy"`. Defaults reproduce the original decision rule exactly (a test and a one-off regression check pin this); asset selection, costs, ESG filter and caps are unchanged so results stay comparable.
* Backtest of the combination: `python scripts/backtest_lstm.py --real` -> `results_lstm/` (GFC 2008 and COVID 2020, same causal rules, paired significance tests vs the historical-mu run, Holm-corrected). **Read `results_lstm/` before drawing conclusions: the forecaster has essentially no skill over the naive baseline.**
* Dashboard (`streamlit run app.py`): tabs Live | Backtest | Ablation | Resources | About. Sidebar controls rebuild the QUBO (ESG minimum, max weight, transaction cost, risk aversion, noise level, mitigation method, mu/lambda source). Solves run in a background thread (`qpo/bgrun.py`); QAOA is limited to 3-4 assets and 256-1024 shots in the UI. Everything is labelled "research demo, not financial advice", with data source and delay.
* One environment: `requirements.txt` pins TensorFlow-CPU to a range that works with NumPy 2.x; `requirements-lock.txt` is the exact set the tests ran with. TensorFlow is imported only inside `stock_prediction/model.py`; if it conflicts with torch/Gurobi on your machine, train the models in a separate venv - the optimiser only needs the saved `models/*.keras`/`.json` through `ModelBank`.
* Screenshots: `python scripts/screenshots_combined.py` (Playwright).

## Live data (A1) and what "streaming" means here
`qpo/feed.py` defines the `next_bar()` hook: `YahooFeed` (polls Yahoo Finance for daily bars newer than the last one, rate-limited) and `ReplayFeed` (offline). `LiveOptimizer(feed=...)` consumes bars as they arrive; `catch_up()` extends the shipped snapshot (to 2023-12-29) to the latest bar. The dashboards have a **Live** data source and fall back to the snapshot, saying so, when Yahoo is unreachable. Yahoo daily bars are end-of-day and intraday quotes ~15 min delayed, so "live" = daily-bar granularity + delayed quotes; the "Streaming portfolio value" panel replays historical bars tick by tick and is labelled "streaming replay". The Yahoo code is tested with mocked payloads only (the build sandbox had no access to Yahoo).

## Logical qubits are modelled, not run (A2)
Stim + PyMatching simulate a rotated-surface-code **memory** experiment (no logical gates, no lattice surgery). The fitted logical error rate per layer enters QAOA only as an *effective independent bit-flip probability* on the measured bits (`qpo/noise.py`). The resource estimator's constants (T-count per rotation, distillation, cycle time...) are assumptions listed in `results_real/resource_assumptions.*` and swept in `results_real/resource_sweep.csv` / `resource_corners.csv` / `resource_tornado.png`.

## Layout
| path | role |
|---|---|
| `stock_prediction/`, `models/`, `app.py`, `qpo/feed.py`, `qpo/bgrun.py`, `qpo/dash_data.py` | Part 2: prediction system, trained models, combined dashboard, live feed, background workers, dashboard loaders |
| `qpo/data.py` | Yahoo/FRED/ESG loaders + synthetic regime-switching market with 2008-like & 2020-like crises |
| `qpo/qubo.py` | multi-period QUBO (verified against direct objective in tests) |
| `qpo/qaoa.py` | statevector QAOA, adaptive depth, ZNE (Richardson/exp), PEC (sampled + exact), decoding |
| `qpo/noise.py` | NISQ+ and logical noise → effective bit-flip channel; DD as idle-error suppression |
| `qpo/qec.py` | Stim memory sweeps, logical-error fit, **logical-qubit resource estimator** |
| `qpo/classical.py` | exact, SA, CVaR LP (SciPy/HiGHS), CVaR (Gurobi), ES agent |
| `qpo/rl_torch.py` | PyTorch REINFORCE agent (**written, not executed**, see below) |
| `qpo/miqp.py` | hard-budget discrete MIQP: Gurobi matrix API + exact DP fallback |
| `qpo/universe.py`, `qpo/ising.py` | ticker universes; QUBO<->Ising/dimod conversions |
| `scripts/fetch_data.py`, `build_esg.py`, `tune.py`, `resource_sweep.py`, `screenshots.py`, `optional_*.py` | data snapshot, ESG csv, causal tuning, resource sensitivity, dashboard screenshots, Ocean / Qiskit demos |
| `report/build_report.py` | generates the technical report PDF from a results folder |
| `qpo/backtest.py`, `qpo/stats.py` | walk-forward engine (no look-ahead; tested), Sharpe bootstrap, Wilcoxon, Holm |
| `qpo/theory.py`, `docs/THEORY.md` | bounds P1–P6 |
| `qpo/live.py`, `dashboard/app.py` | streaming re-optimisation + Streamlit UI |
| `scripts/` | `run_all.py`, `ablation.py`, `scaling.py` |
| `results/` | outputs of one full run on synthetic data (reference) |

## Read this before you trust anything (limitations & honesty)
1. **Everything in `results/` was run on SYNTHETIC data.** Neither build sandbox had internet. The real-data path was hardened and tested against *mocked* Yahoo/FRED payloads (all yfinance layouts, FRED's new `observation_date` header, truncated series, late tickers) but has **never touched the real endpoints**. `meta.json` records the data source, and the report/dashboard print it prominently.
2. **"Prove a quantum advantage" is not something this code (or anyone currently) can deliver.** The pipeline gives rigorous *bounds* (docs/THEORY.md) and *empirical scaling* for n ≤ 16.
   On the included runs QAOA does **not** beat exact search or SA (SA's shots-to-optimum grows slower with n). Write the report as a careful test of advantage criteria / noise thresholds — that is defensible; a claimed advantage is not.
3. **Logical qubits are modelled, not fully simulated.** Stim simulates surface-code *memory* (logical error per round vs d, p). QAOA's non-Clifford gates cannot be simulated at code level together with a 2^n statevector,
   so logical operations enter as an effective logical bit-flip rate. This is standard but must be stated. Resource-estimator constants (T-count per rotation, factory footprint, routing overhead) are order-of-magnitude assumptions — sweep/justify them.
4. **Noise model is phenomenological** (independent bit-flip per qubit derived from gate counts). DD is modelled as suppression of idle error, not pulse-level. Real devices have correlated/coherent noise.
5. n=16 qubits means exhaustive search solves each instance in <1 ms; "exact_qubo" is therefore the true optimum and a ceiling for every QUBO solver. Differences in backtest P&L between QUBO solvers reflect discretisation/noise, not alpha.
6. Weights are discretised to 7 levels over a 6-unit budget, 4 assets pre-selected from the ESG-eligible set (greedy Sharpe/diversification). CVaR-full uses all eligible assets, CVaR-sub the same 4 as the QAOA, so you can separate selection from solver effects.
7. Quantum-variant parameters are trained noiselessly and deployed under noise (the ablation also includes noise-aware training). ZNE/PEC in the backtest decode path use infinite-shot energy models; the ablation uses finite shots.
8. `qpo/rl_torch.py` (Gaussian-logit REINFORCE, same reward as the NumPy ES agent, wired in as `rl_torch_full`) was first run on a real machine and exposed two bugs, both fixed: (a) `RuntimeError: Numpy is not available` - the installed torch wheel was built for NumPy 1.x but NumPy 2.x was installed; the module now never uses torch's NumPy bridge (works with any combination), and `torch>=2.3` is required; (b) training made in-sample reward *worse* (per-state advantage standardisation amplified noise) - now pooled advantage scaling, antithetic noise, K=64. On synthetic data it improves in-sample reward on 15 of 16 seed/dataset combinations tried (not guaranteed per seed). `tests/test_rl_torch.py` is its gate. **Re-run `run_all.py` to regenerate `rl_torch_full` numbers** - any earlier torch results used the buggy estimator. `--require-optional` makes the final run fail if torch or gurobipy are missing.
9. Real-data pitfalls: survivorship bias, static ESG snapshots (look-ahead), adjusted-price dividends, VIX only from 1990, FRED CSV endpoint format changes.
10. Gurobi: `cvar_gurobi_sub`, `miqp_gurobi_full` and the Gurobi branch of `miqp_sub` were **not executed** (no gurobipy). The MIQP's quadratic form is unit-tested without Gurobi, and `miqp_sub` falls back to an exact DP with the same optimum; `tests/test_miqp.py::test_gurobi_matches_enumeration` checks Gurobi against it on your machine. The pip licence is size-limited; use your academic licence for large runs.
11. Resource estimator (changed in the final pass): one consistent exposure time, three-way failure budget incl. magic-state error, T-per-rotation baseline = Ross-Selinger 3 log2(1/eps) (the old `+9.2` constant could not be verified). All constants are swept in `results/resource_*`; the logical-error fit is extrapolated below p=2e-3.
12. On every synthetic instance in `results/` the exact QUBO optimum equals the hard-budget MIQP optimum (fraction 1.00) - so `miqp_sub` and `exact_qubo` coincide there. On real data this may differ; the summary reports the fraction.

## Windows / PowerShell troubleshooting
* `&&` is not valid in Windows PowerShell 5 - run commands on separate lines. Install requirements with `pip install -r requirements.txt`, then `python -m playwright install chromium` (the bare `playwright` command is often not on PATH; do not append `playwright install chromium` to the pip line).
* `RuntimeError: Numpy is not available` / "module compiled using NumPy 1.x cannot be run in NumPy 2.x": `pip install -U torch pyarrow` (torch>=2.3). Fallback: `pip install "numpy<2"`.
* `Windows fatal exception: access violation` while importing pyarrow: same cause (old pyarrow vs NumPy 2) - `pip install -U pyarrow` (or `pip uninstall pyarrow`, pandas does not need it).

## Suggested 4–5 day plan
* **Day 1** run real-data loaders, build ESG CSV, check the universe, re-run `pytest` and `run_all.py --fast --real`.
* **Day 2** full run on real data; inspect QUBO energy gaps, caps/λ_t behaviour in 2008 vs 2020; tune `Config` (not on the test windows!).
* **Day 3** ablation + scaling with more instances/repeats; add a stronger classical baseline (e.g., Gurobi MIQP on the discrete problem) and the PyTorch agent.
* **Day 4** theory section, resource-estimation sweeps (justify constants), dashboard screenshots.
* **Day 5** write report from `docs/REPORT_SKELETON.md`.
