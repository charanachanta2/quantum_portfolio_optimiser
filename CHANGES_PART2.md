# What was added in this package (real gaps A1-A2 + Part 2, E-H)

Status of the checklist items (everything below was run in the build sandbox; "not run" is stated where true).

## A. Real gaps
- [x] **A1 live feed.** `qpo/feed.py` (`next_bar()` hook, `YahooFeed`, `ReplayFeed`, `catch_up`), `LiveOptimizer(feed=...)`, `solve_now()`; "Live" source in `dashboard/app.py` and in the combined `app.py`; "streaming replay" wording where bars are replayed. **Caveat:** the sandbox has no Yahoo access, so the Yahoo code is tested with mocked payloads only (tests/test_feed_dash.py); the app falls back to the snapshot and says so.
- [x] **A2 logical qubits are modelled.** Stated in README (new section), report section 13 (threat bullet), dashboard About tab, and the Quantum-internals caption. Resource constants called assumptions swept in `results_real/resource_*`.

## E. Stock Prediction System repaired (`stock_prediction/`)
- [x] E1 fuzzy logic (`fuzzy.py`) and genetic algorithm (`ga.py`) **implemented** (they did not exist in the original code). GA run with a small budget (6x3): `models/ga_best.json`.
- [x] E2 no training on requests: `scripts/train_predictor.py` (offline) -> `models/*.keras` + `.json`; Flask and Streamlit only load (cached).
- [x] E3 module split (`features`, `model`, `fuzzy`, `ga`, `forecast`, `service`, `data`, `flask_app` = routes only).
- [x] E4 honest evaluation: time-ordered train/val/test with purged labels, train-only scalers, RMSE vs naive "tomorrow = today". **The original 7-day "forecast" was random noise around the last price; it is replaced by direct multi-horizon LSTM outputs.** Result: skill vs naive about 0 (deployed model, test: see `models/deployed.json`).
- [x] E5 pins: `requirements.txt` (ranges) + `requirements-lock.txt` (exact versions the tests ran with: TF-CPU 2.21, NumPy 2.4, pandas 3.0). **Not run:** torch and Gurobi were not installed in the sandbox, so coexistence with them is by version ranges only; process separation is possible via saved models (see README).

## F. Integration
- [x] F1 `stock_prediction.forecast.forecast_returns(tickers, as_of_date, prices, model)` - data strictly before `as_of_date`; `LookAheadError` for models that saw it; walk-forward `ModelBank`.
- [x] F2 `Config.mu_source = historical | lstm | blend` in `qpo.backtest.make_instance` (default path verified identical to the old code).
- [x] F3 `Config.risk_source = vix | vix+fuzzy` (fuzzy risk from volatility, VIX z, forecast confidence).
- [x] F4 `scripts/backtest_lstm.py` -> `results_lstm/` (GFC 2008, COVID 2020; paired bootstrap + Wilcoxon, Holm). **Result: LSTM mu does not beat historical mu** - 0 significant differences in either window; forecast RMSE is not better than predicting zero.
- [x] F5 costs, ESG filter, caps, asset pre-selection unchanged.

## G. Dashboard (`streamlit run app.py`)
- [x] G1-G10 implemented: tabs Live | Backtest | Ablation | Resources | About; market strip with refresh; prediction panel (band, confidence, risk gauge, signal); optimiser panel (QAOA + exact QUBO + MIQP + CVaR + equal weight, turnover, model-implied Sharpe, solve time); constraint sliders; streaming replay P&L plus live mark-to-market when quotes are available; quantum internals; results tabs; background-thread solves (`qpo/bgrun.py`), QAOA limited to 3-4 assets and 256-1024 shots; safety labels.

## H. Tests and delivery
- [x] H1 tests: `test_forecast.py` (no look-ahead, split purging, bank), `test_fuzzy_ga.py`, `test_feed_dash.py` (mocked yfinance), `test_mu_source.py`, `test_app_combined.py` (AppTest). Suite: 51 passed, 5 skipped (torch/Gurobi/Qiskit-only tests).
- [x] H2 one command: `streamlit run app.py`; `requirements.txt`, README quick start updated. Clean-venv install was not performed in the sandbox (packages were installed into the sandbox Python).
- [x] H3 screenshots of every tab: `results_real/screens_combined/` (taken with headless Chromium; the build machine was offline, so banners show "OFFLINE SNAPSHOT").
- [x] H4 report section 10.1 (combined system and limitations); report rebuilt (`report/QPO_Technical_Report.pdf`).

Not part of this package (see the other zip): B (report wording), C (polish), D (final checks).
