### What this app is
A research demo that joins two systems: the **fault-tolerant quantum portfolio optimizer (QPO)** and the **stock prediction system (LSTM + fuzzy logic + genetic algorithm)**.

**Data flow.** daily prices -> LSTM forecast of 21-day returns (`mu_hat`) -> expected returns of the time-dependent QUBO (`mu_source`); VIX z-score and the fuzzy risk score (volatility, VIX stress, forecast confidence) -> risk aversion `lambda_t` (`risk_source`); the QUBO is then solved by QAOA (simulated), exact search, simulated annealing, MIQP and CVaR. The quantum part optimises; it does not detect or predict anything.

### What is modelled, and what is real
* **Quantum hardware is simulated.** QAOA runs on an exact statevector simulator (<= 16 qubits). **Logical qubits are modelled:** Stim simulates surface-code *memory*; QAOA receives the fitted logical error rate as an effective bit-flip probability. Resource-estimator constants are assumptions (swept in `results_real/resource_*`).
* **Data.** Live mode = shipped snapshot (to 2023-12-29) + every daily bar Yahoo Finance has published since; Yahoo daily bars are end-of-day and quotes are ~15 minutes delayed, so "live" means *daily-bar granularity plus delayed quotes*. If Yahoo is unreachable the app says so and uses the snapshot. The streaming panel replays historical bars tick by tick.
* **Forecasts are weak.** The LSTM is evaluated on a time-ordered hold-out against the naive "tomorrow = today" baseline; its skill is about zero (see the skill metric in the prediction panel and `results_lstm/`). Treat bands and signals as illustrations of the pipeline.
* **Small problem.** Only 3-4 of the 30 assets are encoded on qubits (greedy pre-selection), so any portfolio-quality difference is dominated by that selection, discretisation and noise, not by quantum speed-up. The exact QUBO optimum is found by enumeration in under a millisecond at this size.
* Universe = survivors only; ESG = a static S&P-sourced snapshot converted to percentiles (look-ahead for old dates); one ticker has no ESG score and receives the median.

Not financial advice.
