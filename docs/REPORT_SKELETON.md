# Technical report skeleton (15–20 pages) — where every number comes from

| § | Pages | Content | Source in repo |
|---|---|---|---|
| 1 Intro & contribution framing | 1 | State honestly: "bound / test advantage", not "prove advantage" | docs/THEORY.md "NOT proven" |
| 2 Related work | 1.5 | Papers in docs/REFERENCES.md | — |
| 3 Problem formulation | 2 | multi-period QUBO, encoding, time-varying λ_t & caps, ESG hard filter + soft tilt, tx-cost surrogate, repair | qpo/qubo.py, qpo/backtest.py:make_instance |
| 4 Algorithm | 2 | QAOA, layer-wise adaptive depth, decoding, SA/exact/CVaR/Gurobi/RL baselines | qpo/qaoa.py, qpo/classical.py |
| 5 Noise & fault tolerance | 2.5 | NISQ+ noise model, Stim memory experiments, fitted p_L(d,p), logical noise for QAOA | qpo/noise.py, qpo/qec.py, results/qec_* |
| 6 (a) Logical-qubit resource estimation | 1.5 | d, physical qubits, T-count, time vs n, p, p_phys; assumptions table | results/resources.csv |
| 7 (b) Error-mitigation ablation | 2 | DD / ZNE / PEC / combos / noise-aware training / logical | results/ablation_summary.csv |
| 8 (c) Out-of-sample results + significance | 2.5 | GFC-2008 and COVID-2020 windows; Sharpe, drawdown, turnover; bootstrap + Wilcoxon, Holm | results/backtest_*, significance_* |
| 9 Scaling & theory | 2 | shots-to-optimum vs n; P1–P6 | results/scaling*.csv, docs/THEORY.md |
| 10 (d) Dashboard | 0.5 | screenshots of `streamlit run dashboard/app.py` | dashboard/app.py |
| 11 Threats to validity & conclusion | 1.5 | survivorship bias, ESG look-ahead, model-based noise, n ≤ 16, synthetic vs real | README "Limitations" |


**Update (final pass):** the report is now generated, not hand-written: `python report/build_report.py --results results_real` reads every number from the results folder and stamps a SYNTHETIC-DATA DRAFT banner on every page unless `meta.json` says `data_source = yahoo+fred+esg`. The structure above is implemented as sections 1-13 + appendices A-C.
