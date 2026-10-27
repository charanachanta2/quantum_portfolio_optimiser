# STATUS after the final pass (honest version)

**Build environment:** no internet, no `pip`; `stim`, `pymatching`, `torch`, `gurobipy`, `streamlit`, `qiskit`, `dimod`, `pytest` all missing. A tiny pytest stand-in ran the tests.
**Test result here:** 21 passed, 7 skipped (stim, torch x2, gurobipy x2, qiskit, streamlit), 0 failed. On your machine the skipped ones are the acceptance tests for the untested pieces.

| # | Item | Status |
|---|------|--------|
| 1 | Real data | **NOT DONE - needs your laptop.** Loader rewritten and tested on mocked payloads; `fetch_data.py` (frozen csv snapshot), `build_esg.py` (Sustainalytics risk -> percentile score, or your MSCI csv), `qpo/universe.py` (20 core + 10 crisis-hit tickers). Found and fixed a look-ahead bug: `.bfill()` of macro data (FRED now truncates ICE-BofA HY spread to 3 years, which would have copied 2023 values into 2008). I did **not** invent ESG scores; pseudo-ESG is refused unless `--allow-pseudo-esg`. |
| 2 | Re-run on real data | **NOT DONE.** Staged `run_all.py` (one command per stage, `meta.json` provenance, `--require-optional`), causal `tune.py` (GFC tuned on 2006-07 only; COVID on 2012-18 only; criterion = exact_qubo Sharpe). Synthetic pipeline re-run end to end: reproduces the handoff numbers exactly. |
| 3 | PyTorch RL | **Written, NOT executed.** Redesigned (per-state baseline, Gaussian-logit policy, same reward as ES agent), wired as `rl_torch_full`. Gate: `pytest tests/test_rl_torch.py`. |
| 4 | Gurobi MIQP | **Partly verified.** `miqp_sub` runs everywhere (Gurobi or exact DP). DP verified against brute force; QUBO optimum == MIQP optimum verified (cross-validates the QUBO build); Gurobi quadratic form verified in numpy. Gurobi call itself not run. `miqp_gurobi_full` Gurobi-only. |
| 5 | Resource rigor | **Done.** Estimator made consistent (+ magic-state error check), assumptions table with sources/confidence, one-at-a-time + corner sweeps, tornado plot. T-per-rotation constant `+9.2` was unverifiable and removed. Stim = memory only (stated in README/report). |
| 6 | THEORY P3 | **Done.** n=100,q=.03: 2.4e5 (as you said); n=16 was also wrong (6.6 -> 7.2). Pinned by a test. |
| 7 | Optional | Ocean SA + Qiskit circuit demos written; shared math (`qpo/ising.py`) tested in numpy; the dimod/qiskit API calls **not run**. |
| 8 | Dashboard screenshots | **NOT DONE.** Dashboard now has a data-source switch + banner (logic exercised against a Streamlit stub: idle, 25-tick stream, real-data error path). `scripts/screenshots.py` (Playwright) written, not run. The screenshots inside the walkthrough are old synthetic ones, labelled as such. |
| 9 | Technical report | **Draft generated, 15 pages, SYNTHETIC banner on every page.** `report/QPO_Technical_Report_SYNTHETIC_DRAFT.pdf`. All numbers are read from `results/`; running `build_report.py --results results_real` after the real run produces the final one (banner disappears only if `meta.json` says real). Sections 5-6 + theory are data-independent and valid as is. |
| 10 | Walkthrough | **Regenerated from the current (synthetic) results**, paths fixed, numbers computed not typed, labelled SYNTHETIC. Re-run `make walkthrough` after the real run. |

## Your exact commands (laptop; Python 3.10+)
```bash
cd qpo_project && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && playwright install chromium
python -m pytest -q                       # expect: 0 failed; read which tests SKIP
python -m pytest tests/test_rl_torch.py tests/test_miqp.py -q    # gates for the untested pieces
python scripts/fetch_data.py              # internet. check printed "missing price rows"
python scripts/build_esg.py               # or: --from-csv your_msci.csv --ticker-col Ticker --score-col Score --higher-is-better
python scripts/run_all.py --stage qec --out results_real          # re-runs Stim (or copy results/qec_sweep.json to skip)
python scripts/run_all.py --stage resources --out results_real
python scripts/tune.py --real --target GFC_2008 --out results_real; python scripts/tune.py --real --target COVID_2020 --out results_real
python scripts/run_all.py --real --require-optional --stage backtest,ablation,scaling,summary --out results_real
python scripts/screenshots.py --real --out results_real
python report/build_report.py --results results_real --out report/QPO_Technical_Report.pdf
make walkthrough                          # (or the 4 commands inside it)
```
Start with `python scripts/run_all.py --real --fast --allow-pseudo-esg --out /tmp/x` only as a plumbing check (pseudo-ESG results must never be reported).

## Things I would check first on real data
1. `meta.json -> data_meta`: dropped tickers / macro, ESG provenance. 2. How many tickers survive the `esg_min=40` filter. 3. Whether `miqp_sub` still equals `exact_qubo` (fraction in RESULTS.md). 4. Whether the logical-error fit still gives p_th ~ 1e-2.
