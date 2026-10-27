"""F4: does the LSTM forecast (and the fuzzy risk score) improve the quantum/classical portfolio decisions?

Same causal rules as the main backtest: every decision at date t uses prices/macro strictly before t, and the LSTM used at t was trained on
data ending strictly before t (walk-forward ModelBank; the forecaster raises LookAheadError otherwise).  Same tuned configs, same costs, ESG
filter and caps (F5) - only mu (and, in the last variant, lambda_t) changes.  Variants:
    historical     trailing-mean mu, VIX-driven lambda_t   (= the main backtest's decision rule)
    lstm           mu = LSTM 21-day forecast
    blend          mu = 0.5 LSTM + 0.5 historical
    blend+fuzzy    blend, and lambda_t also driven by the fuzzy risk score
Methods that do not use mu (equal_weight_full, cvar_sub) are included as controls: their rows must be identical across variants.
  python scripts/backtest_lstm.py --real --out results_lstm [--window GFC_2008] [--fast]
"""
import argparse, json, os, sys, time, warnings
from dataclasses import replace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from qpo.config import Config
from qpo.data import load_real_market, synthetic_market
from qpo.universe import UNIVERSES
from qpo.backtest import run_backtest, QUANTUM
from qpo import qec, stats
from stock_prediction.forecast import LSTMForecaster, ModelBank

PERIODS = {"GFC_2008": ("2008-01-01", "2009-12-31"), "COVID_2020": ("2019-07-01", "2020-12-31")}
VARIANTS = {"historical": dict(mu_source="historical", risk_source="vix"), "lstm": dict(mu_source="lstm", risk_source="vix"),
            "blend": dict(mu_source="blend", risk_source="vix"), "blend+fuzzy": dict(mu_source="blend", risk_source="vix+fuzzy")}
METHODS = ["equal_weight_full", "cvar_sub", "miqp_sub", "exact_qubo", "sa_qubo", "qaoa_noisy_mitigated", "qaoa_logical"]


def forecast_quality(market, fc: LSTMForecaster, cfg, s, e):
    """Out-of-sample quality of the 21-day forecast at each rebalance date vs the historical-mean predictor and zero."""
    R = market.returns; dates = R.index
    i0 = max(int(np.searchsorted(dates, pd.Timestamp(s))), cfg.lookback); i1 = int(np.searchsorted(dates, pd.Timestamp(e)))
    elig = np.where(market.esg.values >= cfg.esg_min)[0]; tick = [R.columns[k] for k in elig]
    rows = []
    for i in range(i0, i1 - 21, cfg.rebalance_days):
        hat = fc(dates[i], tick)
        hist = R.values[i - cfg.lookback:i][:, elig].mean(0) * 21
        real = (1 + R.iloc[i:i + 21, elig]).prod().values - 1
        ok = ~np.isnan(hat)
        rows.append({"date": dates[i], "rmse_lstm": float(np.sqrt(np.mean((hat[ok] - real[ok]) ** 2))), "rmse_hist": float(np.sqrt(np.mean((hist[ok] - real[ok]) ** 2))),
                     "rmse_zero": float(np.sqrt(np.mean(real[ok] ** 2))), "ic_lstm": float(np.corrcoef(hat[ok], real[ok])[0, 1]) if hat[ok].std() > 0 else 0.0,
                     "ic_hist": float(np.corrcoef(hist[ok], real[ok])[0, 1]), "mean_pred_lstm": float(hat[ok].mean()), "xs_std_lstm": float(hat[ok].std()),
                     "mean_real": float(real[ok].mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true"); ap.add_argument("--out", default="results_lstm"); ap.add_argument("--window", default=None)
    ap.add_argument("--tuned-dir", default="results_real"); ap.add_argument("--models", default="models"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-resume", action="store_true"); ap.add_argument("--max-new", type=int, default=None, help="compute at most this many new variants, then stop (resumable chunks)")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS)); ap.add_argument("--methods", nargs="*", default=METHODS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    market = load_real_market(UNIVERSES["extended"], "2004-06-01", "2023-12-31", "data/esg.csv", "data/raw") if a.real else synthetic_market(16)
    assert a.real, "the LSTM walk-forward models were trained on the real snapshot; use --real"
    qfit = json.load(open(f"{a.tuned_dir}/qec_fit.json"))
    base = Config(n_assets=3, rebalance_days=63, shots=1024, p_max=2, opt_maxiter=40) if a.fast else Config()
    logical_q = qec.logical_q_per_layer(qfit, base.logical_distance, base.logical_p_phys)
    fc = LSTMForecaster(ModelBank(a.models, "wf_"), market.prices)
    meta = {"data_source": market.source, "variants": VARIANTS, "methods": a.methods, "fast": a.fast, "models": [str(c.date()) for c in fc.model.cutoffs]}
    for name, (s, e) in PERIODS.items():
        if a.window and a.window != name:
            continue
        tf = f"{a.tuned_dir}/tuned_config_{name}.json"
        cfg0 = replace(base, **{k: v for k, v in json.load(open(tf)).items() if not k.startswith("_")}) if os.path.exists(tf) else base
        fq = forecast_quality(market, fc, cfg0, s, e); fq.to_csv(f"{a.out}/forecast_quality_{name}.csv", index=False)
        rets = {}
        for v in a.variants:
            t0 = time.time(); print(f"[{name}] variant {v}", flush=True)
            rf = f"{a.out}/{name}_{v}_returns.csv"
            if os.path.exists(rf) and os.path.exists(f"{a.out}/{name}_{v}_records.csv") and not a.no_resume:
                rets[v] = pd.read_csv(rf, index_col=0, parse_dates=True); print("   (resumed from disk)"); continue
            if a.max_new is not None and a.max_new <= 0:
                continue
            a.max_new = None if a.max_new is None else a.max_new - 1
            cfg = replace(cfg0, **VARIANTS[v])
            ret, recs = run_backtest(market, cfg, s, e, a.methods, logical_q=logical_q, verbose=False, forecaster=fc)
            ret.to_csv(rf); recs.to_csv(f"{a.out}/{name}_{v}_records.csv", index=False)
            rets[v] = ret
            print(f"   {time.time() - t0:.0f}s", flush=True)
        if len(rets) < len(a.variants):
            print(f"[{name}] {len(rets)}/{len(a.variants)} variants done; re-run to continue"); continue
        summ = []
        for v, r in rets.items():
            t = stats.summary(r); t.insert(0, "variant", v); t.insert(0, "method", t.index); summ.append(t)
        pd.concat(summ).to_csv(f"{a.out}/summary_{name}.csv", index=False)
        tests, praw = [], {}
        for v in rets:
            if v == "historical" or "historical" not in rets:
                continue
            for m in a.methods:
                t1 = stats.paired_sharpe_test(rets[v][m].values, rets["historical"][m].values, n_boot=1000, seed=0)
                t2 = stats.wilcoxon_periods(rets[v][m], rets["historical"][m], cfg0.rebalance_days)
                tests.append({"variant": v, "method": m, "sharpe_variant": stats.sharpe(rets[v][m]), "sharpe_historical": stats.sharpe(rets["historical"][m]),
                              "diff": t1["diff"], "ci_low": t1["ci_low"], "ci_high": t1["ci_high"], "p_boot": t1["p_value"], "p_wilcoxon": t2["p_value"]})
                praw[f"{v}|{m}"] = t1["p_value"]
        if tests:
            adj = stats.holm(praw); tdf = pd.DataFrame(tests)
            tdf["p_boot_holm"] = [adj[f"{r.variant}|{r.method}"] for r in tdf.itertuples()]
            tdf["p_wilcoxon_holm"] = [stats.holm({f"{r.variant}|{r.method}": r.p_wilcoxon for r in tdf.itertuples()})[f"{r.variant}|{r.method}"] for r in tdf.itertuples()]
            tdf.to_csv(f"{a.out}/significance_vs_historical_{name}.csv", index=False)
        # quantum-vs-classical inside each variant (same test as the main backtest)
        qv = []
        for v, r in rets.items():
            qs = [m for m in a.methods if m in QUANTUM]; cs = [m for m in a.methods if m not in QUANTUM]
            pr = {f"{q}|{c}": stats.paired_sharpe_test(r[q].values, r[c].values, n_boot=500, seed=0)["p_value"] for q in qs for c in cs}
            adj = stats.holm(pr)
            qv.append({"variant": v, "n_tests": len(pr), "n_significant_holm": int(sum(p < 0.05 for p in adj.values()))})
        pd.DataFrame(qv).to_csv(f"{a.out}/quantum_vs_classical_by_variant_{name}.csv", index=False)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
