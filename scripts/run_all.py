"""Reproducible pipeline - ONE COMMAND PER STAGE (or `all`).  Seeds are fixed in Config.seed.

    python scripts/fetch_data.py                       # stage 0 (internet): data/raw/*.csv
    python scripts/build_esg.py                        # stage 0b (internet or your csv): data/esg.csv
    python scripts/run_all.py --stage qec              # Stim surface-code sweep + logical-error fit      (needs stim)
    python scripts/run_all.py --stage resources        # resource estimate + assumption sweeps
    python scripts/tune.py --real --target GFC_2008    # causal hyper-parameter tuning (also COVID_2020)
    python scripts/run_all.py --stage backtest --real  # walk-forward backtests + significance
    python scripts/run_all.py --stage ablation --real  # error-mitigation ablation
    python scripts/run_all.py --stage scaling  --real  # shots-to-optimum vs n
    python scripts/run_all.py --stage summary          # RESULTS.md from the csv files above
    python scripts/run_all.py --real --require-optional   # = all stages; fails if torch/gurobipy are missing
Add --fast for a ~1 min smoke run, --out DIR to keep real and synthetic results apart (results_real/, results/).
"""
import argparse, json, os, platform, sys, time
from dataclasses import replace, asdict
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qpo.config import Config
from qpo.data import synthetic_market, load_real_market
from qpo.backtest import run_backtest, ALL_METHODS, QUANTUM, OPTIONAL_METHODS, ALL_POSSIBLE
from qpo.universe import UNIVERSES
from qpo import qec, stats
import scripts.ablation as ablation
import scripts.scaling as scaling
import scripts.resource_sweep as resource_sweep

PERIODS = {"GFC_2008": ("2008-01-01", "2009-12-31"), "COVID_2020": ("2019-07-01", "2020-12-31")}
STAGES = ["qec", "resources", "backtest", "ablation", "scaling", "summary"]


def _pkg(name):
    try:
        m = __import__(name); return getattr(m, "__version__", "present")
    except Exception:
        return None


def get_market(a):
    if a.real:
        return load_real_market(UNIVERSES[a.universe], "2004-06-01", "2023-12-31", a.esg_csv, a.raw_dir, a.allow_pseudo_esg)
    return synthetic_market(n_assets=16)


def get_cfg(a):
    return Config(n_assets=3, rebalance_days=63, shots=1024, p_max=2, opt_maxiter=40) if a.fast else Config()


def logical_q_from_fit(out, cfg):
    model = json.load(open(f"{out}/qec_fit.json"))
    return model, qec.logical_q_per_layer(model, cfg.logical_distance, cfg.logical_p_phys)


def stage_qec(a, cfg):
    print("[qec] Stim surface-code sweep (cache: qec_sweep.json)")
    rows = qec.sweep(ds=(3, 5, 7), ps=(0.002, 0.004, 0.006), shots=10000 if a.fast else 30000, cache=f"{a.out}/qec_sweep.json")
    pd.DataFrame(rows).to_csv(f"{a.out}/qec_sweep.csv", index=False)
    model = qec.fit_logical_model(rows)
    json.dump(model, open(f"{a.out}/qec_fit.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for d in (3, 5, 7):
        r = [x for x in rows if x["d"] == d]
        ax.errorbar([x["p"] for x in r], [x["p_L_per_round"] for x in r], fmt="o-", label=f"d={d}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("physical error p"); ax.set_ylabel("logical error / round")
    ax.legend(); ax.set_title("Stim + PyMatching rotated surface code (memory only)"); fig.tight_layout()
    fig.savefig(f"{a.out}/qec_threshold.png", dpi=150); plt.close(fig)
    _, lq = logical_q_from_fit(a.out, cfg)
    print(f"  p_th~{model['p_th']:.4f}; logical flip prob/layer at d={cfg.logical_distance}, p={cfg.logical_p_phys}: {lq:.2e}")


def stage_resources(a, cfg):
    print("[resources] estimator + sensitivity sweeps")
    resource_sweep.main(a.out)


def _cfg_for_period(cfg, name, a):
    f = os.path.join(a.tuned_dir, f"tuned_config_{name}.json")
    if os.path.exists(f):
        t = {k: v for k, v in json.load(open(f)).items() if not k.startswith("_")}
        print(f"  using tuned config for {name}: {t}")
        return replace(cfg, **t), t
    return cfg, None


def stage_backtest(a, cfg, market, methods):
    print("[backtest] walk-forward backtests")
    _, logical_q = logical_q_from_fit(a.out, cfg)
    tuned_used = {}
    for name, (s, e) in PERIODS.items():
        cfg_p, t = _cfg_for_period(cfg, name, a)
        tuned_used[name] = t
        print(f" period {name}")
        ret, recs = run_backtest(market, cfg_p, s, e, methods, logical_q=logical_q)
        ret.to_csv(f"{a.out}/backtest_{name}_returns.csv"); recs.to_csv(f"{a.out}/backtest_{name}_records.csv", index=False)
        summ = stats.summary(ret)
        summ["avg_turnover"] = recs.groupby("method")["turnover"].mean()
        summ["avg_seconds"] = recs.groupby("method")["seconds"].mean()
        summ.to_csv(f"{a.out}/backtest_{name}_summary.csv")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for m in ret:
            ax.plot((1 + ret[m]).cumprod(), label=m, lw=2 if m in QUANTUM else 1, ls="-" if m in QUANTUM else "--")
        ax.legend(fontsize=6.5, ncol=2); ax.set_title(f"{name}: cumulative net wealth ({market.source})"); fig.tight_layout()
        fig.savefig(f"{a.out}/backtest_{name}_cum.png", dpi=150); plt.close(fig)
        quantum = [m for m in methods if m in QUANTUM]; classical = [m for m in methods if m not in QUANTUM]
        tests, praw = [], {}
        for q in quantum:
            for c in classical:
                t1 = stats.paired_sharpe_test(ret[q].values, ret[c].values, n_boot=1000, seed=cfg.seed)
                t2 = stats.wilcoxon_periods(ret[q], ret[c], cfg_p.rebalance_days)
                tests.append({"quantum": q, "classical": c, "sharpe_diff": t1["diff"], "ci_low": t1["ci_low"], "ci_high": t1["ci_high"],
                              "p_boot": t1["p_value"], "p_wilcoxon": t2["p_value"]})
                praw[f"{q}|{c}"] = t1["p_value"]
        adj = stats.holm(praw)
        tdf = pd.DataFrame(tests)
        tdf["p_boot_holm"] = [adj[f"{r.quantum}|{r.classical}"] for r in tdf.itertuples()]
        tdf["p_wilcoxon_holm"] = [stats.holm({f"{r.quantum}|{r.classical}": r.p_wilcoxon for r in tdf.itertuples()})[f"{r.quantum}|{r.classical}"]
                                  for r in tdf.itertuples()]
        tdf.to_csv(f"{a.out}/significance_{name}.csv", index=False)
    return tuned_used


def stage_ablation(a, cfg, market):
    print("[ablation] error-mitigation ablation")
    _, logical_q = logical_q_from_fit(a.out, cfg)
    ablation.run(cfg, market, n_inst=4 if a.fast else 8, repeats=6 if a.fast else 12, logical_q=logical_q, out=a.out)


def stage_scaling(a, cfg, market):
    print("[scaling] shots-to-optimum vs n")
    scaling.run(cfg, market, inst_per_size=2 if a.fast else 5, out=a.out)


def _md(df, nd=4):
    return df.round(nd).to_markdown()


def stage_summary(a, meta):
    print("[summary] RESULTS.md")
    O = a.out
    md = ["# Results (auto-generated)\n", f"data source: **{meta.get('data_source', '?')}**\n"]
    if meta.get("data_source", "").startswith("synthetic"):
        md.append("> **SYNTHETIC DATA - nothing below says anything about real markets.**\n")
    for name, (s, e) in PERIODS.items():
        f = f"{O}/backtest_{name}_summary.csv"
        if not os.path.exists(f):
            continue
        summ = pd.read_csv(f, index_col=0); tdf = pd.read_csv(f"{O}/significance_{name}.csv")
        md += [f"\n## Backtest {name} ({s} .. {e})\n", _md(summ),
               f"\nSignificant (Holm-adjusted bootstrap p<0.05) quantum-vs-classical Sharpe differences: {int((tdf.p_boot_holm < 0.05).sum())} / {len(tdf)}"
               f"  (Wilcoxon, Holm: {int((tdf.p_wilcoxon_holm < 0.05).sum())} / {len(tdf)})\n"]
        rec = pd.read_csv(f"{O}/backtest_{name}_records.csv")
        q = rec[rec.method.isin(QUANTUM + ["exact_qubo", "sa_qubo", "miqp_sub"])].groupby("method").agg(
            gap_norm=("gap_norm", "mean"), approx_ratio=("approx_ratio", "mean"), optimum_hit_rate=("is_opt", "mean"))
        md += ["\nQUBO solution quality per solver (0 gap = optimal):\n", _md(q)]
        if "miqp_obj" in rec and {"miqp_sub", "exact_qubo"} <= set(rec.method):
            a1 = rec[rec.method == "miqp_sub"].set_index("date"); a2 = rec[rec.method == "exact_qubo"].set_index("date")
            same = float((np.abs(a1["miqp_obj"] - a2["E_opt"]) < 1e-9).mean())
            md.append(f"\nFraction of rebalances where the exact QUBO optimum equals the hard-budget MIQP optimum: {same:.2f}\n")
    for fname, title in [("ablation_summary.csv", "Error-mitigation ablation"), ("scaling.csv", "Shots/evaluations to optimum vs n")]:
        if os.path.exists(f"{O}/{fname}"):
            md += [f"\n## {title}\n", _md(pd.read_csv(f"{O}/{fname}", index_col=0), 4 if "abl" in fname else 2)]
    if os.path.exists(f"{O}/scaling_fits.csv"):
        md += ["\nFitted exponents (cost ~ 2^(slope*n)):\n", _md(pd.read_csv(f"{O}/scaling_fits.csv", index_col=0), 3)]
    if os.path.exists(f"{O}/resource_corners.csv"):
        md += ["\n## Resource estimate corners\n", _md(pd.read_csv(f"{O}/resource_corners.csv"), 3)]
    open(f"{O}/RESULTS.md", "w").write("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="comma list of: " + ",".join(STAGES) + " | all")
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--real", action="store_true", help="use data/raw (fetch_data.py) + ESG csv")
    ap.add_argument("--universe", default="extended", choices=list(UNIVERSES)); ap.add_argument("--esg-csv", default="data/esg.csv")
    ap.add_argument("--raw-dir", default="data/raw"); ap.add_argument("--out", default=None, help="default: results_real if --real else results")
    ap.add_argument("--tuned-dir", default=None, help="folder with tuned_config_<period>.json (default: --out)")
    ap.add_argument("--allow-pseudo-esg", action="store_true", help="smoke tests only")
    ap.add_argument("--require-optional", action="store_true", help="fail if torch / gurobipy are missing (use for the final real run)")
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args()
    a.out = a.out or ("results_real" if a.real else "results")
    a.tuned_dir = a.tuned_dir or a.out
    os.makedirs(a.out, exist_ok=True)
    stages = STAGES if a.stage == "all" else [s.strip() for s in a.stage.split(",")]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"unknown stage(s) {bad}")
    methods = a.methods or ALL_METHODS
    if a.require_optional:
        miss = [f"{m} (needs {p})" for m, p in OPTIONAL_METHODS.items() if _pkg(p) is None and m in ALL_POSSIBLE]
        if miss:
            sys.exit("--require-optional: missing " + "; ".join(miss) + "   -> pip install torch gurobipy")
    cfg = get_cfg(a)
    meta_f = f"{a.out}/meta.json"
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
    t0 = time.time()
    market = None
    if any(s in stages for s in ("backtest", "ablation", "scaling")):
        market = get_market(a)
        print(f"data source: {market.source}; assets={market.returns.shape[1]}; days={len(market.returns)}; "
              f"{market.returns.index[0].date()}..{market.returns.index[-1].date()}")
        if a.real:
            print("  meta:", {k: v for k, v in market.meta.items() if k in ("dropped_tickers", "dropped_macro", "esg_missing", "esg_source")})
        meta.update({"data_source": market.source, "data_meta": {k: (v if not isinstance(v, (np.ndarray,)) else v.tolist()) for k, v in market.meta.items()},
                     "n_assets": int(market.returns.shape[1]), "first_date": str(market.returns.index[0].date()),
                     "last_date": str(market.returns.index[-1].date())})
    if "qec" in stages:
        stage_qec(a, cfg)
    if "resources" in stages:
        stage_resources(a, cfg)
    if "backtest" in stages:
        meta["tuned_config_used"] = stage_backtest(a, cfg, market, methods)
        meta["methods"] = methods
    if "ablation" in stages:
        stage_ablation(a, cfg, market)
    if "scaling" in stages:
        stage_scaling(a, cfg, market)
    meta.update({"config": asdict(cfg), "fast": a.fast, "seed": cfg.seed, "stages_last_run": stages,
                 "python": platform.python_version(),
                 "packages": {p: _pkg(p) for p in ("numpy", "scipy", "pandas", "stim", "pymatching", "torch", "gurobipy", "yfinance", "qiskit", "dimod")},
                 "optional_methods_available": {m: (_pkg(p) is not None) for m, p in OPTIONAL_METHODS.items()},
                 "unix_time": int(time.time())})
    if "summary" in stages:
        stage_summary(a, meta)
    json.dump(meta, open(meta_f, "w"), indent=1, default=str)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}/")


if __name__ == "__main__":
    main()
