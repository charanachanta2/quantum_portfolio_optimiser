"""STAGE `tune`: choose Config hyper-parameters using ONLY data that precedes the test window being evaluated.

    python scripts/tune.py --real --target GFC_2008      # validation: 2006-01..2007-09 (pre-crisis)
    python scripts/tune.py --real --target COVID_2020    # validation: 2012-01..2013-12 and 2016-01..2018-06

Selection criterion = net Sharpe of the EXACT QUBO optimiser (solver-independent: no quantum/classical method is
favoured, all methods then use the same Config).  Output: results/tuned_config_<target>.json (+ full grid csv).
Honest limitation: the GFC validation window is short and calm (no crash in it), so it can only pick sensible
risk-aversion scales - it cannot "learn" crisis behaviour; the VIX-driven constraints must do that.
"""
import argparse, itertools, json, os, sys
from dataclasses import replace, asdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo.config import Config
from qpo.data import synthetic_market, load_real_market
from qpo.backtest import run_backtest
from qpo.stats import sharpe
from qpo.universe import UNIVERSES

VAL = {"GFC_2008": [("2006-01-01", "2007-09-30")],
       "COVID_2020": [("2012-01-01", "2013-12-31"), ("2016-01-01", "2018-06-30")]}
TEST_START = {"GFC_2008": "2008-01-01", "COVID_2020": "2019-07-01"}
GRID = {"lambda0": [3.0, 6.0, 10.0], "kappa_vix": [0.0, 0.5, 1.0], "esg_tilt": [0.0, 0.002, 0.01], "tc_quad": [0.005, 0.02, 0.05]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(VAL)); ap.add_argument("--real", action="store_true")
    ap.add_argument("--universe", default="extended"); ap.add_argument("--esg-csv", default="data/esg.csv")
    ap.add_argument("--raw-dir", default="data/raw"); ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--allow-pseudo-esg", action="store_true")
    a = ap.parse_args()
    cfg0 = Config(n_assets=3, rebalance_days=63) if a.fast else Config()
    market = (load_real_market(UNIVERSES[a.universe], "2004-06-01", "2023-12-31", a.esg_csv, a.raw_dir, a.allow_pseudo_esg)
              if a.real else synthetic_market(n_assets=16))
    for s, e in VAL[a.target]:
        assert pd.Timestamp(e) < pd.Timestamp(TEST_START[a.target]), "validation window must precede the test window"
    keys = list(GRID); rows = []
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    for k, vals in enumerate(combos):
        cfg = replace(cfg0, **dict(zip(keys, vals)))
        shs = []
        for s, e in VAL[a.target]:
            ret, _ = run_backtest(market, cfg, s, e, ["exact_qubo"], verbose=False)
            shs.append(sharpe(ret["exact_qubo"].values))
        rows.append({**dict(zip(keys, vals)), "val_sharpe": float(np.mean(shs))})
        if k % 10 == 0:
            print(f"  grid {k + 1}/{len(combos)}  best so far {max(r['val_sharpe'] for r in rows):.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/tune_grid_{a.target}.csv", index=False)
    best = df.iloc[0]
    out = {k: float(best[k]) for k in keys}
    out["_meta"] = {"target": a.target, "validation_windows": VAL[a.target], "criterion": "mean net Sharpe of exact_qubo",
                    "val_sharpe_best": float(best["val_sharpe"]), "val_sharpe_median_grid": float(df["val_sharpe"].median()),
                    "data_source": market.source, "grid": GRID}
    json.dump(out, open(f"{a.out}/tuned_config_{a.target}.json", "w"), indent=1)
    print(df.head(5).round(4).to_string()); print("-> best", {k: out[k] for k in keys})


if __name__ == "__main__":
    main()
