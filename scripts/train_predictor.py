"""Train the LSTM predictor OFFLINE and save it (E2).

  python scripts/train_predictor.py                      # deployed model: all data in the snapshot, 70/15/15 time-ordered split
  python scripts/train_predictor.py --refresh            # same, but first download fresh prices with yfinance (needs internet)
  python scripts/train_predictor.py --walk-forward 2007-12-31 2008-12-31 2018-12-31 2019-12-31   # models for the backtests
  python scripts/train_predictor.py --hparams models/ga_best.json                                 # use GA-tuned hyper-parameters
"""
import argparse, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_prediction import data as D
from stock_prediction.model import HParams, train_pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models"); ap.add_argument("--hparams", default=None)
    ap.add_argument("--walk-forward", nargs="*", default=None, help="training cut-off dates; saves wf_<date> models (no test split)")
    ap.add_argument("--refresh", action="store_true"); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    hp = HParams()
    if a.hparams and os.path.exists(a.hparams):
        h = json.load(open(a.hparams))["hparams"]
        hp = HParams(**{k: v for k, v in h.items() if k in HParams.__dataclass_fields__})
    if a.epochs:
        hp.epochs = a.epochs
    if a.refresh:
        snap = D.load_snapshot()
        px = D.fetch_prices(tuple(snap.columns), start="2004-06-01")
        px = px.reindex(columns=snap.columns).ffill()
    else:
        px = D.load_snapshot()
    print(f"prices {px.shape} {px.index[0].date()}..{px.index[-1].date()}  hp={hp}")
    if a.walk_forward:
        for c in a.walk_forward:
            p = train_pooled(px, c, hp, frac_val=0.2, frac_test=0.0, seed=a.seed, name=f"wf_{c}")
            p.save(a.out)
            sk = p.metrics["val"]["skill_vs_naive"]
            print(f"  wf_{c}: n_train={p.metrics['n_train']} val skill vs naive h1={sk['h1']:+.3f} h21={sk['h21']:+.3f}")
    else:
        p = train_pooled(px, px.index[-1], hp, frac_val=0.15, frac_test=0.15, seed=a.seed, name="deployed")
        p.save(a.out)
        t = p.metrics["test"]
        print("test (out-of-sample, time-ordered) skill vs naive:", t["skill_vs_naive"])
        print("direction accuracy:", t["direction_acc"], " base rate:", t["direction_base_rate"])


if __name__ == "__main__":
    main()
