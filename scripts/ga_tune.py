"""Genetic-algorithm tuning of the LSTM hyper-parameters (E1).  Fitness = validation skill loss of a short training run,
computed on data BEFORE --train-end only.  Budget is deliberately small (default 6 x 3); raise --pop/--gens for a real search.
  python scripts/ga_tune.py --train-end 2018-12-31 --pop 6 --gens 3       ->  models/ga_best.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_prediction import data as D
from stock_prediction.ga import decode, real_ga, NAMES
from stock_prediction.model import HParams, val_loss_for_ga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2018-12-31"); ap.add_argument("--pop", type=int, default=6); ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--out", default="models/ga_best.json"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    px = D.load_snapshot()
    px = px[[c for c in px.columns][:12]]                     # 12 tickers keep each fitness call to a few seconds
    t0 = time.time(); n = [0]

    def fit(ch):
        d = decode(ch)
        hp = HParams(seq_len=d["seq_len"], units=d["units"], dropout=d["dropout"], lr=d["lr"], batch=d["batch"], epochs=a.epochs, stride=4)
        n[0] += 1
        return val_loss_for_ga(px, a.train_end, hp, seed=a.seed)

    best, f, hist = real_ga(fit, a.pop, a.gens, a.seed, verbose=True)
    d = decode(best)
    json.dump({"hparams": d, "chromosome": dict(zip(NAMES, map(float, best))), "fitness": f, "history": hist, "train_end": a.train_end,
               "evaluations": n[0], "pop": a.pop, "gens": a.gens, "seconds": round(time.time() - t0)}, open(a.out, "w"), indent=1)
    print("best", d, "fitness", round(f, 5), "history", [round(x, 5) for x in hist], f"({n[0]} evaluations)")


if __name__ == "__main__":
    main()
