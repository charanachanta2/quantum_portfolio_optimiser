"""Pooled LSTM return forecaster - trained OFFLINE (E2), evaluated on a time-ordered hold-out (E4).

Honest-evaluation rules implemented here
  * time-ordered split train | validation | test, never shuffled across the split boundary;
  * a sample whose label window (t .. t+21) reaches into the next split is dropped (purging), so no label leaks;
  * feature/target scalers are fitted on the TRAIN samples only;
  * metrics are reported against the naive baseline "tomorrow = today" (predicted return 0);
  * `Predictor.train_end` is stored so callers can refuse to use a model on dates it has already seen (forecast.py).
One model serves every ticker (features are price-normalised); the original prototype trained one model per request.
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .features import FEATURES, HORIZONS, H_MAX, build_features, make_targets

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class HParams:
    seq_len: int = 30
    units: int = 32
    dense: int = 32
    dropout: float = 0.2
    lr: float = 1e-3
    batch: int = 128
    epochs: int = 8
    stride: int = 3            # take every k-th end date per ticker (keeps pooled training set small; overlap is high anyway)


def _keras():
    from tensorflow import keras
    return keras


def build_network(hp: HParams, n_feat: int, n_out: int):
    k = _keras()
    m = k.Sequential([k.layers.Input((hp.seq_len, n_feat)), k.layers.LSTM(hp.units), k.layers.Dropout(hp.dropout),
                      k.layers.Dense(hp.dense, activation="relu"), k.layers.Dense(n_out)])
    m.compile(optimizer=k.optimizers.Adam(hp.lr), loss="mse")
    return m


def prepare_panel(close: pd.DataFrame) -> dict:
    """Per-ticker (dates, features, targets) arrays; columns with NaN are dropped (caller forward-fills)."""
    out = {}
    for t in close.columns:
        c = close[t].dropna()
        if len(c) < 120:
            continue
        f, y = build_features(c), make_targets(c)
        out[t] = (c.index, f.values.astype(np.float64), y.values.astype(np.float64))
    return out


def _samples(prep, seq_len, d_lo, d_hi, label_limit, stride):
    """Windows ending at date e with d_lo <= date_e <= d_hi (d_hi exclusive if None) and label end date <= label_limit."""
    X, Y, D, T = [], [], [], []
    for t, (idx, F, Yt) in prep.items():
        valid = ~np.isnan(F).any(1)
        p0 = int(np.argmax(valid)) if valid.any() else len(idx)
        lo_pos = p0 + seq_len - 1
        for e in range(lo_pos, len(idx) - H_MAX, stride):
            d = idx[e]
            if d < d_lo or (d_hi is not None and d >= d_hi):
                continue
            if idx[e + H_MAX] > label_limit or np.isnan(Yt[e]).any():
                continue
            X.append(F[e - seq_len + 1:e + 1]); Y.append(Yt[e]); D.append(d); T.append(t)
    if not X:
        return np.zeros((0, seq_len, len(FEATURES))), np.zeros((0, len(HORIZONS))), [], []
    return np.asarray(X), np.asarray(Y), D, T


class Predictor:
    def __init__(self, net, hp: HParams, x_mean, x_std, y_std, resid_std, train_end, metrics, name="model"):
        self.net, self.hp = net, hp
        self.x_mean, self.x_std, self.y_std = np.asarray(x_mean), np.asarray(x_std), np.asarray(y_std)
        self.resid_std = np.asarray(resid_std)
        self.train_end = pd.Timestamp(train_end)
        self.metrics = metrics
        self.name = name

    # ---------------------------------------------------------------- inference
    def predict_panel(self, close: pd.DataFrame, as_of=None) -> pd.DataFrame:
        """Predicted cumulative log-returns (rows = tickers, cols = h1..h21) from data STRICTLY BEFORE `as_of`
        (all data if as_of is None).  Tickers with too little clean history return NaN."""
        rows, names = [], []
        for t in close.columns:
            c = close[t].dropna()
            if as_of is not None:
                c = c[c.index < pd.Timestamp(as_of)]
            f = build_features(c).dropna() if len(c) else pd.DataFrame()
            names.append(t)
            if len(f) < self.hp.seq_len:
                rows.append(np.full((self.hp.seq_len, len(FEATURES)), np.nan)); continue
            rows.append(f.values[-self.hp.seq_len:])
        X = np.asarray(rows)
        ok = ~np.isnan(X).any((1, 2))
        out = np.full((len(names), len(HORIZONS)), np.nan)
        if ok.any():
            Xs = ((X[ok] - self.x_mean) / self.x_std).astype(np.float32)
            out[ok] = self.net.predict(Xs, verbose=0) * self.y_std
        return pd.DataFrame(out, index=names, columns=[f"h{h}" for h in HORIZONS])

    # ---------------------------------------------------------------- persistence
    def save(self, folder: str, name: str | None = None):
        name = name or self.name
        os.makedirs(folder, exist_ok=True)
        self.net.save(os.path.join(folder, f"{name}.keras"))
        meta = {"hp": asdict(self.hp), "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(), "y_std": self.y_std.tolist(),
                "resid_std": self.resid_std.tolist(), "train_end": str(self.train_end.date()), "metrics": self.metrics,
                "features": FEATURES, "horizons": list(HORIZONS)}
        json.dump(meta, open(os.path.join(folder, f"{name}.json"), "w"), indent=1)

    @classmethod
    def load(cls, folder: str, name: str) -> "Predictor":
        meta = json.load(open(os.path.join(folder, f"{name}.json")))
        if meta["features"] != FEATURES or meta["horizons"] != list(HORIZONS):
            raise ValueError(f"model {name} was trained with a different feature/horizon set; retrain it")
        net = _keras().models.load_model(os.path.join(folder, f"{name}.keras"))
        return cls(net, HParams(**meta["hp"]), meta["x_mean"], meta["x_std"], meta["y_std"], meta["resid_std"],
                   meta["train_end"], meta["metrics"], name)


def evaluate(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Model vs naive 'tomorrow = today' (zero predicted return) on the SAME samples."""
    e_m = np.sqrt(np.mean((pred - truth) ** 2, 0)); e_n = np.sqrt(np.mean(truth ** 2, 0))
    up = (truth > 0).mean(0)
    dir_acc = ((pred > 0) == (truth > 0)).mean(0)
    return {"n_samples": int(len(truth)), "rmse_model": dict(zip([f"h{h}" for h in HORIZONS], e_m.round(5).tolist())),
            "rmse_naive": dict(zip([f"h{h}" for h in HORIZONS], e_n.round(5).tolist())),
            "skill_vs_naive": dict(zip([f"h{h}" for h in HORIZONS], (1 - e_m / e_n).round(4).tolist())),
            "direction_acc": dict(zip([f"h{h}" for h in HORIZONS], dir_acc.round(4).tolist())),
            "direction_base_rate": dict(zip([f"h{h}" for h in HORIZONS], np.maximum(up, 1 - up).round(4).tolist())),
            "corr_ic": dict(zip([f"h{h}" for h in HORIZONS], [float(np.corrcoef(pred[:, j], truth[:, j])[0, 1]) if pred[:, j].std() > 0 else 0.0 for j in range(truth.shape[1])]))}


def train_pooled(close: pd.DataFrame, train_end, hp: HParams | None = None, frac_val: float = 0.2, frac_test: float = 0.0,
                 seed: int = 0, verbose: int = 0, name: str = "model") -> Predictor:
    """Train on data up to `train_end` (inclusive).  Time-ordered split by DATE: train | val | (test if frac_test > 0).
    With frac_test = 0 (walk-forward use) the later backtest window is the out-of-sample test."""
    hp = hp or HParams()
    k = _keras(); k.utils.set_random_seed(seed)
    train_end = pd.Timestamp(train_end)
    close = close[close.index <= train_end]
    prep = prepare_panel(close)
    dates = close.index
    n = len(dates)
    t_test = dates[int(n * (1 - frac_test))] if frac_test > 0 else None
    t_val = dates[int(n * (1 - frac_test - frac_val))]
    d0 = dates[0]
    Xtr, Ytr, _, _ = _samples(prep, hp.seq_len, d0, t_val, t_val - pd.Timedelta(days=1), hp.stride)         # labels end before val starts
    Xva, Yva, _, _ = _samples(prep, hp.seq_len, t_val, t_test, (t_test - pd.Timedelta(days=1)) if t_test is not None else train_end, hp.stride)
    if len(Xtr) < 200 or len(Xva) < 50:
        raise ValueError(f"too few samples (train {len(Xtr)}, val {len(Xva)}); need more history before {train_end.date()}")
    x_mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); x_std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-9     # TRAIN-only scalers
    y_std = Ytr.std(0) + 1e-9
    f = lambda X: ((X - x_mean) / x_std).astype(np.float32)
    net = build_network(hp, Xtr.shape[-1], Ytr.shape[-1])
    cb = [k.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss")]
    net.fit(f(Xtr), (Ytr / y_std).astype(np.float32), validation_data=(f(Xva), (Yva / y_std).astype(np.float32)),
            epochs=hp.epochs, batch_size=hp.batch, shuffle=True, verbose=verbose, callbacks=cb)      # shuffle only INSIDE the train split
    pva = net.predict(f(Xva), verbose=0) * y_std
    resid = (pva - Yva).std(0)
    metrics = {"val": evaluate(pva, Yva), "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "val_start": str(t_val.date()),
               "train_end": str(train_end.date())}
    if t_test is not None:
        Xte, Yte, _, _ = _samples(prep, hp.seq_len, t_test, None, train_end, hp.stride)
        metrics["test"] = evaluate(net.predict(f(Xte), verbose=0) * y_std, Yte); metrics["test_start"] = str(t_test.date())
    return Predictor(net, hp, x_mean, x_std, y_std, resid, train_end, metrics, name)


def val_loss_for_ga(close, train_end, hp: HParams, seed=0) -> float:
    """GA fitness: validation MSE (scaled) of a short training run."""
    p = train_pooled(close, train_end, hp, seed=seed)
    return float(np.mean([(1 - s) ** 2 for s in p.metrics["val"]["skill_vs_naive"].values()]))     # lower = better skill
