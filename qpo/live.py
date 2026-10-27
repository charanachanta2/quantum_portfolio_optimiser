"""Streaming / live re-optimisation engine (used by the dashboard and tests).
Consumes daily bars tick-by-tick - either replayed from a Market (default) or pulled from a feed via the `next_bar()` hook
(qpo.feed.YahooFeed for live data, qpo.feed.ReplayFeed for offline) - and every `rebalance_days` bars re-builds the
time-dependent QUBO from the data seen SO FAR and re-solves it.  `solve_now` makes a single as-of-today decision."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .backtest import make_instance, solve_instance, noise_models
from .config import Config
from .data import Market
from .feed import append_bar, append_placeholder


class LiveOptimizer:
    def __init__(self, market: Market, cfg: Config, start: str | None, methods, logical_q=1e-6, feed=None, forecaster=None):
        self.m, self.cfg, self.methods = market, cfg, list(methods)
        self.feed, self.forecaster = feed, forecaster
        # replay mode: start at `start`; feed mode: history is everything already in `market`, new bars arrive from the feed
        self.i = len(market.returns) if feed is not None else max(int(np.searchsorted(market.returns.index, pd.Timestamp(start))), cfg.lookback)
        N = market.returns.shape[1]
        self.hold = {k: np.full(N, 1.0 / N) for k in self.methods}
        self.target = {k: self.hold[k].copy() for k in self.methods}
        self.nm = noise_models(cfg, logical_q)
        self.rng = np.random.default_rng(cfg.seed)
        self.pnl = {k: [1.0] for k in self.methods}
        self.dates = [market.returns.index[self.i - 1]]
        self.log = []
        self.since = cfg.rebalance_days   # force optimisation on first tick
        self.last_info = {}
        self.agent = None

    def _reoptimise(self):
        R = self.m.returns.values
        for k in self.methods:
            inst = make_instance(self.m, self.i, self.hold[k], self.cfg, self.forecaster)
            E = inst.qubo.energies()
            w, info = solve_instance(k, inst, E, self.cfg, self.nm, self.rng, None, {}, self.hold[k],
                                     R[self.i - self.cfg.lookback:self.i], self.agent)
            cost = self.cfg.tc_bps / 1e4 * np.abs(w - self.hold[k]).sum()
            self.pnl[k][-1] *= (1 - cost)
            self.hold[k] = w
            self.target[k] = w
            self.last_info[k] = {**info, "stress_cap": inst.caps[0], "lambda0": inst.lam[0]}
            self.log.append({"date": self.m.returns.index[self.i], "method": k, "cost": cost, **info})

    def step(self) -> dict:
        """Consume one new bar. Returns dict(date, reoptimised: bool)."""
        if self.feed is not None:
            bar = self.feed.next_bar()
            if bar is None:
                return {"done": False, "waiting": True, "reoptimised": False}
            self.m = append_bar(self.m, bar)
        if self.i >= len(self.m.returns):
            return {"done": True}
        reopt = self.since >= self.cfg.rebalance_days
        if reopt:
            self._reoptimise()
            self.since = 0
        r = self.m.returns.values[self.i]
        for k in self.methods:
            w = self.hold[k]
            pr = float(w @ r)
            self.pnl[k].append(self.pnl[k][-1] * (1 + pr))
            self.hold[k] = w * (1 + r) / (1 + pr)
        self.dates.append(self.m.returns.index[self.i])
        self.i += 1
        self.since += 1
        return {"done": False, "date": self.dates[-1], "reoptimised": reopt}


def solve_now(market: Market, cfg: Config, methods, w_hold: np.ndarray | None = None, logical_q: float = 1e-6, forecaster=None,
              seed: int | None = None) -> dict:
    """One as-of-today decision from the data in `market` (all bars strictly before `date`): returns weights, info, timings per method.
    Used by the dashboard's background worker."""
    import time
    N = market.returns.shape[1]
    w_hold = np.full(N, 1.0 / N) if w_hold is None else w_hold
    date = market.returns.index[-1] + pd.offsets.BDay(1)
    m = append_placeholder(market, date)
    i = len(m.returns) - 1
    R = m.returns.values
    nm = noise_models(cfg, logical_q)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    inst0 = make_instance(m, i, w_hold, cfg, forecaster)
    E = inst0.qubo.energies()
    out = {"date": date, "tickers": list(m.returns.columns), "sub": [m.returns.columns[k] for k in inst0.sub], "weights": {}, "info": {},
           "seconds": {}, "n_qubits": inst0.qubo.n, "mu_sub": inst0.mu.tolist(), "lam": inst0.lam, "caps": inst0.caps,
           "elig": [m.returns.columns[k] for k in inst0.elig], "mu_elig": inst0.mu_elig, "Sigma_elig": inst0.Sigma_elig,
           "n_zz": inst0.qubo.n_couplings(), "n_lin": inst0.qubo.n_linear(), "mu_source": cfg.mu_source, "risk_source": cfg.risk_source}
    cache = {}
    for k in methods:
        t0 = time.time()
        w, info = solve_instance(k, inst0, E, cfg, nm, rng, None, cache, w_hold, R[i - cfg.lookback:i], None)
        out["weights"][k] = w
        out["info"][k] = info
        out["seconds"][k] = time.time() - t0
        out["info"][k]["turnover"] = float(np.abs(w - w_hold).sum())
    return out
