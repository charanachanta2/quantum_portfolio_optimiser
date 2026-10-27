"""Walk-forward (receding-horizon) backtest.  No look-ahead: every decision at
date t uses returns/macro strictly before t.  Costs: linear bps on traded notional."""
from __future__ import annotations
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import classical as C
from . import miqp as MQ
from .config import Config
from .data import Market
from .noise import NoiseModel
from .qaoa import QAOA, train_qaoa, run_and_decode
from .qubo import Qubo, build_qubo, repair_weights

QUANTUM = ["qaoa_ideal", "qaoa_noisy_raw", "qaoa_noisy_mitigated", "qaoa_logical"]
_HAS_GUROBI = MQ.has_gurobi()
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
# miqp_sub always runs (Gurobi if present, else the exact DP - identical optimum); the others need their package.
CLASSICAL = (["equal_weight_full", "cvar_full", "cvar_sub"] + (["cvar_gurobi_sub"] if _HAS_GUROBI else []) +
             ["miqp_sub"] + (["miqp_gurobi_full"] if _HAS_GUROBI else []) +
             ["rl_es_full"] + (["rl_torch_full"] if _HAS_TORCH else []) + ["exact_qubo", "sa_qubo"])
ALL_METHODS = CLASSICAL + QUANTUM
OPTIONAL_METHODS = {"cvar_gurobi_sub": "gurobipy", "miqp_gurobi_full": "gurobipy", "rl_torch_full": "torch"}
ALL_POSSIBLE = ["equal_weight_full", "cvar_full", "cvar_sub", "cvar_gurobi_sub", "miqp_sub", "miqp_gurobi_full",
                "rl_es_full", "rl_torch_full", "exact_qubo", "sa_qubo"] + QUANTUM


@dataclass
class Instance:
    date: pd.Timestamp
    sub: np.ndarray            # indices (into full universe) of encoded assets
    elig: np.ndarray           # indices of ESG-eligible assets
    qubo: Qubo
    mu: np.ndarray
    Sigma: np.ndarray
    w_prev_sub: np.ndarray
    lam: list
    caps: list
    esg_sub: np.ndarray
    mu_elig: np.ndarray = None
    Sigma_elig: np.ndarray = None
    esg_elig: np.ndarray = None


def _shrink_cov(R, delta=0.2):
    S = np.cov(R.T)
    return (1 - delta) * S + delta * np.diag(np.diag(S))


def select_universe(window: np.ndarray, elig: np.ndarray, A: int) -> np.ndarray:
    """Greedy: best trailing Sharpe, then penalise correlation with already-chosen."""
    if len(elig) <= A:
        return elig
    sub = window[:, elig]
    sr = sub.mean(0) / (sub.std(0) + 1e-9)
    corr = np.corrcoef(sub.T)
    chosen = [int(np.argmax(sr))]
    while len(chosen) < A:
        score = sr - 3.0 * corr[:, chosen].mean(1) * sr.std()
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return elig[chosen]


def vix_z(market: Market, i: int, look=252) -> float:
    v = market.macro["VIX"].values[max(0, i - look):i]
    return float((v[-1] - v.mean()) / (v.std() + 1e-9))


def _lstm_mu(market: Market, i: int, elig: np.ndarray, hist_elig: np.ndarray, cfg: Config, forecaster):
    """Per-eligible-asset expected return over the rebalance horizon from the prediction system (F2).
    `forecaster(date, tickers)` returns 21-trading-day simple returns using only data before `date`.  NaN -> historical value."""
    if forecaster is None:
        raise ValueError(f"cfg.mu_source={cfg.mu_source!r} needs a forecaster (see stock_prediction.forecast.LSTMForecaster)")
    tick = [market.returns.columns[k] for k in elig]
    hat = np.asarray(forecaster(market.returns.index[i], tick), dtype=float) * (cfg.rebalance_days / 21.0)
    hat = np.where(np.isnan(hat), hist_elig, hat)
    return hat if cfg.mu_source == "lstm" else cfg.lstm_blend * hat + (1 - cfg.lstm_blend) * hist_elig


def make_instance(market: Market, i: int, w_hold: np.ndarray, cfg: Config, forecaster=None) -> Instance:
    R = market.returns.values
    window = R[i - cfg.lookback:i]
    elig = np.where(market.esg.values >= cfg.esg_min)[0]                # HARD ESG filter
    sub = select_universe(window, elig, cfg.n_assets)
    h = cfg.rebalance_days
    all_mu = window[:, elig].mean()
    mu = (0.5 * window[:, sub].mean(0) + 0.5 * all_mu) * h               # shrunk expected return
    Sigma = _shrink_cov(window[:, sub]) * h
    mu_e = (0.5 * window[:, elig].mean(0) + 0.5 * all_mu) * h               # same estimator on the whole eligible set
    if cfg.mu_source != "historical":                                    # LSTM / blend replaces mu for the QUBO AND the full-universe MIQP
        mu_e = _lstm_mu(market, i, elig, mu_e, cfg, forecaster)
        pos = {int(k): j for j, k in enumerate(elig)}
        mu = mu_e[[pos[int(k)] for k in sub]]
    z = vix_z(market, i)
    lam0 = cfg.lambda0 * (1 + cfg.kappa_vix * float(np.clip(z, -1, 3)))
    if cfg.risk_source == "vix+fuzzy":                                   # F3: fuzzy risk score (vol, VIX z, forecast confidence)
        from stock_prediction.fuzzy import fuzzy_risk
        vol = float(window[-20:, sub].mean(1).std() * np.sqrt(252))
        conf = float(forecaster.confidence(market.returns.index[i])) if hasattr(forecaster, "confidence") else 0.25
        r, _ = fuzzy_risk(vol, z, conf)
        lam0 = max(cfg.lambda0 * (1 + cfg.kappa_vix * float(np.clip(z, -1, 3)) + cfg.kappa_risk * (2 * r - 1)), 0.25 * cfg.lambda0)
    lam = [lam0 + (0.5 * (cfg.lambda0 - lam0)) * t for t in range(cfg.n_periods)]   # mean-reverting
    stress = z > cfg.stress_z
    caps = [cfg.cap_levels_stress if (stress and t == 0) else cfg.cap_levels_calm
            for t in range(cfg.n_periods)]
    esg_sub = market.esg.values[sub]
    qb = build_qubo(mu, Sigma, w_hold[sub], lam, caps, cfg, esg=esg_sub)
    Sig_e = _shrink_cov(window[:, elig]) * h
    return Instance(market.returns.index[i], sub, elig, qb, mu, Sigma, w_hold[sub], lam, caps, esg_sub,
                    mu_e, Sig_e, market.esg.values[elig])


def noise_models(cfg: Config, logical_q: float):
    return {
        "raw": NoiseModel(cfg.p2, cfg.p1, cfg.p_readout, cfg.idle_rate, 0.0, None, "raw"),
        "mitigated": NoiseModel(cfg.p2, cfg.p1, cfg.p_readout, cfg.idle_rate, cfg.dd_suppression, None, "dd"),
        "logical": NoiseModel(logical_q_per_layer=logical_q, name="logical"),
    }


def solve_quantum(method: str, inst: Instance, E: np.ndarray, cfg: Config, nm: dict, rng, cache: dict):
    qb = inst.qubo
    q = QAOA(E, qb.n, qb.n_couplings(), qb.n_linear())
    if "ideal_train" not in cache:     # train once per instance; deployed under each noise model
        cache["ideal_train"] = train_qaoa(q, None, p_max=cfg.p_max, maxiter=cfg.opt_maxiter, tol=cfg.adapt_tol, rng=rng)
    tr = cache["ideal_train"]
    if method == "qaoa_ideal":
        idx, es = run_and_decode(q, tr.params, None, shots=cfg.shots, rng=rng)
    elif method == "qaoa_noisy_raw":
        idx, es = run_and_decode(q, tr.params, nm["raw"], shots=cfg.shots, rng=rng)
    elif method == "qaoa_noisy_mitigated":
        idx, es = run_and_decode(q, tr.params, nm["mitigated"], shots=cfg.shots, rng=rng, mitigate_readout=True)
    elif method == "qaoa_logical":
        idx, es = run_and_decode(q, tr.params, nm["logical"], shots=cfg.shots, rng=rng)
    else:
        raise ValueError(method)
    x = (idx >> np.arange(qb.n)) & 1
    pr = q.probs(tr.params)
    return x, {"p": tr.p, "p_opt_ideal": float(pr[int(np.argmin(E))])}


def solve_instance(method, inst, E, cfg, nm, rng, ctx, cache, w_hold, R_window, agent):
    N = len(w_hold)
    sub = inst.sub
    full = np.zeros(N)
    info = {}
    if method == "equal_weight_full":
        full[inst.elig] = 1.0 / len(inst.elig)
    elif method in ("cvar_full", "cvar_sub", "cvar_gurobi_sub"):
        idx = inst.elig if method == "cvar_full" else sub
        wp = w_hold[idx]
        wp = wp / wp.sum() if wp.sum() > 1e-9 else np.full(len(idx), 1 / len(idx))
        cap = max(cfg.cap_levels_calm / cfg.units, 1.0 / len(idx))        # same 50 % cap as the QUBO
        solver = C.cvar_gurobi if method == "cvar_gurobi_sub" else C.cvar_scenario_lp
        full[idx] = solver(R_window[:, idx], wp, alpha=cfg.cvar_alpha, tc=cfg.tc_bps / 1e4,
                           rho=cfg.cvar_rho, cap=cap)
    elif method in ("rl_es_full", "rl_torch_full"):
        idx = inst.elig
        wp = w_hold[idx]
        wp = wp / wp.sum() if wp.sum() > 1e-9 else np.full(len(idx), 1 / len(idx))
        ag = agent[method] if isinstance(agent, dict) else agent
        full[idx] = ag.act(R_window[-126:, idx], wp)
    elif method in ("miqp_sub", "miqp_gurobi_full"):
        if method == "miqp_sub":
            prob = MQ.MiqpProblem(inst.mu, inst.Sigma, inst.w_prev_sub, inst.lam, inst.caps, cfg.units, cfg.tc_quad,
                                  cfg.esg_tilt, inst.esg_sub)
            w, f, info = MQ.solve(prob, "auto")
            full[sub] = w
        else:
            idx = inst.elig
            prob = MQ.MiqpProblem(inst.mu_elig, inst.Sigma_elig, w_hold[idx], inst.lam, inst.caps, cfg.units, cfg.tc_quad,
                                  cfg.esg_tilt, inst.esg_elig)
            Y, f, info = MQ.solve_gurobi(prob, cfg.miqp_time_limit)
            info = {"solver": "gurobi", **info}
            full[idx] = MQ.first_period_weights(Y, cfg.units)
        info = {"miqp_obj": float(f), **{k: v for k, v in info.items() if k in ("solver", "mip_gap")}}
    elif method in ("exact_qubo", "sa_qubo") or method in QUANTUM:
        if method == "exact_qubo":
            x = (int(np.argmin(E)) >> np.arange(inst.qubo.n)) & 1
        elif method == "sa_qubo":
            x, _ = C.simulated_annealing(inst.qubo.Q, inst.qubo.offset, cfg.shots, rng)
        else:
            x, info = solve_quantum(method, inst, E, cfg, nm, rng, cache)
        xi = np.asarray(x).astype(int)
        e_sel = float(xi.astype(float) @ inst.qubo.Q @ xi.astype(float) + inst.qubo.offset)
        info.update({"E_sel": e_sel, "E_opt": float(E.min()), "E_max": float(E.max()), "E_mean": float(E.mean()),
                     "approx_ratio": float((E.max() - e_sel) / max(E.max() - E.min(), 1e-12)),
                     "gap_norm": float((e_sel - E.min()) / max(E.mean() - E.min(), 1e-12)),
                     "is_opt": bool(e_sel <= E.min() + 1e-9)})
        full[sub] = inst.qubo.first_period_weights(xi)
    else:
        raise ValueError(method)
    return full, info


def run_backtest(market: Market, cfg: Config, start: str, end: str, methods=None,
                 logical_q: float = 1e-6, rl_train_years: int = 4, verbose: bool = True, forecaster=None):
    methods = methods or ALL_METHODS
    rng = np.random.default_rng(cfg.seed)
    R = market.returns.values
    dates = market.returns.index
    N = R.shape[1]
    i0 = int(np.searchsorted(dates, pd.Timestamp(start)))
    i1 = int(np.searchsorted(dates, pd.Timestamp(end)))
    i0 = max(i0, cfg.lookback)
    nm = noise_models(cfg, logical_q)

    agent = {}
    a = max(cfg.lookback, i0 - 252 * rl_train_years)          # RL agents train ONLY on data before the test start
    elig = np.where(market.esg.values >= cfg.esg_min)[0]
    if "rl_es_full" in methods:
        agent["rl_es_full"] = C.ESAgent(len(elig), seed=cfg.seed).fit(R[a:i0][:, elig], rebalance=cfg.rebalance_days,
                                                                      iters=25)
    if "rl_torch_full" in methods:
        from .rl_torch import TorchAgent
        agent["rl_torch_full"] = TorchAgent(hidden=cfg.rl_hidden, seed=cfg.seed).fit(
            R[a:i0][:, elig], rebalance=cfg.rebalance_days, epochs=cfg.rl_epochs)

    hold = {m: np.full(N, 1.0 / N) for m in methods}
    daily = {m: [] for m in methods}
    recs = []
    reb_idx = list(range(i0, i1, cfg.rebalance_days))
    t_start = time.time()
    for k, i in enumerate(reb_idx):
        # decision instance depends on current holdings -> build per method group (QUBO depends on w_prev)
        j_end = min(i + cfg.rebalance_days, i1)
        cache_by_hold = {}
        for m in methods:
            inst = make_instance(market, i, hold[m], cfg, forecaster)
            key = tuple(np.round(inst.w_prev_sub, 4)) + tuple(inst.sub)
            c = cache_by_hold.setdefault(key, {"E": inst.qubo.energies(), "qaoa": {}})
            t0 = time.time()
            w_tgt, info = solve_instance(m, inst, c["E"], cfg, nm, rng, None, c["qaoa"], hold[m], R[i - cfg.lookback:i], agent)
            dt = time.time() - t0
            hold_before = hold[m].copy()
            cost = cfg.tc_bps / 1e4 * np.abs(w_tgt - hold[m]).sum()
            w = w_tgt.copy()
            rets = []
            for d in range(i, j_end):
                r = float(w @ R[d])
                if d == i:
                    r -= cost
                rets.append(r)
                w = w * (1 + R[d]) / (1 + r + (cost if d == i else 0))
            hold[m] = w / w.sum()
            daily[m].extend(rets)
            recs.append({"date": dates[i], "method": m, "turnover": float(np.abs(w_tgt - hold_before).sum()),
                         "cost": cost, "seconds": dt, **info})
        if verbose and (k % 5 == 0 or k == len(reb_idx) - 1):
            print(f"  rebalance {k + 1}/{len(reb_idx)}  {dates[i].date()}  elapsed {time.time() - t_start:.0f}s", flush=True)
    idx = dates[reb_idx[0]:i1][: len(daily[methods[0]])]
    ret = pd.DataFrame({m: pd.Series(daily[m], index=idx) for m in methods})
    return ret, pd.DataFrame(recs)
