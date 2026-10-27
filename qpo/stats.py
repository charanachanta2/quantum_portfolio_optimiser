import numpy as np
import pandas as pd
from scipy import stats


def sharpe(r, periods=252):
    r = np.asarray(r)
    return float(np.sqrt(periods) * r.mean() / (r.std(ddof=1) + 1e-12))


def max_drawdown(r):
    c = np.cumprod(1 + np.asarray(r))
    return float((c / np.maximum.accumulate(c) - 1).min())


def summary(ret: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for m in ret:
        r = ret[m].values
        out[m] = {"ann_return": float((1 + r).prod() ** (252 / len(r)) - 1), "ann_vol": float(r.std(ddof=1) * np.sqrt(252)),
                  "sharpe": sharpe(r), "max_drawdown": max_drawdown(r), "cum_return": float((1 + r).prod() - 1)}
    return pd.DataFrame(out).T


def stationary_bootstrap_idx(n, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def paired_sharpe_test(r1, r2, n_boot=2000, mean_block=10, seed=0):
    """H0: Sharpe(r1) == Sharpe(r2).  Paired stationary bootstrap (Politis-Romano);
    returns difference, 95 % CI, two-sided p-value (centered bootstrap)."""
    rng = np.random.default_rng(seed)
    r1, r2 = np.asarray(r1), np.asarray(r2)
    n = len(r1)
    d0 = sharpe(r1) - sharpe(r2)
    ds = np.empty(n_boot)
    for b in range(n_boot):
        ix = stationary_bootstrap_idx(n, mean_block, rng)
        ds[b] = sharpe(r1[ix]) - sharpe(r2[ix])
    p = float(np.mean(np.abs(ds - ds.mean()) >= abs(d0)))
    return {"diff": d0, "ci_low": float(np.percentile(ds, 2.5)), "ci_high": float(np.percentile(ds, 97.5)), "p_value": p}


def wilcoxon_periods(r1: pd.Series, r2: pd.Series, h=21):
    """Wilcoxon signed-rank on per-rebalance-period compounded returns."""
    a = (1 + r1).groupby(np.arange(len(r1)) // h).prod() - 1
    b = (1 + r2).groupby(np.arange(len(r2)) // h).prod() - 1
    d = (a - b).values
    if np.allclose(d, 0):
        return {"stat": 0.0, "p_value": 1.0, "n": len(d)}
    st = stats.wilcoxon(d)
    return {"stat": float(st.statistic), "p_value": float(st.pvalue), "n": len(d)}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for k, (name, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p))
        out[name] = run
    return out
