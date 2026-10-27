"""Empirical scaling of shots/evaluations-to-optimum vs problem size n (qubits).
Compares: random sampling (2^n), QAOA (p fixed, ideal), noisy QAOA lower bound, SA,
Grover-type reference.  This is EVIDENCE about scaling trends on n<=16, not a proof."""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from qpo.config import Config
from qpo.data import synthetic_market
from qpo.backtest import make_instance
from qpo.qaoa import QAOA, train_qaoa
from qpo import classical as C, theory

SIZES = [(3, 1), (4, 1), (5, 1), (3, 2), (7, 1), (4, 2)]   # (assets, periods) -> n = 2*A*T qubits


def run(cfg: Config, market, inst_per_size=4, sa_seeds=5, q_noise=0.02, out="results", verbose=True):
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for A, T in SIZES:
        c = replace(cfg, n_assets=A, n_periods=T)
        for r in range(inst_per_size):
            i = int(rng.integers(c.lookback + 1, len(market.returns) - 30))
            w = rng.dirichlet(np.ones(market.returns.shape[1]))
            inst = make_instance(market, i, w, c)
            E = inst.qubo.energies(); n = inst.qubo.n
            q = QAOA(E, n, inst.qubo.n_couplings(), inst.qubo.n_linear())
            tr = train_qaoa(q, None, p_max=3, maxiter=cfg.opt_maxiter, tol=0.0, rng=rng)
            p_opt = float(q.probs(tr.params)[int(np.argmin(E))])
            sa = [C.evals_to_optimum_sa(inst.qubo.Q, inst.qubo.offset, E.min(), rng, max_evals=100000) for _ in range(sa_seeds)]
            rows.append({"n": n, "p_layers": tr.p, "p_opt_ideal": p_opt, "N99_qaoa": theory.shots_to_success(p_opt),
                         "N99_qaoa_noisy_LB": theory.shots_to_success(theory.p_opt_noisy_lower_bound(p_opt, n, q_noise)),
                         "N99_random": theory.shots_to_success(2.0 ** -n), "SA_evals_median": float(np.median(sa)),
                         "grover_queries": theory.grover_queries(n)})
        if verbose:
            print(f"  scaling n={2 * A * T} done", flush=True)
    df = pd.DataFrame(rows)
    g = df.groupby("n").median(numeric_only=True)
    fits = {}
    for col in ["N99_qaoa", "N99_qaoa_noisy_LB", "N99_random", "SA_evals_median", "grover_queries"]:
        slope, icpt = np.polyfit(g.index.values, np.log2(g[col].values), 1)
        fits[col] = float(slope)          # fitted exponent: cost ~ 2^(slope * n)
    os.makedirs(out, exist_ok=True)
    g.to_csv(f"{out}/scaling.csv")
    pd.Series(fits).to_csv(f"{out}/scaling_fits.csv", header=["log2_slope_per_qubit"])
    return g, fits


if __name__ == "__main__":
    g, f = run(Config(), synthetic_market(n_assets=12))
    print(g.round(2).to_string()); print(f)
