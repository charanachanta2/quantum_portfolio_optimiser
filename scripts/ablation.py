"""Ablation of error-mitigation techniques (deliverable (b)).
Instances = QUBOs from random historical dates.  For each configuration we report,
averaged over instances x repeats (finite shots):
  est_err      |E_estimated - E_ideal(params)|   (energy-estimation accuracy)
  gap_norm     (E_decoded - E_opt)/(E_mean - E_opt)   (solution quality; 0 = optimal)
  hit_opt      fraction of runs whose decoded bitstring is the global optimum
  p_opt_model  exact probability of sampling the optimum under the noise model
  N99          shots to see the optimum w.p. 0.99 given p_opt_model
  overhead     sampling overhead factor of the mitigation (variance multiplier)"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo.config import Config
from qpo.data import synthetic_market
from qpo.backtest import make_instance, noise_models
from qpo.qaoa import QAOA, train_qaoa, estimate_energy, run_and_decode
from qpo import qec, theory


def sample_instances(market, cfg, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(cfg.lookback + 1, len(market.returns) - 30, size=k)
    N = market.returns.shape[1]
    out = []
    for i in idx:
        w = rng.dirichlet(np.ones(N))
        out.append(make_instance(market, int(i), w, cfg))
    return out


def run(cfg: Config, market, n_inst=6, repeats=10, logical_q=None, out="results", verbose=True):
    rng = np.random.default_rng(cfg.seed)
    if logical_q is None:
        model = qec.fit_logical_model(qec.sweep())
        logical_q = qec.logical_q_per_layer(model, cfg.logical_distance, cfg.logical_p_phys)
    nm = noise_models(cfg, logical_q)
    raw, dd, lg = nm["raw"], nm["mitigated"], nm["logical"]
    configs = [
        ("ideal", None, "raw", False, None),
        ("raw", raw, "raw", False, None),
        ("DD", dd, "raw", False, None),
        ("ZNE", raw, "zne", False, None),
        ("DD+ZNE", dd, "zne", False, None),
        ("PEC", raw, "pec", True, None),
        ("DD+PEC", dd, "pec", True, None),
        ("raw (noise-aware training)", raw, "raw", False, "aware"),
        ("logical (d=%d)" % cfg.logical_distance, lg, "raw", False, None),
    ]
    rows = []
    for k, inst in enumerate(sample_instances(market, cfg, n_inst, cfg.seed)):
        E = inst.qubo.energies()
        q = QAOA(E, inst.qubo.n, inst.qubo.n_couplings(), inst.qubo.n_linear())
        ideal = train_qaoa(q, None, p_max=cfg.p_max, maxiter=cfg.opt_maxiter, tol=cfg.adapt_tol, rng=rng)
        e_ideal = q.energy_exact(ideal.params)
        iopt = int(np.argmin(E))
        for name, noise, est, mit_dec, train in configs:
            params = ideal.params
            if train == "aware":
                params = train_qaoa(q, noise, p_max=cfg.p_max, maxiter=cfg.opt_maxiter, tol=cfg.adapt_tol, rng=rng, method="raw").params
            p_layers = len(params) // 2
            pr = q.probs(params) if noise is None else q.noisy_probs(params, noise)
            p_opt = float(pr[iopt])
            errs, gaps, hits, ovh = [], [], [], 1.0
            for _ in range(repeats):
                if noise is None:
                    e_est = float(E[rng.choice(len(E), cfg.shots, p=pr)].mean())
                else:
                    e_est, ovh = estimate_energy(q, params, noise, shots=cfg.shots, rng=rng, method=est, zne_scales=cfg.zne_scales)
                idx, _ = run_and_decode(q, params, noise, shots=cfg.shots, rng=rng, mitigate_readout=mit_dec)
                errs.append(abs(e_est - q.energy_exact(params)))
                gaps.append((E[idx] - E.min()) / max(E.mean() - E.min(), 1e-12))
                hits.append(idx == iopt)
            rows.append({"instance": k, "config": name, "est_err": np.mean(errs), "gap_norm": np.mean(gaps),
                         "hit_opt": np.mean(hits), "p_opt_model": p_opt,
                         "N99": theory.shots_to_success(p_opt), "overhead": ovh,
                         "q_eff": 0.0 if noise is None else q.q_eff(p_layers, noise), "p_layers": p_layers})
        if verbose:
            print(f"  ablation instance {k + 1}/{n_inst} done", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("config", sort=False).agg(est_err=("est_err", "mean"), est_err_sd=("est_err", "std"),
                                               gap_norm=("gap_norm", "mean"), hit_opt=("hit_opt", "mean"),
                                               p_opt_model=("p_opt_model", "mean"), N99=("N99", "median"),
                                               overhead=("overhead", "mean"), q_eff=("q_eff", "mean"))
    os.makedirs(out, exist_ok=True)
    df.to_csv(f"{out}/ablation_raw.csv", index=False)
    agg.to_csv(f"{out}/ablation_summary.csv")
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    cfg = Config()
    print(run(cfg, synthetic_market(n_assets=12), a.instances, a.repeats).round(4).to_string())
