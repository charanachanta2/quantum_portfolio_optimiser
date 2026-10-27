"""OPTIONAL (item 7): D-Wave Ocean SDK simulated annealing (dimod + neal) on the project's QUBOs, compared with the
exact optimum and with our own SA.   pip install dwave-ocean-sdk    python scripts/optional_ocean_sa.py [--real]
STATUS: the QUBO->dimod conversion is unit-tested (tests/test_ising.py); the dimod/neal calls below were NOT run in
the build sandbox (package unavailable) - run it once and check that `exact_hit` is high for n<=16."""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo.config import Config
from qpo.data import synthetic_market, load_real_market
from qpo.ising import qubo_to_dict
from qpo import classical as C
from scripts.ablation import sample_instances

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true"); ap.add_argument("--esg-csv", default="data/esg.csv")
    ap.add_argument("--instances", type=int, default=10); ap.add_argument("--reads", type=int, default=100); ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    import dimod, neal                                           # ImportError here = Ocean not installed
    from qpo.universe import UNIVERSES
    cfg = Config(n_assets=4)
    market = load_real_market(UNIVERSES["extended"], "2004-06-01", "2023-12-31", a.esg_csv) if a.real else synthetic_market(n_assets=16)
    rng = np.random.default_rng(0); rows = []
    for k, inst in enumerate(sample_instances(market, cfg, a.instances, 0)):
        qb = inst.qubo; E = qb.energies(); n = qb.n
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_to_dict(qb.Q), offset=qb.offset)
        ss = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=a.reads, num_sweeps=a.sweeps, seed=k)
        e_ocean = float(ss.first.energy)
        x, e_ours = C.simulated_annealing(qb.Q, qb.offset, a.reads * a.sweeps, rng)
        rows.append({"instance": k, "n": n, "E_opt": E.min(), "E_ocean": e_ocean, "E_ours": e_ours,
                     "ocean_hit": abs(e_ocean - E.min()) < 1e-9, "ours_hit": abs(e_ours - E.min()) < 1e-9,
                     "ocean_read_hit_frac": float(np.mean(np.abs(ss.record.energy - E.min()) < 1e-9))})
    df = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True); df.to_csv(f"{a.out}/ocean_sa.csv", index=False)
    print(df.round(4).to_string()); print("hit rates  ocean:", df.ocean_hit.mean(), " ours:", df.ours_hit.mean())

if __name__ == "__main__":
    main()
