"""STAGE `resources`: logical-qubit resource estimation + sensitivity to every modelling constant.

Writes (data-independent: none of this depends on market data):
  resources.csv           grid over (assets, periods, QAOA layers p, p_phys) at baseline constants
  resource_assumptions.csv/.md   the assumptions table (value, swept range, source, confidence)
  resource_sweep.csv      one-at-a-time sweep for a small (n=16) and a large (n=120) circuit
  resource_corners.csv    all-optimistic / baseline / all-pessimistic corners
  resource_tornado.png    sensitivity plot
"""
import argparse, copy, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo import qec

BASE = dict(p_phys=1e-3, eps_total=0.05, cycle_time_us=1.0, n_factories=4, factory_footprint_d2=72.0,
            t_factory_cycles_per_T=5.5, routing_overhead=1.5, synth_a=3.0, synth_b=0.0, cnot_parallel_frac=1.0,
            overlap_T_and_clifford=True, level2_footprint_mult=3.0, level2_time_mult=2.0)
CIRCUITS = {"small n=16 (A=4,T=2,p=3)": (4, 2, 3), "large n=120 (A=20,T=3,p=3)": (20, 3, 3)}

# name -> (values swept, baseline, source / justification, confidence)
ASSUMPTIONS = [
    ("p_phys", [3e-3, 1e-3, 3e-4, 1e-4], "physical error rate (circuit-level depolarising, Stim model)",
     "Design variable. Fit data only cover p=2e-3..6e-3, so p<=1e-3 is an EXTRAPOLATION.", "low (extrapolated)"),
    ("synth", [("RUS 1.15*log2(1/e)", 1.15, 0.0), ("RS 3*log2(1/e) [base]", 3.0, 0.0), ("RS 3*log2+5", 3.0, 5.0),
               ("RS 4*log2(1/e)", 4.0, 0.0)], "T gates per RZ: a*log2(1/eps)+b",
     "Ross-Selinger arXiv:1403.2975 (3 log2 + O(loglog), ancilla-free); RUS slope 1.15: Bocharov et al. PRL 114, 080502. "
     "The '+9.2' of the first draft was not verifiable.", "medium"),
    ("routing_overhead", [1.0, 1.5, 2.0, 3.0], "data-patch area multiplier for lattice-surgery routing",
     "Typical layouts use ~1.5-2x (Litinski, Game of Surface Codes); dense all-to-all ZZ couplings are routing-heavy.", "medium"),
    ("n_factories", [1, 2, 4, 8, 16], "parallel T factories", "Design choice; trades qubits for time.", "n/a (design)"),
    ("factory_footprint_d2", [36.0, 72.0, 150.0], "factory area in units of d^2 physical qubits",
     "Order of 10^1-10^2 d^2 for 15-to-1 factories (Litinski, Magic state distillation: not as costly as you think). Not re-verified.", "low"),
    ("t_factory_cycles_per_T", [3.0, 5.5, 11.0], "logical steps (d rounds each) per T state per factory",
     "Order-of-magnitude from the same references; layout dependent.", "low"),
    ("cycle_time_us", [0.5, 1.0, 2.0], "QEC round time (us)", "Superconducting-like ~1 us; trapped ions/atoms are 10-1000x slower.", "medium (platform)"),
    ("cnot_parallel_frac", [0.25, 0.5, 1.0], "fraction of the n/2 ideal parallel CNOTs that routing allows",
     "Dense couplings serialise on a 2D layout; 1.0 is the optimistic end.", "low"),
    ("overlap_T_and_clifford", [True, False], "T-state supply overlaps Clifford steps?",
     "True is standard in game-of-surface-codes style schedules; False is the sequential upper bound.", "medium"),
    ("eps_total", [0.01, 0.05, 0.2], "total failure budget per circuit execution (split in 3)", "Design variable.", "n/a (design)"),
    ("level2_footprint_mult", [2.0, 3.0, 6.0], "area multiplier if 2-level distillation is needed", "Assumption (only used when needed).", "low"),
    ("level2_time_mult", [1.5, 2.0, 3.0], "T-supply time multiplier if 2-level distillation is needed", "Assumption (only used when needed).", "low"),
    ("model_A", [0.5, 1.0, 2.0], "multiplier on fitted prefactor A of p_L=A(p/p_th)^((d+1)/2)",
     "Fit uncertainty: 9 points from Stim; exponent fixed to 1.", "medium"),
    ("model_pth", [0.8, 1.0, 1.25], "multiplier on fitted threshold p_th", "Fit uncertainty (Stim threshold ~0.7-1% for this noise model).", "medium"),
]


def run_one(model, A, T, p, **kw):
    n = A * 2 * T                                   # K=2 bits/asset/period
    n_zz = n * (n - 1) // 2 * 3 // 4                # ~75 % dense couplings
    return qec.estimate_resources(n, n_zz, n, p, model, **kw)


def apply(model, kw, name, val):
    model, kw = dict(model), dict(kw)
    if name == "synth":
        kw["synth_a"], kw["synth_b"] = val[1], val[2]
    elif name == "model_A":
        model["A"] = model["A"] * val
    elif name == "model_pth":
        model["p_th"] = model["p_th"] * val
    else:
        kw[name] = val
    return model, kw


def label(name, val):
    return val[0] if name == "synth" else val


def main(out="results"):
    os.makedirs(out, exist_ok=True)
    model = json.load(open(f"{out}/qec_fit.json"))
    # ---- grid at baseline constants (the original resources.csv, regenerated) ----
    rows = []
    for A, T in [(3, 1), (4, 1), (4, 2), (6, 2), (8, 2), (12, 2), (20, 3)]:
        for p in (1, 3, 5):
            for pp in (1e-3, 1e-4):
                r = run_one(model, A, T, p, **{**BASE, "p_phys": pp}); r.update({"assets": A, "periods": T, "p_phys": pp})
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/resources.csv", index=False)
    # ---- assumptions table ----
    tab = pd.DataFrame([{"parameter": n, "baseline": (BASE.get(n) if n not in ("synth", "model_A", "model_pth") else
                         {"synth": "RS 3*log2(1/e)", "model_A": 1.0, "model_pth": 1.0}[n]),
                         "swept": ", ".join(str(label(n, v)) for v in vals), "meaning": mean, "source / justification": src,
                         "confidence": conf} for n, vals, mean, src, conf in ASSUMPTIONS])
    tab.to_csv(f"{out}/resource_assumptions.csv", index=False)
    open(f"{out}/resource_assumptions.md", "w").write(tab.to_markdown(index=False))
    # ---- one-at-a-time sweep ----
    sw = []
    for cname, (A, T, p) in CIRCUITS.items():
        for name, vals, *_ in ASSUMPTIONS:
            for v in vals:
                m2, kw2 = apply(model, BASE, name, v)
                r = run_one(m2, A, T, p, **kw2)
                sw.append({"circuit": cname, "parameter": name, "value": label(name, v), "distance": r.get("distance"),
                           "physical_qubits": r.get("physical_qubits"), "time_per_shot_ms": r.get("time_per_shot_ms"),
                           "distill_levels": r.get("distill_levels"), "T_count": r.get("T_count"), "note": r.get("note", "")})
    swd = pd.DataFrame(sw); swd.to_csv(f"{out}/resource_sweep.csv", index=False)
    # ---- corners ----
    opt = dict(BASE, synth_a=1.15, routing_overhead=1.0, n_factories=16, factory_footprint_d2=36.0, t_factory_cycles_per_T=3.0,
               cycle_time_us=0.5, cnot_parallel_frac=1.0, overlap_T_and_clifford=True, level2_footprint_mult=2.0, level2_time_mult=1.5)
    pes = dict(BASE, synth_a=3.0, synth_b=5.0, routing_overhead=3.0, n_factories=1, factory_footprint_d2=150.0, t_factory_cycles_per_T=11.0,
               cycle_time_us=2.0, cnot_parallel_frac=0.25, overlap_T_and_clifford=False, level2_footprint_mult=6.0, level2_time_mult=3.0)
    cor = []
    for cname, (A, T, p) in CIRCUITS.items():
        for pp in (1e-3, 1e-4):
            for tag, kw, mA, mp in [("optimistic", opt, 0.5, 1.25), ("baseline", BASE, 1.0, 1.0), ("pessimistic", pes, 2.0, 0.8)]:
                m2 = dict(model); m2["A"] *= mA; m2["p_th"] *= mp
                r = run_one(m2, A, T, p, **{**kw, "p_phys": pp})
                cor.append({"circuit": cname, "p_phys": pp, "scenario": tag, "distance": r.get("distance"),
                            "physical_qubits": r.get("physical_qubits"), "time_per_shot_ms": r.get("time_per_shot_ms"),
                            "distill_levels": r.get("distill_levels"), "note": r.get("note", "")})
    pd.DataFrame(cor).to_csv(f"{out}/resource_corners.csv", index=False)
    # ---- tornado: qubits (top) and time per shot (bottom); grey = design variables, teal = modelling assumptions ----
    design = {"p_phys", "n_factories", "eps_total"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for j, cname in enumerate(CIRCUITS):
        sub = swd[swd.circuit == cname]
        for i, (met, lab) in enumerate([("physical_qubits", "physical qubits"), ("time_per_shot_ms", "time per shot")]):
            ax = axes[i, j]
            base = sub[(sub.parameter == "eps_total") & (sub.value == 0.05)][met].iloc[0]
            spans = []
            for name, g in sub.groupby("parameter", sort=False):
                g = g.dropna(subset=[met])
                spans.append((name, g[met].min() / base, g[met].max() / base))
            spans.sort(key=lambda x: x[2] / x[1])
            for k, (name, lo, hi) in enumerate(spans):
                ax.barh(k, np.log10(hi) - np.log10(lo), left=np.log10(lo), color="#8A94A6" if name in design else "#139A8C")
            ax.axvline(0, color="k", lw=.8); ax.set_yticks(range(len(spans))); ax.set_yticklabels([x[0] for x in spans], fontsize=7.5)
            ax.set_xlabel(f"log10( {lab} / baseline )", fontsize=8)
            fmt = f"{int(base):,} qubits" if met == "physical_qubits" else f"{base:,.1f} ms/shot"
            ax.set_title(f"{cname}\nbaseline {fmt}", fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{out}/resource_tornado.png", dpi=150); plt.close(fig)
    print(pd.DataFrame(cor).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); main(ap.parse_args().out)
