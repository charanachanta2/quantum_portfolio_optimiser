"""Surface-code memory experiments with Stim + PyMatching, a fitted logical-error
model, and a logical-qubit resource estimator for the QAOA circuits.

Everything about the *fault-tolerant* part of the project lives here:
  * `simulate_memory`   : Monte-Carlo logical error rate (per d rounds) via Stim
  * `fit_logical_model` : p_L(d,p) per round = A (p/p_th)^((d+1)/2)
  * `estimate_resources`: logical qubits, T-count, code distance, physical qubits, time
The resource formulas are standard back-of-envelope models; every constant is an
explicit argument so you can justify/sweep it in the report.
"""
from __future__ import annotations
import json
import math
import os
import numpy as np


def simulate_memory(d: int, p: float, shots: int = 20000, rounds: int | None = None,
                    basis: str = "z", seed: int = 0) -> dict:
    import stim
    import pymatching
    rounds = rounds or d
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=d, rounds=rounds,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)
    dem = circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    pred = matcher.decode_batch(det)
    fails = int(np.sum(np.any(pred != obs, axis=1)))
    pl = fails / shots
    return {"d": d, "p": p, "rounds": rounds, "shots": shots, "fails": fails,
            "p_L": pl, "p_L_per_round": 1 - (1 - 2 * pl) ** (1 / rounds) if pl < 0.5 else 0.5,
            "stderr": math.sqrt(max(pl * (1 - pl), 1e-12) / shots)}


def sweep(ds=(3, 5, 7), ps=(0.002, 0.004, 0.006), shots=20000, cache="data/cache/qec_sweep.json"):
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    out = [simulate_memory(d, p, shots, seed=17 * d) for d in ds for p in ps]
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(out, open(cache, "w"), indent=1)
    return out


def fit_logical_model(rows) -> dict:
    """Weighted least squares for  log pL = logA + (d+1)/2 * (log p - log p_th)
    (exponent fixed to 1 -> only 2 free parameters, far more stable when extrapolating
    from p ~ 2e-3..6e-3 down to p ~ 1e-3).  Weights = number of observed failures
    (1/var of log pL for Poisson counts).  Rows with <5 failures are dropped."""
    X, y, w = [], [], []
    for r in rows:
        if r["fails"] < 5 or r["p_L_per_round"] <= 0:
            continue
        m = (r["d"] + 1) / 2.0
        X.append([1.0, -m])                       # unknowns: logA, log p_th
        y.append(math.log(r["p_L_per_round"]) - m * math.log(r["p"]))
        w.append(math.sqrt(r["fails"]))
    X, y, w = np.array(X), np.array(y), np.array(w)
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    logA, log_pth = coef
    return {"logA": float(logA), "log_p_th": float(log_pth), "p_exponent": 1.0,
            "A": float(math.exp(logA)), "p_th": float(math.exp(log_pth)), "n_points": int(len(y))}


def logical_error_per_round(model: dict, d: int, p: float) -> float:
    return float(model["A"] * (p / model["p_th"]) ** (model["p_exponent"] * (d + 1) / 2))


def logical_q_per_layer(model: dict, d: int, p: float, rounds_per_layer: int | None = None) -> float:
    """Flip probability seen by a Z-basis measurement after one logical gate layer
    (lattice-surgery CNOT ~ d rounds).  Only X-type logical errors flip the outcome -> /2."""
    rounds_per_layer = rounds_per_layer or d
    return 0.5 * min(1.0, rounds_per_layer * logical_error_per_round(model, d, p))


def synth_t_count(eps: float, a: float = 3.0, b: float = 0.0) -> float:
    """T gates to synthesize one RZ(theta) to precision eps:  a*log2(1/eps) + b.
    Baseline a=3 (Ross & Selinger, arXiv:1403.2975: ancilla-free Clifford+T, 3 log2(1/eps)+O(loglog), verified).
    a=1.15 is the *expected* count of repeat-until-success circuits (Bocharov-Roetteler-Svore, PRL 114, 080502, verified
    slope; the '+9.2' used in the first draft could NOT be verified, so it is only a sweep point)."""
    return a * math.log2(1 / eps) + b


T_DISTILL_COEFF = 35.0          # 15-to-1 output error ~ 35 p^3 (Bravyi & Kitaev 2005; recalled, NOT re-verified here)


def t_state_error(p_in: float, levels: int) -> float:
    """Output infidelity of `levels` rounds of 15-to-1 distillation starting from raw error p_in."""
    e = p_in
    for _ in range(levels):
        e = T_DISTILL_COEFF * e ** 3
    return e


def estimate_resources(n_logical: int, n_zz: int, n_lin: int, p_layers: int, model: dict,
                       p_phys: float = 1e-3, eps_total: float = 0.05, cycle_time_us: float = 1.0,
                       n_factories: int = 4, factory_footprint_d2: float = 72.0,
                       t_factory_cycles_per_T: float = 5.5, routing_overhead: float = 1.5,
                       n_shots: int = 2048, synth_a: float = 3.0, synth_b: float = 0.0,
                       cnot_parallel_frac: float = 1.0, overlap_T_and_clifford: bool = True,
                       level2_footprint_mult: float = 3.0, level2_time_mult: float = 2.0,
                       distill_p_in: float | None = None) -> dict:
    """Smallest odd distance d such that the *total* failure budget eps_total is met, split in three equal parts:
        eps_total/3  logical (memory + lattice-surgery) errors of the data patches,
        eps_total/3  rotation-synthesis error,
        eps_total/3  magic-state (T) infidelity  (1 or 2 levels of 15-to-1 distillation, whichever suffices).
    ONE exposure time is used for both the failure estimate and the runtime (the first draft used two
    inconsistent ones).  Every constant is an argument; resource_sweep.py sweeps them.  ORDER-OF-MAGNITUDE tool."""
    n_rot = (n_zz + n_lin + n_logical) * p_layers
    eps_part = eps_total / 3.0
    eps_syn = eps_part / max(n_rot, 1)
    t_per_rot = synth_t_count(eps_syn, synth_a, synth_b)
    t_count = n_rot * t_per_rot
    cnots = 2 * n_zz * p_layers
    # --- magic states: smallest number of distillation levels meeting eps_part over all T gates
    p_in = distill_p_in if distill_p_in is not None else p_phys
    levels = None
    for lv in (1, 2):
        if t_count * t_state_error(p_in, lv) <= eps_part:
            levels = lv
            break
    if levels is None:
        return {"n_logical": n_logical, "p_layers": p_layers, "distance": None,
                "note": "T-state error budget not met with <=2 distillation levels"}
    foot_mult = 1.0 if levels == 1 else level2_footprint_mult
    time_mult = 1.0 if levels == 1 else level2_time_mult
    # --- time in logical steps (1 step = d QEC rounds)
    par = max(1.0, cnot_parallel_frac * n_logical / 2)
    steps_cnot = cnots / par
    steps_T = t_count * t_factory_cycles_per_T * time_mult / max(n_factories, 1)
    steps_total = max(steps_cnot, steps_T) if overlap_T_and_clifford else steps_cnot + steps_T
    for d in range(3, 81, 2):
        p_fail = (n_logical * routing_overhead) * steps_total * d * logical_error_per_round(model, d, p_phys)
        if p_fail <= eps_part:
            phys_data = n_logical * routing_overhead * (2 * d * d - 1)
            phys_fact = n_factories * factory_footprint_d2 * foot_mult * d * d
            time_us = steps_total * d * cycle_time_us
            return {"n_logical": n_logical, "p_layers": p_layers, "rotations": n_rot,
                    "T_per_rotation": t_per_rot, "T_count": t_count, "cnots": cnots,
                    "distill_levels": levels, "T_state_error_total": t_count * t_state_error(p_in, levels),
                    "distance": d, "physical_qubits": int(phys_data + phys_fact),
                    "phys_data": int(phys_data), "phys_factories": int(phys_fact),
                    "steps_clifford": steps_cnot, "steps_T": steps_T, "T_bound": bool(steps_T > steps_cnot),
                    "p_fail_circuit": p_fail, "time_per_shot_ms": time_us / 1e3,
                    "total_time_s": time_us / 1e6 * n_shots}
    return {"n_logical": n_logical, "p_layers": p_layers, "distance": None, "note": "no distance <=79 meets budget"}
