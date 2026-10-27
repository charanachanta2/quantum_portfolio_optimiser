"""Real-valued genetic algorithm (RVGA) - tournament selection, arithmetic crossover, Gaussian mutation, elitism (E1).
Chromosome (as in the README): [learning_rate, lstm_units, dropout, batch_size, window_size].  Integer genes are rounded on decode."""
from __future__ import annotations
from typing import Callable
import numpy as np

BOUNDS = {"lr": (3e-4, 5e-3), "units": (16, 96), "dropout": (0.0, 0.4), "batch": (32, 256), "window": (15, 60)}
NAMES = list(BOUNDS)


def decode(chrom) -> dict:
    lr, units, dr, batch, win = [float(x) for x in chrom]
    return {"lr": lr, "units": int(round(units / 8) * 8), "dropout": dr, "batch": int(round(batch / 32) * 32), "seq_len": int(round(win))}


def real_ga(fitness: Callable[[np.ndarray], float], pop: int = 8, gens: int = 4, seed: int = 0, tournament: int = 3,
            p_cx: float = 0.8, sigma: float = 0.15, elite: int = 1, verbose: bool = False):
    """Minimises `fitness(chromosome)`.  Returns (best_chrom, best_fit, history)."""
    rng = np.random.default_rng(seed)
    lo = np.array([BOUNDS[k][0] for k in NAMES]); hi = np.array([BOUNDS[k][1] for k in NAMES])
    P = lo + rng.random((pop, len(NAMES))) * (hi - lo)
    F = np.array([fitness(c) for c in P])
    hist = [float(F.min())]
    for g in range(gens):
        order = np.argsort(F)
        newP = [P[i].copy() for i in order[:elite]]
        while len(newP) < pop:
            def pick():
                idx = rng.integers(0, pop, tournament)
                return P[idx[np.argmin(F[idx])]]
            a, b = pick(), pick()
            if rng.random() < p_cx:
                lam = rng.random(len(NAMES))                       # arithmetic crossover
                child = lam * a + (1 - lam) * b
            else:
                child = a.copy()
            child = child + rng.normal(0, sigma, len(NAMES)) * (hi - lo) * (rng.random(len(NAMES)) < 0.3)   # Gaussian mutation
            newP.append(np.clip(child, lo, hi))
        P = np.array(newP)
        F = np.array([fitness(c) for c in P])
        hist.append(float(F.min()))
        if verbose:
            print(f"  GA gen {g + 1}/{gens}: best val loss {F.min():.5f}", flush=True)
    k = int(np.argmin(F))
    return P[k], float(F[k]), hist
