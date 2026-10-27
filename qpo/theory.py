"""Closed-form bounds used in docs/THEORY.md (all derived for the independent
bit-flip noise model of qpo.noise; each function states what it bounds)."""
from __future__ import annotations
import math
import numpy as np


def hoeffding_shots(value_range: float, eps: float, delta: float = 0.05) -> int:
    """Shots N s.t. |mean(E(x_i)) - <E>| <= eps w.p. >= 1-delta, for E in an interval of width `value_range`.
    N >= range^2 ln(2/delta) / (2 eps^2)."""
    return int(math.ceil(value_range ** 2 * math.log(2 / delta) / (2 * eps ** 2)))


def pec_gamma(n: int, q: float) -> float:
    """1-norm of the inverse of n independent bit-flip(q) channels: gamma = (1-2q)^(-n)."""
    return (1 - 2 * q) ** (-n)


def pec_shots(value_range: float, eps: float, n: int, q: float, delta: float = 0.05) -> int:
    """Shots for PEC to reach error eps: samples lie in [-gamma*|E|max, +gamma*|E|max] -> Hoeffding with range 2*gamma*Emax."""
    return hoeffding_shots(2 * pec_gamma(n, q) * value_range, eps, delta)


def zne_variance_factor(scales) -> float:
    """Variance inflation of Richardson extrapolation with equal shots per scale: sum c_i^2."""
    s = np.asarray(scales, float)
    V = np.vander(s, increasing=True).T
    e = np.zeros(len(s)); e[0] = 1
    c = np.linalg.solve(V, e)
    return float(np.sum(c ** 2))


def p_opt_noisy_lower_bound(p_opt_ideal: float, n: int, q: float) -> float:
    """P_noisy(z*) >= P_ideal(z*) (1-q)^n  (the no-error branch of the channel)."""
    return p_opt_ideal * (1 - q) ** n


def shots_to_success(p: float, delta: float = 0.01) -> float:
    """Shots N with P(see optimum at least once) >= 1-delta:  N = ln(delta)/ln(1-p)."""
    if p <= 0:
        return float("inf")
    if p >= 1:
        return 1.0
    return math.log(delta) / math.log1p(-p)


def grover_queries(n: int) -> float:
    return math.pi / 4 * math.sqrt(2 ** n)


def noise_budget_for_advantage(n: int, p_opt_ideal: float, shots_classical: float, delta=0.01) -> float:
    """Largest per-qubit flip prob q such that the *lower bound* on QAOA shots-to-solution
    stays below `shots_classical`.  Solves ln(delta)/ln(1-p_opt (1-q)^n) <= shots_classical."""
    target_p = 1 - delta ** (1 / shots_classical)
    if target_p > p_opt_ideal:
        return 0.0
    return float(1 - (target_p / p_opt_ideal) ** (1 / n))
