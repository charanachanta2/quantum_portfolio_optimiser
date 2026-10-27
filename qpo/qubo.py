"""Multi-period, time-dependent portfolio QUBO.

Variables x[t,i,k] in {0,1};  level[t,i] = sum_k 2^k x[t,i,k] ;  w[t,i] = level/units.

Objective (to MINIMISE), summed over t = 0..T-1:
    - mu_t . w_t                                   (expected return, horizon units)
    + lambda_t/2 * w_t' Sigma_t w_t                (risk, time-varying lambda_t)
    + tc_quad * || w_t - w_{t-1} ||^2              (quadratic surrogate of linear costs)
    + P_b (1' w_t - 1)^2                           (soft budget constraint)
    + P_c * x[t,i,0] x[t,i,1]  if cap_t == 2 levels (soft/"hard-ish" cap, K=2)
    - esg_tilt * s . w_t                           (soft ESG preference)
HARD constraints: ESG exclusion (removed before encoding) and post-decode repair.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Qubo:
    Q: np.ndarray          # symmetric, E(x) = x'Qx + offset  (diag = linear terms)
    offset: float
    n: int
    A: int
    K: int
    T: int
    units: int

    def energies(self) -> np.ndarray:
        """Energies of all 2^n bitstrings; index bit j == variable j."""
        return all_energies(self.Q, self.offset)

    def levels(self, x: np.ndarray) -> np.ndarray:
        """x: (n,) bits -> (T, A) integer levels."""
        x = np.asarray(x).reshape(self.T, self.A, self.K)
        return (x * (2 ** np.arange(self.K))).sum(-1)

    def first_period_weights(self, x) -> np.ndarray:
        lv = self.levels(x)[0].astype(float)
        return repair_weights(lv / self.units)

    def n_couplings(self) -> int:
        U = np.triu(self.Q, 1)
        return int((np.abs(U) > 1e-12).sum())

    def n_linear(self) -> int:
        return int((np.abs(np.diag(self.Q)) > 1e-12).sum())


def repair_weights(w: np.ndarray) -> np.ndarray:
    """Hard-constraint repair: non-negative, fully invested."""
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 1e-12 else np.full_like(w, 1.0 / len(w))


def bits_matrix(n: int) -> np.ndarray:
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.float64)


def all_energies(Q: np.ndarray, offset: float = 0.0) -> np.ndarray:
    B = bits_matrix(Q.shape[0])
    return np.einsum("ij,ij->i", B @ Q, B) + offset


def build_qubo(mu, Sigma, w_prev, lam, caps, cfg, esg=None) -> Qubo:
    """mu: (A,) horizon-scaled; Sigma: (A,A) horizon-scaled;
    w_prev: (A,) current holdings on the encoded assets;
    lam: list of length T of risk aversions; caps: list of length T (max level);
    esg: (A,) scores in [0,100] or None."""
    A, K, T, U = len(mu), cfg.n_bits, cfg.n_periods, cfg.units
    nv = T * A * K
    # M maps bits -> weights for every (t,i)
    M = np.zeros((T * A, nv))
    for t in range(T):
        for i in range(A):
            for k in range(K):
                M[t * A + i, (t * A + i) * K + k] = (2 ** k) / U
    H = np.zeros((T * A, T * A))
    h = np.zeros(T * A)
    c = 0.0
    ones = np.ones(A)
    for t in range(T):
        s = slice(t * A, (t + 1) * A)
        h[s] += -mu
        if esg is not None:
            h[s] += -cfg.esg_tilt * np.asarray(esg) / 100.0
        H[s, s] += 0.5 * lam[t] * Sigma
        H[s, s] += cfg.budget_pen * np.outer(ones, ones)
        h[s] += -2 * cfg.budget_pen * ones
        c += cfg.budget_pen
        # transaction-cost surrogate
        H[s, s] += cfg.tc_quad * np.eye(A)
        if t == 0:
            h[s] += -2 * cfg.tc_quad * np.asarray(w_prev)
            c += cfg.tc_quad * float(np.dot(w_prev, w_prev))
        else:
            sp = slice((t - 1) * A, t * A)
            H[sp, sp] += cfg.tc_quad * np.eye(A)
            H[s, sp] += -cfg.tc_quad * np.eye(A)
            H[sp, s] += -cfg.tc_quad * np.eye(A)
    H = 0.5 * (H + H.T)
    Q = M.T @ H @ M
    Q = Q + np.diag(M.T @ h)          # x^2 = x  => linear terms on the diagonal
    # cap penalty: forbid level 3 (both bits set) when cap is 2 levels
    if K == 2:
        for t in range(T):
            if caps[t] < 3:
                for i in range(A):
                    a, b = (t * A + i) * K, (t * A + i) * K + 1
                    Q[a, b] += cfg.cap_pen / 2
                    Q[b, a] += cfg.cap_pen / 2
    return Qubo(0.5 * (Q + Q.T), float(c), nv, A, K, T, U)
