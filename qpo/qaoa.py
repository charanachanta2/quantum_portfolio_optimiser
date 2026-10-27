"""Exact statevector QAOA for diagonal cost functions + noisy/mitigated variants.

State index s has bit j = qubit j (matches qubo.bits_matrix).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from .noise import NoiseModel, readout_only_q


def _flip_axes(p: np.ndarray, n: int):
    shape = (2,) * n
    return p.reshape(shape), shape


def apply_flip_channel(p: np.ndarray, n: int, q: float) -> np.ndarray:
    """Independent bit-flip (prob q) on every qubit of a probability vector."""
    if q <= 0:
        return p
    t = p.reshape((2,) * n)
    for ax in range(n):
        t = (1 - q) * t + q * np.flip(t, axis=ax)
    return t.reshape(-1)


def apply_inverse_flip_channel(p: np.ndarray, n: int, q: float) -> np.ndarray:
    """Exact inverse of apply_flip_channel (quasi-probability; may be negative)."""
    if q <= 0:
        return p
    t = p.reshape((2,) * n)
    for ax in range(n):
        t = ((1 - q) * t - q * np.flip(t, axis=ax)) / (1 - 2 * q)
    return t.reshape(-1)


class QAOA:
    def __init__(self, energies: np.ndarray, n: int, n_zz: int, n_lin: int):
        self.E = energies
        self.n = n
        self.n_zz, self.n_lin = n_zz, n_lin
        self.Emin, self.Emax = float(energies.min()), float(energies.max())
        self.En = (energies - self.Emin) / max(self.Emax - self.Emin, 1e-12)  # in [0,1]
        self.Emean = float(energies.mean())
        self.psi0 = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)

    # ---- ideal ----------------------------------------------------------
    def probs(self, params: np.ndarray) -> np.ndarray:
        p = len(params) // 2
        gam, bet = params[:p], params[p:]
        n = self.n
        psi = self.psi0.copy()
        for l in range(p):
            psi = psi * np.exp(-1j * gam[l] * np.pi * self.En)
            t = psi.reshape((2,) * n)
            c, s = np.cos(bet[l]), -1j * np.sin(bet[l])
            for ax in range(n):
                t = c * t + s * np.flip(t, axis=ax)
            psi = t.reshape(-1)
        pr = np.abs(psi) ** 2
        return pr / pr.sum()

    def energy_exact(self, params) -> float:
        return float(self.probs(params) @ self.E)

    # ---- noisy ------------------------------------------------------------
    def noisy_probs(self, params, noise: NoiseModel, scale: float = 1.0) -> np.ndarray:
        p = len(params) // 2
        q = noise.effective_q(self.n, self.n_zz, self.n_lin, p, scale)
        return apply_flip_channel(self.probs(params), self.n, q)

    def q_eff(self, p, noise, scale=1.0):
        return noise.effective_q(self.n, self.n_zz, self.n_lin, p, scale)


# ---------------------------------------------------------------------------
# error mitigation
# ---------------------------------------------------------------------------
def richardson_coeffs(scales) -> np.ndarray:
    """Weights c_i with sum c_i = 1 s.t. sum c_i f(s_i) extrapolates polynomially to s=0."""
    s = np.asarray(scales, float)
    V = np.vander(s, increasing=True).T          # rows: powers 0..m-1
    e = np.zeros(len(s)); e[0] = 1.0
    return np.linalg.solve(V, e)


def estimate_energy(qaoa: QAOA, params, noise: NoiseModel, *, shots: int | None,
                    rng: np.random.Generator, method: str = "raw",
                    zne_scales=(1.0, 2.0, 3.0), extrap: str = "richardson"):
    """Energy estimate under noise with optional mitigation.
    method: 'raw' | 'zne' | 'pec'.  shots=None -> infinite-shot (exact in the model).
    Returns (estimate, sampling_overhead_factor)."""
    n, E = qaoa.n, qaoa.E
    p = len(params) // 2

    def sample_mean(pr):
        if shots is None:
            return float(pr @ E)
        idx = rng.choice(len(pr), size=shots, p=pr / pr.sum())
        return float(E[idx].mean())

    if method == "raw":
        return sample_mean(qaoa.noisy_probs(params, noise)), 1.0
    if method == "zne":
        vals = np.array([sample_mean(qaoa.noisy_probs(params, noise, s)) for s in zne_scales])
        if extrap == "richardson":
            c = richardson_coeffs(zne_scales)
            return float(c @ vals), float(np.sum(c ** 2) * len(zne_scales))
        if extrap == "exp":   # a*exp(-b s)+c requires 3 pts; fallback to linear if degenerate
            try:
                s = np.asarray(zne_scales)
                d1, d2 = vals[1] - vals[0], vals[2] - vals[1]
                r = d2 / d1
                if r <= 0 or abs(r - 1) < 1e-9:
                    raise ValueError
                b = -np.log(r) / (s[1] - s[0])
                a = d1 / (np.exp(-b * s[1]) - np.exp(-b * s[0]))
                c0 = vals[0] - a * np.exp(-b * s[0])
                return float(a + c0), 1.0
            except Exception:
                pass
        c = richardson_coeffs(zne_scales[:2])
        return float(c @ vals[:2]), float(np.sum(c ** 2) * 2)
    if method == "pec":
        q = qaoa.q_eff(p, noise)
        ro = readout_only_q(noise)
        gamma = (1 - 2 * q) ** (-n)
        pr = qaoa.noisy_probs(params, noise)
        if shots is None:
            return float(apply_inverse_flip_channel(pr, n, q) @ E), gamma ** 2
        idx = rng.choice(len(pr), size=shots, p=pr / pr.sum())
        flips = rng.random((shots, n)) < q                  # quasi-prob sampling of inverse channel
        mask = (flips * (1 << np.arange(n))).sum(1).astype(np.int64)
        sign = np.where(flips.sum(1) % 2 == 1, -1.0, 1.0)
        return float(gamma * np.mean(sign * E[idx ^ mask])), gamma ** 2
    raise ValueError(method)


def mitigated_distribution(qaoa: QAOA, counts: np.ndarray, q: float) -> np.ndarray:
    """Exact inversion of the flip channel on an empirical histogram (PEC in the
    'full-distribution' form). Result is a quasi-probability vector."""
    return apply_inverse_flip_channel(counts / counts.sum(), qaoa.n, q)


# ---------------------------------------------------------------------------
# training (layer-wise adaptive depth)
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    params: np.ndarray
    p: int
    energy: float
    n_evals: int


def train_qaoa(qaoa: QAOA, noise: NoiseModel | None, *, p_max=3, maxiter=60, tol=0.01,
               rng=None, method="raw", shots=None, zne_scales=(1.0, 2.0, 3.0),
               init: np.ndarray | None = None) -> TrainResult:
    """Layer-wise (adaptive-depth) optimisation.  Adds a layer only while the
    relative improvement (w.r.t. the energy range) exceeds `tol`."""
    rng = rng or np.random.default_rng(0)
    rng_range = max(qaoa.Emax - qaoa.Emin, 1e-12)
    n_evals = 0

    def obj_factory(p):
        def f(x):
            nonlocal n_evals
            n_evals += 1
            if noise is None:
                return qaoa.energy_exact(x)
            return estimate_energy(qaoa, x, noise, shots=shots, rng=rng, method=method,
                                   zne_scales=zne_scales)[0]
        return f

    params = np.array([0.4, 0.4]) if init is None else np.asarray(init, float)
    best = None
    prev_val = None
    for p in range(1, p_max + 1):
        if p > 1:
            g, b = params[: p - 1], params[p - 1:]
            params = np.concatenate([g, [g[-1]], b, [b[-1]]])
        res = minimize(obj_factory(p), params, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.3})
        params = res.x
        val = float(qaoa.energy_exact(params))   # evaluate the TRUE (noiseless) quality for bookkeeping
        cur = TrainResult(params.copy(), p, val, n_evals)
        if prev_val is not None and (prev_val - val) / rng_range < tol:
            best = best if best is not None else cur
            break
        best, prev_val = cur, val
    return best


def run_and_decode(qaoa: QAOA, params, noise: NoiseModel | None, *, shots, rng,
                   mitigate_readout: bool = False, top_k: int = 32):
    """Sample the (noisy) circuit, decode the best bitstring by classical
    evaluation of the cost function.  Returns (state_index, energies of samples)."""
    p = len(params) // 2
    pr = qaoa.probs(params) if noise is None else qaoa.noisy_probs(params, noise)
    idx = rng.choice(len(pr), size=shots, p=pr / pr.sum())
    if mitigate_readout and noise is not None:
        counts = np.bincount(idx, minlength=len(pr)).astype(float)
        q = qaoa.q_eff(p, noise)
        quasi = mitigated_distribution(qaoa, counts, q)
        cand = np.argsort(-quasi)[:top_k]                       # most probable after inversion
        cand = np.concatenate([cand, np.unique(idx)])           # + the raw samples
        best = cand[np.argmin(qaoa.E[cand])]
        return int(best), qaoa.E[idx]
    e = qaoa.E[idx]
    return int(idx[np.argmin(e)]), e
