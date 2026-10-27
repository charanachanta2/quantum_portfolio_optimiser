"""QUBO <-> Ising / dimod conversions shared by the optional Ocean and Qiskit demos.
Convention: x_i = (1 - z_i)/2  (qubit |0> <-> x=0 <-> z=+1), so  E(x) = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j."""
from __future__ import annotations
import numpy as np


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0):
    """Q symmetric (E = x'Qx + offset, diagonal = linear). Returns h (n,), J (n,n upper-triangular), const."""
    n = Q.shape[0]
    Qs = 0.5 * (Q + Q.T)
    h = -0.5 * np.diag(Qs) - 0.5 * (Qs.sum(1) - np.diag(Qs))
    J = np.triu(0.5 * Qs, 1)
    const = offset + 0.5 * np.trace(Qs) + 0.5 * np.triu(Qs, 1).sum()
    return h, J, float(const)


def ising_energy(h, J, const, x: np.ndarray) -> float:
    z = 1 - 2 * np.asarray(x, float)
    return float(const + h @ z + z @ J @ z)


def qubo_to_dict(Q: np.ndarray):
    """dimod-style QUBO dict {(i,j): coeff} with E = sum_{i<=j} coeff x_i x_j (+ offset handled separately)."""
    n = Q.shape[0]
    Qs = 0.5 * (Q + Q.T)
    d = {(i, i): float(Qs[i, i]) for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Qs[i, j]) > 1e-15:
                d[(i, j)] = float(2 * Qs[i, j])
    return d


def dict_energy(d: dict, offset: float, x) -> float:
    return float(offset + sum(c * x[i] * x[j] for (i, j), c in d.items()))
