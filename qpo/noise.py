"""Phenomenological NISQ+ / logical noise model for QAOA on diagonal-cost QUBOs.

Every gate error is mapped to an *effective independent bit-flip probability q per
qubit* applied to the output distribution (flip channel on the measured Z-basis
string).  This keeps the simulation exact and cheap (2^n vector) while giving
noise that is NON-uniform on bitstrings (unlike plain global depolarising) and
for which ZNE / PEC have a clean closed form.  It is a model, not hardware truth;
state this limitation in the report.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class NoiseModel:
    p2: float = 3e-3
    p1: float = 3e-4
    p_readout: float = 1e-2
    idle_rate: float = 2e-3
    dd_suppression: float = 0.0     # fraction of idle error removed by dynamical decoupling
    logical_q_per_layer: float | None = None   # if set: logical-qubit mode
    name: str = "physical"

    def circuit_stats(self, n: int, n_zz: int, n_lin: int, p: int):
        cnots = 2 * n_zz * p
        rz = (n_zz + n_lin + n) * p       # ZZ->RZ, local RZ, mixer RX
        return cnots, rz

    def effective_q(self, n, n_zz, n_lin, p, scale: float = 1.0) -> float:
        """Total flip probability per qubit at noise-scale factor `scale` (ZNE)."""
        if self.logical_q_per_layer is not None:
            lam = -0.5 * np.log(max(1 - 2 * self.logical_q_per_layer, 1e-12)) * p * scale
            q_gate = 0.5 * (1 - np.exp(-2 * lam))
            return float(q_gate)   # readout assumed corrected by logical measurement
        cnots, rz = self.circuit_stats(n, n_zz, n_lin, p)
        events = self.p2 * cnots + self.p1 * rz
        idle = self.idle_rate * (1 - self.dd_suppression) * n * p
        lam = (2.0 / 3.0) * (events + idle) / n * scale      # flipping events / qubit
        q_gate = 0.5 * (1 - np.exp(-2 * lam))
        r = self.p_readout
        return float(q_gate + r - 2 * q_gate * r)            # readout NOT scaled by ZNE


def readout_only_q(noise: NoiseModel) -> float:
    return 0.0 if noise.logical_q_per_layer is not None else noise.p_readout
