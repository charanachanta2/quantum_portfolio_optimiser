"""OPTIONAL (item 7): circuit-level QAOA in Qiskit for one instance, cross-checked against qpo.qaoa and against the gate-count
model used by qpo.noise.   pip install qiskit [qiskit-aer]    python scripts/optional_qiskit_qaoa.py
Checks printed: (1) max |P_qiskit - P_qpo| (must be ~1e-12); (2) transpiled CX / RZ counts vs noise.circuit_stats;
(3) if qiskit-aer is installed: sampled energy under a depolarising noise model for a few error rates.
STATUS: the Ising decomposition is unit-tested in numpy (tests/test_ising.py); the Qiskit API calls were NOT run in the build
sandbox (qiskit unavailable).  Qiskit's API drifts between releases - if a call fails, fix the call, the math is checked."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo.config import Config
from qpo.data import synthetic_market
from qpo.backtest import make_instance
from qpo.qaoa import QAOA, train_qaoa
from qpo.ising import qubo_to_ising
from qpo.noise import NoiseModel


def build_circuit(qb, E, params):
    from qiskit import QuantumCircuit
    n = qb.n; p = len(params) // 2
    h, J, _ = qubo_to_ising(qb.Q, qb.offset)
    delta = E.max() - E.min()
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for l in range(p):
        theta = params[l] * np.pi / delta                       # exp(-i theta H_C),  H_C = sum h Z + sum J ZZ
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2 * theta * J[i, j], i, j)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * theta * h[i], i)
        qc.rx(2 * params[p + l], range(n))                      # exp(-i beta X)
    return qc


def main():
    from qiskit.quantum_info import Statevector
    from qiskit import transpile
    cfg = Config(n_assets=3)
    m = synthetic_market(n_assets=10)
    N = m.returns.shape[1]
    inst = make_instance(m, int(np.searchsorted(m.returns.index, np.datetime64("2008-10-15"))), np.full(N, 1 / N), cfg)
    qb = inst.qubo; E = qb.energies()
    q = QAOA(E, qb.n, qb.n_couplings(), qb.n_linear())
    tr = train_qaoa(q, None, p_max=2, maxiter=40)
    qc = build_circuit(qb, E, tr.params)
    p_qk = np.abs(Statevector(qc).data) ** 2
    print(f"n={qb.n} qubits, p={tr.p}; max |P_qiskit - P_qpo| = {np.max(np.abs(p_qk - q.probs(tr.params))):.2e}")
    tq = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=0)
    ops = tq.count_ops()
    cnots, rz = NoiseModel().circuit_stats(qb.n, qb.n_couplings(), qb.n_linear(), tr.p)
    print(f"transpiled: cx={ops.get('cx', 0)} rz={ops.get('rz', 0)}   model: cnots={cnots} rz={rz}")
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel as AerNoise, depolarizing_error
    except ImportError:
        print("qiskit-aer not installed: skipping noisy simulation"); return
    qm = qc.copy(); qm.measure_all()
    for p2 in (1e-4, 1e-3, 3e-3, 1e-2):
        nm = AerNoise(); nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        res = AerSimulator(noise_model=nm).run(transpile(qm, basis_gates=["cx", "rz", "sx", "x"]), shots=4096, seed_simulator=1).result()
        counts = res.get_counts(); e = sum(E[int(k.replace(" ", ""), 2)] * v for k, v in counts.items()) / 4096
        print(f"  cx depolarising p2={p2:g}: <E>={e:.4f}   (ideal {q.energy_exact(tr.params):.4f}, random {E.mean():.4f})")


if __name__ == "__main__":
    main()
