import numpy as np, pytest
from qpo.config import Config
from qpo.data import synthetic_market
from qpo.qubo import build_qubo, all_energies, bits_matrix
from qpo.qaoa import QAOA, richardson_coeffs, apply_flip_channel, apply_inverse_flip_channel, estimate_energy, train_qaoa
from qpo.noise import NoiseModel
from qpo import theory, qec, classical as C


@pytest.fixture(scope="module")
def inst():
    rng = np.random.default_rng(0)
    cfg = Config(n_assets=3)
    A = 3
    mu = rng.normal(0.01, 0.01, A); S = np.cov(rng.normal(0, .05, (100, A)).T)
    wp = np.ones(A) / A
    return cfg, build_qubo(mu, S, wp, [6, 6], [3, 2], cfg), (mu, S, wp)


def test_qubo_matches_direct_objective(inst):
    cfg, q, (mu, S, wp) = inst
    E = q.energies(); rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.integers(0, 2, q.n); lv = q.levels(x) / cfg.units
        f, prev = 0.0, wp
        for t in range(2):
            w = lv[t]
            f += -mu @ w + 3 * w @ S @ w + cfg.budget_pen * (w.sum() - 1) ** 2 + cfg.tc_quad * ((w - prev) ** 2).sum()
            f += cfg.cap_pen * (t == 1) * (q.levels(x)[1] == 3).sum()
            prev = w
        assert abs(f - E[int((x * 2 ** np.arange(q.n)).sum())]) < 1e-10


def test_qaoa_normalised_and_improves(inst):
    cfg, q, _ = inst
    Q = QAOA(q.energies(), q.n, q.n_couplings(), q.n_linear())
    assert abs(Q.probs(np.array([0.3, 0.2])).sum() - 1) < 1e-12
    tr = train_qaoa(Q, None, p_max=2, maxiter=40)
    assert tr.energy < Q.Emean          # better than uniform sampling


def test_flip_channel_inverse():
    rng = np.random.default_rng(0); p = rng.random(2 ** 5); p /= p.sum()
    assert np.allclose(apply_inverse_flip_channel(apply_flip_channel(p, 5, 0.07), 5, 0.07), p)


def test_pec_exact_and_zne_reduces_bias(inst):
    cfg, q, _ = inst
    Q = QAOA(q.energies(), q.n, q.n_couplings(), q.n_linear()); par = np.array([0.4, 0.3])
    nm = NoiseModel(); rng = np.random.default_rng(0)
    e0 = Q.energy_exact(par)
    raw = estimate_energy(Q, par, nm, shots=None, rng=rng, method="raw")[0]
    zne = estimate_energy(Q, par, nm, shots=None, rng=rng, method="zne")[0]
    pec = estimate_energy(Q, par, nm, shots=None, rng=rng, method="pec")[0]
    assert abs(pec - e0) < 1e-9
    assert abs(zne - e0) < abs(raw - e0)


def test_pec_sampling_unbiased(inst):
    cfg, q, _ = inst
    Q = QAOA(q.energies(), q.n, q.n_couplings(), q.n_linear()); par = np.array([0.4, 0.3])
    nm = NoiseModel(p2=1e-3, p_readout=2e-3); rng = np.random.default_rng(3)
    est = np.mean([estimate_energy(Q, par, nm, shots=4000, rng=rng, method="pec")[0] for _ in range(10)])
    assert abs(est - Q.energy_exact(par)) < 0.1 * (Q.Emax - Q.Emin) ** 0.5 + 0.05


def test_richardson_weights():
    c = richardson_coeffs([1, 2, 3]); assert np.allclose(c, [3, -3, 1]); assert abs(c.sum() - 1) < 1e-12


def test_theory_bounds(inst):
    cfg, q, _ = inst
    Q = QAOA(q.energies(), q.n, q.n_couplings(), q.n_linear()); par = np.array([0.4, 0.3])
    qq = 0.02
    p_ideal = Q.probs(par); p_noisy = apply_flip_channel(p_ideal, q.n, qq)
    for z in range(len(p_ideal)):    # rigorous lower bound holds for EVERY bitstring
        assert p_noisy[z] >= theory.p_opt_noisy_lower_bound(p_ideal[z], q.n, qq) - 1e-15
    assert theory.hoeffding_shots(1.0, 0.1) > 100
    assert theory.zne_variance_factor([1, 2, 3]) == pytest.approx(19)


def test_stim_surface_code_suppression():
    pytest.importorskip("stim"); pytest.importorskip("pymatching")
    a = qec.simulate_memory(3, 0.003, shots=4000, seed=1); b = qec.simulate_memory(5, 0.003, shots=4000, seed=1)
    assert b["p_L"] < a["p_L"]


def test_no_lookahead_and_backtest_runs():
    from qpo.backtest import run_backtest
    m = synthetic_market(n_assets=10, start="2006-01-01", end="2009-12-31")
    cfg = Config(n_assets=3, rebalance_days=63, shots=256, p_max=1, opt_maxiter=15)
    ret, rec = run_backtest(m, cfg, "2008-01-01", "2008-12-31", ["equal_weight_full", "exact_qubo", "qaoa_ideal", "cvar_sub"], verbose=False)
    assert ret.shape[1] == 4 and np.isfinite(ret.values).all()
    # perturb the FUTURE: first-rebalance decisions must not change
    m2 = synthetic_market(n_assets=10, start="2006-01-01", end="2009-12-31")
    m2.returns.iloc[m2.returns.index.get_loc(m2.returns.index[m2.returns.index >= "2008-03-01"][0]):] *= 1.5
    ret2, _ = run_backtest(m2, cfg, "2008-01-01", "2008-12-31", ["exact_qubo"], verbose=False)
    first = ret["exact_qubo"].iloc[:30]; first2 = ret2["exact_qubo"].iloc[:30]
    assert np.allclose(first.values, first2.values)


def test_classical_lp_vs_gurobi():
    gp = pytest.importorskip("gurobipy")
    rng = np.random.default_rng(0); R = rng.normal(0.0005, 0.01, (200, 4)); wp = np.ones(4) / 4
    assert np.allclose(C.cvar_scenario_lp(R, wp), C.cvar_gurobi(R, wp), atol=1e-4)


def test_qiskit_crosscheck(inst):
    """My statevector QAOA must equal Qiskit's for the same Ising/QUBO diagonal."""
    qk = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector, Operator
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    cfg, q, _ = inst
    n = 6                                         # sub-problem (first 6 qubits) keeps it fast
    rng = np.random.default_rng(5)
    E = rng.normal(size=2 ** n)
    Qm = QAOA(E, n, 5, 5)
    g, b = 0.37, 0.52
    mine = Qm.probs(np.array([g, b]))
    # qiskit: diagonal phase exp(-i g pi En) then RX(2b) on all qubits (qiskit little-endian == bit j = qubit j)
    qc = QuantumCircuit(n); qc.h(range(n))
    from qiskit.circuit.library import DiagonalGate
    qc.append(DiagonalGate(list(np.exp(-1j * g * np.pi * Qm.En))), range(n))
    for j in range(n):
        qc.rx(2 * b, j)
    theirs = np.abs(Statevector(qc).data) ** 2
    assert np.allclose(mine, theirs, atol=1e-10)


def test_pec_gamma_reference_values():
    """docs/THEORY.md P3 numbers (errata: the doc used to say 6.6 and 3.6e4)."""
    assert theory.pec_gamma(16, 0.03) ** 2 == pytest.approx(7.243, rel=1e-3)
    assert theory.pec_gamma(100, 0.03) ** 2 == pytest.approx(2.368e5, rel=1e-3)
    assert theory.pec_gamma(12, 0.026) ** 2 == pytest.approx(3.602, rel=1e-3)
