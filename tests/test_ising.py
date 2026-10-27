import numpy as np
from qpo.config import Config
from qpo.qubo import build_qubo, bits_matrix
from qpo.ising import qubo_to_ising, ising_energy, qubo_to_dict, dict_energy
from qpo.qaoa import QAOA


def _qubo():
    rng = np.random.default_rng(2); A = 3; cfg = Config(n_assets=A)
    mu = rng.normal(0.01, 0.01, A); S = np.cov(rng.normal(0, .05, (100, A)).T)
    return build_qubo(mu, S, np.ones(A) / A, [6, 5], [3, 2], cfg, esg=rng.uniform(20, 90, A))


def test_ising_and_dict_reproduce_every_qubo_energy():
    q = _qubo(); h, J, c = qubo_to_ising(q.Q, q.offset); d = qubo_to_dict(q.Q)
    B = bits_matrix(q.n); E = q.energies()
    for k in np.random.default_rng(0).integers(0, 2 ** q.n, 200):
        assert abs(ising_energy(h, J, c, B[k]) - E[k]) < 1e-10
        assert abs(dict_energy(d, q.offset, B[k]) - E[k]) < 1e-10


def test_ising_phase_layer_equals_qaoa_cost_layer_up_to_global_phase():
    """The Qiskit demo applies exp(-i*theta*(sum h Z + sum J ZZ)) with theta=gamma*pi/Delta; that must equal the
    simulator's diagonal phase exp(-i*gamma*pi*En) up to one global phase."""
    q = _qubo(); E = q.energies(); Qa = QAOA(E, q.n, q.n_couplings(), q.n_linear())
    h, J, c = qubo_to_ising(q.Q, q.offset); B = bits_matrix(q.n); Z = 1 - 2 * B
    gamma = 0.37; theta = gamma * np.pi / (Qa.Emax - Qa.Emin)
    H = c + Z @ h + np.einsum("ki,ij,kj->k", Z, J, Z)
    ph_ising = np.exp(-1j * theta * H); ph_sim = np.exp(-1j * gamma * np.pi * Qa.En)
    ratio = ph_ising / ph_sim
    assert np.allclose(ratio, ratio[0], atol=1e-9)
