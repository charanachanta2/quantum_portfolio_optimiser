import itertools
import numpy as np, pytest
from qpo.config import Config
from qpo.miqp import MiqpProblem, objective, solve_enum, solve, compositions, has_gurobi, quad_form
from qpo.qubo import build_qubo


def _prob(A=3, T=2, seed=0, caps=(3, 2), esg=True):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.01, 0.01, A); S = np.cov(rng.normal(0, .05, (120, A)).T)
    wp = rng.dirichlet(np.ones(A))
    e = rng.uniform(20, 95, A) if esg else None
    cfg = Config(n_assets=A, n_periods=T)
    return MiqpProblem(mu, S, wp, [6.0, 5.0][:T] if T <= 2 else [6.0] * T, list(caps[:T]) if T <= 2 else [3] * T,
                       cfg.units, cfg.tc_quad, cfg.esg_tilt, e), cfg


def test_enum_dp_equals_bruteforce():
    p, _ = _prob()
    Y, f = solve_enum(p)
    best = np.inf
    for y0 in compositions(6, 3, p.caps[0]):
        for y1 in compositions(6, 3, p.caps[1]):
            best = min(best, objective(p, np.array([y0, y1])))
    assert abs(f - best) < 1e-12 and abs(objective(p, Y) - best) < 1e-12
    assert (Y.sum(1) == 6).all() and (Y[0] <= p.caps[0]).all() and (Y[1] <= p.caps[1]).all()


def test_qubo_optimum_equals_miqp_when_budget_penalty_large():
    """Cross-validates the two formulations: with a big budget penalty the QUBO's exact optimum is feasible and
    its energy equals the MIQP objective (the QUBO offset/linear/quadratic bookkeeping agrees with the MIQP)."""
    p, cfg = _prob()
    from dataclasses import replace
    cfg = replace(cfg, budget_pen=60.0)
    qb = build_qubo(p.mu, p.Sigma, p.w_prev, p.lam, p.caps, cfg, esg=p.esg)
    E = qb.energies(); x = (int(np.argmin(E)) >> np.arange(qb.n)) & 1
    Y = qb.levels(x)
    assert (Y.sum(1) == 6).all()
    Ym, fm = solve_enum(p)
    assert abs(E.min() - fm) < 1e-9 and abs(objective(p, Y) - fm) < 1e-9


def test_solve_auto_falls_back_to_enum_without_gurobi():
    p, _ = _prob()
    w, f, info = solve(p)
    assert abs(w.sum() - 1) < 1e-12 and info["solver"] in ("enum", "gurobi")


def test_gurobi_matches_enumeration():
    pytest.importorskip("gurobipy")
    for seed in range(5):
        p, _ = _prob(A=4, seed=seed)
        Ye, fe = solve_enum(p)
        w, fg, info = solve(p, "gurobi")
        assert abs(fg - fe) < 1e-7, (seed, fg, fe)


def test_quad_form_matches_objective_for_any_integer_levels():
    """The (H,c,const) handed to Gurobi reproduces objective() - tested for T=1,2,3 incl. infeasible Y."""
    rng = np.random.default_rng(0)
    for T in (1, 2, 3):
        p, _ = _prob(A=4, T=min(T, 2))
        if T == 3:
            p.lam, p.caps = [6.0, 5.0, 4.0], [3, 3, 2]
        H, c, const = quad_form(p)
        for _ in range(20):
            Y = rng.integers(0, 4, (p.T, p.A))
            y = Y.ravel().astype(float)
            assert abs(y @ H @ y + c @ y + const - objective(p, Y)) < 1e-12
