"""Discrete multi-period portfolio problem as a convex MIQP (the classical "gold standard" for the QUBO's
own problem) - solved by Gurobi when available, and by an EXACT dynamic program otherwise.

Same economics as qubo.build_qubo but with HARD constraints instead of penalties:
    min  sum_t [ -mu.w_t - esg_tilt * s.w_t/100 + lam_t/2 w_t' Sigma w_t + tc_quad * ||w_t - w_{t-1}||^2 ]
    s.t. y_t in {0,..,cap_t}^A integer,  sum_i y_t[i] = units,   w_t = y_t / units,   w_0 = current holdings.
Differences to the QUBO that matter when comparing solvers:
  * budget is an exact equality (QUBO: soft penalty -> can return infeasible bitstrings that need repair);
  * levels are not limited to 2 bits, so the same code runs on the FULL eligible universe (the QUBO can only
    encode ~4 assets on qubits) -> `miqp_full` separates "asset selection" from "solver" effects.
The DP is exact because the only inter-period coupling is the pairwise tc_quad term.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass
import numpy as np


@dataclass
class MiqpProblem:
    mu: np.ndarray          # (A,) horizon-scaled expected return
    Sigma: np.ndarray       # (A,A) horizon-scaled covariance
    w_prev: np.ndarray      # (A,) current holdings (sum 1)
    lam: list               # (T,) risk aversion
    caps: list              # (T,) max integer level per asset
    units: int
    tc_quad: float
    esg_tilt: float = 0.0
    esg: np.ndarray | None = None

    @property
    def T(self): return len(self.lam)
    @property
    def A(self): return len(self.mu)


def objective(prob: MiqpProblem, Y: np.ndarray) -> float:
    """Y: (T,A) integer levels. Same quantity the QUBO energy equals when the budget holds."""
    W = np.asarray(Y, float) / prob.units
    f, prev = 0.0, prob.w_prev
    lin = -prob.mu - (prob.esg_tilt * np.asarray(prob.esg) / 100.0 if prob.esg is not None else 0.0)
    for t in range(prob.T):
        w = W[t]
        f += lin @ w + 0.5 * prob.lam[t] * w @ prob.Sigma @ w + prob.tc_quad * np.sum((w - prev) ** 2)
        prev = w
    return float(f)


def compositions(total: int, parts: int, cap: int):
    """All integer vectors of length `parts`, entries in [0,cap], summing to `total`."""
    def rec(left, k):
        if k == 1:
            if 0 <= left <= cap:
                yield (left,)
            return
        for v in range(min(cap, left) + 1):
            for rest in rec(left - v, k - 1):
                yield (v,) + rest
    yield from rec(total, parts)


def solve_enum(prob: MiqpProblem, max_states: int = 4000):
    """Exact DP over periods (Viterbi on the composition lattice). Returns (Y (T,A) ints, objective)."""
    A, T, U = prob.A, prob.T, prob.units
    states = [np.array(list(compositions(U, A, prob.caps[t])), dtype=float) for t in range(T)]
    if max(len(s) for s in states) > max_states:
        raise ValueError(f"too many states ({max(len(s) for s in states)}) for exact enumeration - use Gurobi")
    lin = -prob.mu - (prob.esg_tilt * np.asarray(prob.esg) / 100.0 if prob.esg is not None else 0.0)
    cost, back = None, []
    prev_W = (prob.w_prev[None, :])
    prev_cost = np.zeros(1)
    for t in range(T):
        W = states[t] / U
        node = W @ lin + 0.5 * prob.lam[t] * np.einsum("ij,jk,ik->i", W, prob.Sigma, W)
        d2 = ((W[:, None, :] - prev_W[None, :, :]) ** 2).sum(-1)              # (S_t, S_{t-1})
        tot = prev_cost[None, :] + prob.tc_quad * d2
        arg = tot.argmin(1)
        prev_cost = node + tot[np.arange(len(W)), arg]
        back.append(arg)
        prev_W = W
    j = int(prev_cost.argmin())
    best = float(prev_cost[j])
    idx = [j]
    for t in range(T - 1, 0, -1):
        j = int(back[t][j]); idx.append(j)
    idx = idx[::-1]
    Y = np.stack([states[t][idx[t]] for t in range(T)]).astype(int)
    return Y, best


def quad_form(prob: MiqpProblem):
    """Matrices (H, c, const) with  objective(Y) = y'Hy + c'y + const  for y = Y.ravel() (integer levels).
    Split out so the algebra handed to Gurobi is unit-tested without Gurobi."""
    A, T, U = prob.A, prob.T, prob.units
    n = T * A
    lin = -prob.mu - (prob.esg_tilt * np.asarray(prob.esg) / 100.0 if prob.esg is not None else 0.0)
    H = np.zeros((n, n)); c = np.zeros(n); const = 0.0
    for t in range(T):
        s = slice(t * A, (t + 1) * A)
        H[s, s] += 0.5 * prob.lam[t] * prob.Sigma / U ** 2 + prob.tc_quad * np.eye(A) / U ** 2
        c[s] += lin / U
        if t == 0:
            c[s] += -2 * prob.tc_quad * prob.w_prev / U
            const += prob.tc_quad * float(prob.w_prev @ prob.w_prev)
        else:
            sp = slice((t - 1) * A, t * A)
            H[sp, sp] += prob.tc_quad * np.eye(A) / U ** 2
            H[s, sp] += -prob.tc_quad * np.eye(A) / U ** 2
            H[sp, s] += -prob.tc_quad * np.eye(A) / U ** 2
    return 0.5 * (H + H.T), c, const


def solve_gurobi(prob: MiqpProblem, time_limit: float = 30.0, mip_gap: float = 0.0):
    """Convex MIQP via Gurobi's matrix API (NOT executed in the build sandbox: gurobipy unavailable there;
    `tests/test_miqp.py::test_gurobi_matches_enumeration` checks it against the exact DP on your machine)."""
    import gurobipy as gp
    from gurobipy import GRB
    A, T, U = prob.A, prob.T, prob.units
    n = T * A
    H, c, const = quad_form(prob)
    m = gp.Model("qpo_miqp")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    ub = np.repeat(np.asarray(prob.caps, float), A)
    y = m.addMVar(n, vtype=GRB.INTEGER, lb=0, ub=ub)
    S = np.zeros((T, n))
    for t in range(T):
        S[t, t * A:(t + 1) * A] = 1.0
    m.addConstr(S @ y == U)
    m.setObjective(y @ H @ y + c @ y + const, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"Gurobi found no solution (status {m.Status})")
    Y = np.rint(y.X).astype(int).reshape(T, A)
    return Y, float(m.ObjVal), {"status": int(m.Status), "mip_gap": float(m.MIPGap), "runtime": float(m.Runtime)}


def first_period_weights(Y: np.ndarray, units: int) -> np.ndarray:
    return np.asarray(Y[0], float) / units


def solve(prob: MiqpProblem, prefer: str = "auto"):
    """-> (weights_first_period, objective, info). prefer: 'auto' | 'gurobi' | 'enum'."""
    if prefer in ("auto", "gurobi"):
        try:
            Y, f, info = solve_gurobi(prob)
            return first_period_weights(Y, prob.units), f, {"solver": "gurobi", **info}
        except ImportError:
            if prefer == "gurobi":
                raise
        except Exception as e:                     # e.g. size-limited licence, numerical trouble
            if prefer == "gurobi":
                raise
            info_fallback = {"gurobi_error": f"{type(e).__name__}: {e}"[:120]}
            Y, f = solve_enum(prob)
            return first_period_weights(Y, prob.units), f, {"solver": "enum", **info_fallback}
    Y, f = solve_enum(prob)
    return first_period_weights(Y, prob.units), f, {"solver": "enum"}


def has_gurobi() -> bool:
    try:
        import gurobipy  # noqa: F401
        return True
    except Exception:
        return False
