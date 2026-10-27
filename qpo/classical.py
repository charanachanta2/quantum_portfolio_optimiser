"""Classical baselines.

* exact_qubo      : brute force over all 2^n states (n<=~22) = true optimum of the QUBO
* simulated_annealing : QUBO heuristic with an energy-evaluation budget
* cvar_scenario_lp: single-period CVaR stochastic program (HiGHS via SciPy)
* cvar_gurobi     : same program solved with Gurobi (if gurobipy importable)
* ESAgent         : small MLP policy trained with evolution strategies (NumPy "deep RL")
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp


def exact_qubo(energies: np.ndarray) -> int:
    return int(np.argmin(energies))


def simulated_annealing(Q: np.ndarray, offset: float, n_evals: int, rng, t0=None, t1=None):
    """Single-flip SA, `n_evals` single-flip energy evaluations. Returns (x, E, trace)."""
    n = Q.shape[0]
    x = rng.integers(0, 2, n).astype(float)
    E = float(x @ Q @ x + offset)
    scale = np.abs(Q).max()
    t0 = t0 or 0.5 * scale
    t1 = t1 or 0.01 * scale
    best_x, best_E = x.copy(), E
    for it in range(n_evals):
        T = t0 * (t1 / t0) ** (it / max(n_evals - 1, 1))
        i = rng.integers(n)
        dE = (1 - 2 * x[i]) * (Q[i, i] + 2 * (Q[i] @ x - Q[i, i] * x[i]))
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            x[i] = 1 - x[i]
            E += dE
            if E < best_E:
                best_E, best_x = E, x.copy()
    return best_x, best_E


def evals_to_optimum_sa(Q, offset, E_opt, rng, max_evals=200000, tol=1e-9, restarts_budget=None):
    """Number of single-flip evaluations until SA (with geometric restarts of length 500n) hits the optimum."""
    n = Q.shape[0]
    L = 200 * n
    used = 0
    while used < max_evals:
        _, E = simulated_annealing(Q, offset, L, rng)
        used += L
        if E <= E_opt + tol:
            return used
    return max_evals


# ---------------------------------------------------------------------------
def _cvar_lp_matrices(R, w_prev, alpha, tc, rho, cap):
    S, A = R.shape
    # variables: w(A) u(A) d(A) z(S) eta(1)
    nv = 3 * A + S + 1
    c = np.zeros(nv)
    c[A:2 * A] = tc
    c[2 * A:3 * A] = tc
    c[3 * A:3 * A + S] = 1.0 / ((1 - alpha) * S)
    c[-1] = 1.0
    c[:A] -= rho * R.mean(0)
    # z_s >= -R_s w - eta   ->  -R_s w - eta - z_s <= 0
    A_ub = sp.hstack([sp.csr_matrix(-R), sp.csr_matrix((S, 2 * A)), -sp.eye(S), -np.ones((S, 1))]).tocsr()
    b_ub = np.zeros(S)
    A_eq = sp.vstack([
        sp.hstack([sp.csr_matrix(np.ones((1, A))), sp.csr_matrix((1, 2 * A + S + 1))]),
        sp.hstack([sp.eye(A), -sp.eye(A), sp.eye(A), sp.csr_matrix((A, S + 1))]),
    ]).tocsr()
    b_eq = np.concatenate([[1.0], w_prev])
    bounds = [(0, cap)] * A + [(0, None)] * (2 * A) + [(0, None)] * S + [(None, None)]
    return c, A_ub, b_ub, A_eq, b_eq, bounds


def cvar_scenario_lp(R, w_prev, alpha=0.9, tc=0.001, rho=1.0, cap=1.0):
    """min CVaR_alpha(-R w) - rho*E[R w] + tc*|w-w_prev|_1 ; s.t. sum w = 1, 0<=w<=cap."""
    A = R.shape[1]
    c, A_ub, b_ub, A_eq, b_eq, bounds = _cvar_lp_matrices(R, w_prev, alpha, tc, rho, cap)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return np.full(A, 1.0 / A)
    w = np.clip(res.x[:A], 0, None)
    return w / w.sum()


def cvar_gurobi(R, w_prev, alpha=0.9, tc=0.001, rho=1.0, cap=1.0):
    import gurobipy as gp
    from gurobipy import GRB
    S, A = R.shape
    m = gp.Model()
    m.Params.OutputFlag = 0
    w = m.addVars(A, lb=0, ub=cap)
    u = m.addVars(A, lb=0)
    d = m.addVars(A, lb=0)
    z = m.addVars(S, lb=0)
    eta = m.addVar(lb=-GRB.INFINITY)
    m.addConstr(w.sum() == 1)
    for i in range(A):
        m.addConstr(w[i] - w_prev[i] == u[i] - d[i])
    for s in range(S):
        m.addConstr(z[s] >= -gp.quicksum(R[s, i] * w[i] for i in range(A)) - eta)
    mean = R.mean(0)
    m.setObjective(eta + gp.quicksum(z[s] for s in range(S)) / ((1 - alpha) * S)
                   - rho * gp.quicksum(mean[i] * w[i] for i in range(A))
                   + tc * gp.quicksum(u[i] + d[i] for i in range(A)), GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return np.full(A, 1.0 / A)
    x = np.array([w[i].X for i in range(A)])
    x = np.clip(x, 0, None)
    return x / x.sum()


# ---------------------------------------------------------------------------
class ESAgent:
    """Policy w = softmax(MLP(features)), trained by evolution strategies on a
    training window (stand-in for a PyTorch RL agent; see rl_torch.py for the torch version).
    features per asset: [mean/vol (z-scored over window), vol, current weight]."""

    def __init__(self, n_assets, hidden=8, seed=0):
        self.rng = np.random.default_rng(seed)
        self.A, self.h = n_assets, hidden
        self.theta = self.rng.normal(0, 0.1, 3 * hidden + hidden + hidden + 1)

    def _unpack(self, th):
        h = self.h
        W1 = th[:3 * h].reshape(3, h)
        b1 = th[3 * h:4 * h]
        W2 = th[4 * h:5 * h]
        b2 = th[5 * h]
        return W1, b1, W2, b2

    @staticmethod
    def features(window: np.ndarray, w_prev: np.ndarray):
        mu = window.mean(0)
        vol = window.std(0) + 1e-9
        sr = mu / vol
        sr = (sr - sr.mean()) / (sr.std() + 1e-9)
        return np.stack([sr, vol / vol.mean() - 1, w_prev * len(w_prev) - 1], 1)

    def act(self, window, w_prev, th=None):
        W1, b1, W2, b2 = self._unpack(self.theta if th is None else th)
        f = self.features(window, w_prev)
        s = np.tanh(f @ W1 + b1) @ W2 + b2
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    def fit(self, rets: np.ndarray, rebalance=21, lookback=126, tc=0.001, risk=3.0,
            iters=40, pop=16, sigma=0.15, lr=0.1):
        T = len(rets)
        starts = list(range(lookback, T - rebalance, rebalance))

        def fitness(th):
            w = np.full(self.A, 1 / self.A)
            tot = []
            for s in starts:
                wn = self.act(rets[s - lookback:s], w, th)
                r = rets[s:s + rebalance] @ wn
                tot.append(r.sum() - tc * np.abs(wn - w).sum() - risk * r.var() * rebalance)
                w = wn
            return np.mean(tot)

        for _ in range(iters):
            eps = self.rng.normal(0, 1, (pop, len(self.theta)))
            f = np.array([fitness(self.theta + sigma * e) for e in eps])
            f = (f - f.mean()) / (f.std() + 1e-9)
            self.theta = self.theta + lr / (pop * sigma) * (eps.T @ f) * sigma
        return self
