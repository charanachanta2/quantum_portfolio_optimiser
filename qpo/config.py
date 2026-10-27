from dataclasses import dataclass


@dataclass
class Config:
    # ---- problem size -------------------------------------------------
    n_assets: int = 4          # assets actually encoded on qubits
    n_bits: int = 2            # bits per asset per period (levels 0..2^K-1)
    n_periods: int = 2         # look-ahead horizon (receding horizon execution)
    units: int = 6             # budget = `units` unit-weights; w = level/units
    # ---- finance ------------------------------------------------------
    rebalance_days: int = 21
    lookback: int = 252
    lambda0: float = 6.0       # base risk aversion
    kappa_vix: float = 0.5     # lambda_t = lambda0*(1+kappa*z_vix)
    cap_levels_calm: int = 3   # max level / asset  (3/6 = 50 %)
    cap_levels_stress: int = 2 # tightened cap in stress (2/6 = 33 %)
    stress_z: float = 1.0      # z-score threshold that flips the cap
    tc_bps: float = 10.0       # linear transaction cost (bps of traded notional)
    tc_quad: float = 0.02      # quadratic surrogate coefficient inside the QUBO
    budget_pen: float = 2.0
    cap_pen: float = 2.0
    esg_min: float = 40.0      # HARD filter: assets below are removed from universe
    esg_tilt: float = 0.002    # SOFT reward per unit of weighted ESG score / 100
    cvar_alpha: float = 0.9
    cvar_rho: float = 0.5      # weight of expected return vs CVaR in the scenario program
    miqp_time_limit: float = 30.0   # Gurobi MIQP time limit (s)
    rl_epochs: int = 120       # PyTorch REINFORCE epochs
    rl_hidden: int = 16
    # ---- QAOA ---------------------------------------------------------
    p_max: int = 3
    shots: int = 2048
    opt_maxiter: int = 60
    adapt_tol: float = 0.01    # stop adding layers if rel. improvement < tol
    # ---- noise (physical) ----------------------------------------------
    p2: float = 3e-3           # two-qubit gate error
    p1: float = 3e-4           # one-qubit gate error
    p_readout: float = 1e-2
    idle_rate: float = 2e-3    # idle error / qubit / layer (phenomenological)
    dd_suppression: float = 0.9
    zne_scales: tuple = (1.0, 2.0, 3.0)
    # ---- logical -------------------------------------------------------
    logical_p_phys: float = 1e-3
    logical_distance: int = 7
    seed: int = 0
    # ---- prediction-system integration (Part 2) ---------------------------
    mu_source: str = "historical"   # expected returns: "historical" (trailing mean, default) | "lstm" | "blend"
    lstm_blend: float = 0.5         # weight of the LSTM forecast when mu_source == "blend"
    risk_source: str = "vix"        # lambda_t driver: "vix" (default) | "vix+fuzzy" (adds the fuzzy risk score)
    kappa_risk: float = 0.5         # strength of the fuzzy-risk term in lambda_t
