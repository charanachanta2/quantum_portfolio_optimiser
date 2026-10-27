# Problem Statement
**PS Name:** Fault-Tolerant Quantum Portfolio Optimizer with Dynamic Constraints

**Objective:** Design and implement a hybrid quantum-classical portfolio optimizer that incorporates logical qubits (surface-code or equivalent simulated error correction), time-varying risk constraints, transaction costs, and ESG filters, then prove (or rigorously bound) a quantum advantage in solution quality or sample complexity under realistic NISQ+ noise models.

**Overview:** Formulate a multi-period stochastic portfolio problem as a time-dependent QUBO/Ising model with soft and hard constraints. Implement a variational algorithm (QAOA or VQE with adaptive ansatz) running on logical qubits via a simulated or real error-corrected backend. Include dynamical decoupling, zero-noise extrapolation, and probabilistic error cancellation. Compare against state-of-the-art classical solvers (Gurobi with stochastic programming, deep RL portfolio agents) on the same out-of-sample periods. Provide theoretical sample-complexity or approximation-ratio bounds.

**Expected Outcome:** Fully reproducible pipeline + 15-20 page technical report containing: (a) logical-qubit resource estimation, (b) ablation of error-mitigation techniques, (c) statistical significance tests of quantum vs classical outperformance, (d) interactive dashboard showing live re-optimization under streaming market data.

**Dataset:** Yahoo Finance + FRED macroeconomic series + public ESG scores (MSCI or Sustainalytics open subsets). Must include at least one crisis period (2008, 2020).

**Major Tech Stack:** Qiskit / Cirq + Stim (surface-code simulation) or Azure Quantum + logical qubit frameworks, Ocean SDK, Gurobi/CPLEX, PyTorch (classical baselines), Streamlit/Gradio.
