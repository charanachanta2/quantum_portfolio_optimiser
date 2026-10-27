# References

## Found via search during project setup (verified existence; read the PDFs before citing details)
1. A. I. Weinberg, *Quantum-Assisted Optimal Rebalancing with Uncorrelated Asset Selection for Algorithmic Trading: Walk-Forward QUBO Scheduling via QAOA*, arXiv:2603.16904. (Walk-forward QUBO rebalancing; classical-simulator QAOA; S&P 500; transaction-cost reduction.)
2. *Quantum Annealing for Dynamic Portfolio Optimization under Realistic Transaction Costs*, arXiv:2607.03218. (Multi-period QUBO with quadratic transaction-cost surrogate; yearly decomposition; compares to CVaR.)
3. H. Yuan, C. K. Long, H. V. Lepage, C. H. W. Barnes, *Quantifying the advantages of applying quantum approximate algorithms to portfolio optimisation*, arXiv:2410.16265. (Noise models, shot-scaling analysis: closest to your "bound the advantage" goal.)
4. *From Portfolio Optimization to Quantum Blockchain and Security: A Systematic Review of Quantum Computing in Finance*, arXiv:2307.01155. (Survey; includes QAOA-MVPO benchmarking and easy/hard instance discussion.)
5. *Constrained Portfolio Optimization via QAOA with (hard-constraint) XY-mixer / Dicke states*, arXiv:2602.14827. (Cardinality constraints, 5–50 % bounds, 5 bps costs.)
6. M. Rosenberg et al., *Solving the Optimal Trading Trajectory Problem Using a Quantum Annealer*, IEEE JSTSP 2016, doi:10.1109/jstsp.2016.2574703. (Original multi-period QUBO with transaction costs / market impact and integer encodings.)
7. C. Gidney, *Stim: a fast stabilizer circuit simulator*, arXiv:2103.02202 (Quantum 2021).
8. K. Temme, S. Bravyi, J. M. Gambetta, *Error mitigation for short-depth quantum circuits*, arXiv:1612.02058 (ZNE + PEC).
9. Riverlane QEC textbook, "Simulating surface codes using stim" (Stim + PyMatching tutorial); NVIDIA CUDA-Q Logical docs (resource-estimation tooling above Stim).

## Standard references I am citing FROM MEMORY — look each one up and verify before putting it in the report
* Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028.
* Rebentrost & Lloyd, *Quantum computational finance: quantum algorithm for portfolio optimization*, arXiv:1811.03975.
* Brandhofer et al., *Benchmarking the performance of portfolio optimization with QAOA*, Quantum Inf. Process. 2023.
* Herman et al., *Quantum computing for finance*, Nat. Rev. Phys. 2023 (arXiv:2201.02773).
* Fowler, Mariantoni, Martinis, Cleland, *Surface codes: Towards practical large-scale quantum computation*, PRA 2012.
* Fowler & Gidney, *Low overhead quantum computation using lattice surgery*, arXiv:1808.06709.
* Litinski, *A Game of Surface Codes*, Quantum 2019; *Magic State Distillation: Not as Costly as You Think*, Quantum 2019 (factory footprints).
* Higgott & Gidney, *PyMatching v2 / Sparse Blossom*, arXiv:2303.15933.
* van den Berg et al., *Probabilistic error cancellation with sparse Pauli-Lindblad models*, Nat. Phys. 2023.
* Stilck França & García-Patrón, *Limitations of optimization algorithms on noisy quantum devices*, Nat. Phys. 2021.
* Chakrabarti et al., *A threshold for quantum advantage in derivative pricing*, Quantum 2021 (template for a finance resource-estimation / advantage-threshold section).
* Bocharov, Roetteler, Svore, *Efficient synthesis of universal repeat-until-success circuits* (T-count per rotation constant used in `qec.synth_t_count`).
* Politis & Romano, *The stationary bootstrap*, JASA 1994 (used in `stats.paired_sharpe_test`); Ledoit & Wolf covariance shrinkage; Rockafellar & Uryasev, *Optimization of CVaR*, 2000.

## Added during the final pass
* N. J. Ross, P. Selinger, *Optimal ancilla-free Clifford+T approximation of z-rotations*, arXiv:1403.2975 - verified: T-count 3 log2(1/eps)+O(log log(1/eps)).
* A. Bocharov, M. Roetteler, K. M. Svore, *Efficient synthesis of universal repeat-until-success quantum circuits*, PRL 114, 080502 (2015) - verified: expected T-count ~1.15 log2(1/eps) (the "+9.2" constant used in the first draft could NOT be verified and is no longer used).
* A. Paetznick, K. M. Svore, *Repeat-until-success*, arXiv:1311.1074 (1.26 log2(1/eps) - 3.53).
* S. Bravyi, A. Kitaev, *Universal quantum computation with ideal Clifford gates and noisy ancillas*, PRA 71, 022316 (2005) - source of the 15-to-1 output error ~35 p^3 (recalled, **not re-verified**).
* FRED, series BAMLH0A0HYM2 and other ICE BofA series: since April 2026 only the last 3 years are published; CSV header is `observation_date`.
* Sustainalytics ESG Risk Ratings (lower = less unmanaged risk), shown on Yahoo Finance since 2018 - availability via yfinance `Ticker.sustainability` changes over time; check before relying on it.
