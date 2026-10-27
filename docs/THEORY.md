# Theory: bounds proven for this project's noise model

**Setting.** Cost `E: {0,1}^n -> R` (the QUBO energy), QAOA output distribution `P` on bitstrings,
global optimum `z*`, `Delta = E_max - E_min`. Noise = independent bit-flip with probability `q` per qubit applied to the
measured string (`qpo/noise.py`; `q` is derived from gate error rates, see `NoiseModel.effective_q`).
All statements below are **for this model**; say so in the report. Each one is unit-tested in `tests/test_core.py`.

## P1. Noise lower bound on success probability (rigorous)
For every bitstring `z`: `P_noisy(z) >= P_ideal(z) (1-q)^n`.
*Proof.* With probability `(1-q)^n` no qubit flips and the noisy string equals the ideal one; the remaining branches add non-negative mass. ∎
**Use:** shots-to-optimum for noisy QAOA satisfies `N99 <= ln(0.01) / ln(1 - P_ideal(z*) (1-q)^n)`  (`theory.p_opt_noisy_lower_bound`, `shots_to_success`).
This is an *upper bound on shots needed*; no hardware claim.

## P2. Energy-estimation sample complexity (Hoeffding)
Raw estimator of `<E>` from `N` shots: `N >= Delta^2 ln(2/delta) / (2 eps^2)` gives `|est - <E>| <= eps` w.p. `>= 1-delta`.

## P3. PEC for the flip channel
The inverse of one flip(q) channel is `((1-q) I - q X)/(1-2q)`, 1-norm `1/(1-2q)`. For `n` qubits `gamma = (1-2q)^(-n)`.
The estimator `gamma * mean(sign * E(x'))` is unbiased for the noiseless-circuit-under-measurement distribution; samples lie in
`[-gamma*Emax, gamma*Emax]` so Hoeffding gives `N_PEC = O(gamma^2 Delta^2 ln(1/delta)/eps^2)`  (`theory.pec_shots`).
Exponential in `n q`: `gamma^2 = (1-2q)^(-2n)`. With `q=0.03`: `n=16` gives `gamma^2 = 0.94^(-32) = 7.2`; `n=100` gives `0.94^(-200) = 2.4e5`; `n=120` gives `2.8e6`. (Errata: an earlier version of this file said 6.6 and 3.6e4; both were wrong. Pinned by `tests/test_core.py::test_pec_gamma_reference_values`.) This *is* the scaling limit of mitigation.

## P4. ZNE variance inflation
Richardson extrapolation with scales `s_i` uses weights `c_i` (`sum c_i = 1`, `sum c_i s_i^k = 0`, k=1..m-1). With equal shots per scale the variance is
inflated by `sum c_i^2` (19 for `s = 1,2,3`) and there is residual *bias* from truncating the noise expansion (visible in the ablation).

## P5. Random-sampling baseline
Uniform sampling hits `z*` with probability `2^-n` per shot: `N99 = ln(0.01)/ln(1-2^-n) ≈ 4.6 · 2^n`.
Grover-type search needs `(pi/4) 2^(n/2)` oracle calls but each call is a full cost-function circuit, so it is a *reference line*, not a competitor at NISQ+ depth.

## P6. Noise budget for a sample-complexity advantage
Given a classical solver needing `N_c` evaluations to find `z*` w.p. 0.99, QAOA's guaranteed-by-P1 shots stay below `N_c` only if
`q <= 1 - ( p* / P_ideal(z*) )^(1/n)`, with `p* = 1 - 0.01^(1/N_c)` (`theory.noise_budget_for_advantage`). Because `(1-q)^n ≈ e^{-nq}`, the tolerated `q` shrinks like `1/n`:
this is the quantitative reason logical qubits (`q_L ~ 1e-5`) are required for any advantage at scale.

## What is NOT proven (say this explicitly in the report)
* There is **no known proof** that QAOA at constant depth beats the best classical solvers on QUBO portfolio instances. Our scaling study (`results/scaling.csv`) is empirical evidence for `n <= 16` only.
* At `n <= 20` exhaustive search is trivial; any "advantage" must be a *scaling* statement.
* Related negative results to discuss: noisy circuits lose to trivial classical algorithms beyond a noise-depth threshold (Stilck França & García-Patrón, Nat. Phys. 2021 — verify citation), and the numerical finding that realistic thermal relaxation noise precludes advantage while favourable shot scaling appears only when measurement noise dominates (Yuan et al., arXiv:2410.16265).
* Mean-variance structure is dequantization-prone for some quantum-finance algorithms (Tang-style arguments) — discuss as a threat to validity.
