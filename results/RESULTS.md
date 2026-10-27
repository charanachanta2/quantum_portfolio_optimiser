# Results (auto-generated)

data source: **synthetic**

> **SYNTHETIC DATA - nothing below says anything about real markets.**


## Backtest GFC_2008 (2008-01-01 .. 2009-12-31)

|                      |   ann_return |   ann_vol |   sharpe |   max_drawdown |   cum_return |   avg_turnover |   avg_seconds |
|:---------------------|-------------:|----------:|---------:|---------------:|-------------:|---------------:|--------------:|
| equal_weight_full    |       0.1712 |    0.302  |   0.6736 |        -0.3874 |       0.3873 |         0.0703 |        0      |
| cvar_full            |       0.196  |    0.2493 |   0.8425 |        -0.2567 |       0.4487 |         0.1519 |        0.0105 |
| cvar_sub             |       0.3377 |    0.2819 |   1.1736 |        -0.2671 |       0.8271 |         0.3458 |        0.0077 |
| miqp_sub             |       0.3435 |    0.3076 |   1.1136 |        -0.3277 |       0.8433 |         0.4093 |        0.0006 |
| rl_es_full           |       0.0738 |    0.2821 |   0.393  |        -0.3261 |       0.159  |         0.1228 |        0.0002 |
| exact_qubo           |       0.3435 |    0.3076 |   1.1136 |        -0.3277 |       0.8433 |         0.4093 |        0.0003 |
| sa_qubo              |       0.3355 |    0.3092 |   1.0903 |        -0.332  |       0.8209 |         0.4864 |        0.0107 |
| qaoa_ideal           |       0.316  |    0.3098 |   1.0413 |        -0.3277 |       0.7662 |         0.5266 |        1.9605 |
| qaoa_noisy_raw       |       0.3885 |    0.3137 |   1.2031 |        -0.3324 |       0.9736 |         0.5032 |        1.4499 |
| qaoa_noisy_mitigated |       0.3572 |    0.3112 |   1.1371 |        -0.3306 |       0.8826 |         0.5099 |        0.973  |
| qaoa_logical         |       0.368  |    0.3114 |   1.1619 |        -0.3324 |       0.9138 |         0.472  |        0.8952 |

Significant (Holm-adjusted bootstrap p<0.05) quantum-vs-classical Sharpe differences: 0 / 28  (Wilcoxon, Holm: 0 / 28)


QUBO solution quality per solver (0 gap = optimal):

| method               |   gap_norm |   approx_ratio |   optimum_hit_rate |
|:---------------------|-----------:|---------------:|-------------------:|
| exact_qubo           |    -0      |         1      |               1    |
| miqp_sub             |   nan      |       nan      |             nan    |
| qaoa_ideal           |     0.0016 |         0.9998 |               0.2  |
| qaoa_logical         |     0.0018 |         0.9997 |               0.08 |
| qaoa_noisy_mitigated |     0.0022 |         0.9997 |               0.04 |
| qaoa_noisy_raw       |     0.0018 |         0.9997 |               0.2  |
| sa_qubo              |     0.0029 |         0.9996 |               0.08 |

Fraction of rebalances where the exact QUBO optimum equals the hard-budget MIQP optimum: 1.00


## Backtest COVID_2020 (2019-07-01 .. 2020-12-31)

|                      |   ann_return |   ann_vol |   sharpe |   max_drawdown |   cum_return |   avg_turnover |   avg_seconds |
|:---------------------|-------------:|----------:|---------:|---------------:|-------------:|---------------:|--------------:|
| equal_weight_full    |      -0.1666 |    0.3529 |  -0.3414 |        -0.4663 |      -0.2474 |         0.0759 |        0      |
| cvar_full            |       0.068  |    0.2885 |   0.3708 |        -0.3355 |       0.108  |         0.1613 |        0.0096 |
| cvar_sub             |      -0.0194 |    0.3222 |   0.0992 |        -0.4047 |      -0.0301 |         0.5499 |        0.0072 |
| miqp_sub             |      -0.0114 |    0.3246 |   0.126  |        -0.3906 |      -0.0177 |         0.4891 |        0.0006 |
| rl_es_full           |      -0.1937 |    0.3672 |  -0.4037 |        -0.5049 |      -0.2851 |         0.3769 |        0.0002 |
| exact_qubo           |      -0.0114 |    0.3246 |   0.126  |        -0.3906 |      -0.0177 |         0.4891 |        0.0003 |
| sa_qubo              |      -0.091  |    0.3302 |  -0.125  |        -0.4481 |      -0.1383 |         0.6683 |        0.0104 |
| qaoa_ideal           |      -0.0198 |    0.3263 |   0.1009 |        -0.381  |      -0.0307 |         0.5332 |        1.8983 |
| qaoa_noisy_raw       |      -0.0408 |    0.3303 |   0.0378 |        -0.3978 |      -0.0629 |         0.515  |        1.244  |
| qaoa_noisy_mitigated |      -0.0657 |    0.3292 |  -0.0428 |        -0.4467 |      -0.1005 |         0.5811 |        0.9113 |
| qaoa_logical         |      -0.0552 |    0.3313 |  -0.0068 |        -0.3998 |      -0.0848 |         0.5584 |        0.4713 |

Significant (Holm-adjusted bootstrap p<0.05) quantum-vs-classical Sharpe differences: 0 / 28  (Wilcoxon, Holm: 0 / 28)


QUBO solution quality per solver (0 gap = optimal):

| method               |   gap_norm |   approx_ratio |   optimum_hit_rate |
|:---------------------|-----------:|---------------:|-------------------:|
| exact_qubo           |    -0      |         1      |          1         |
| miqp_sub             |   nan      |       nan      |        nan         |
| qaoa_ideal           |     0.0029 |         0.9996 |          0.157895  |
| qaoa_logical         |     0.0033 |         0.9995 |          0.210526  |
| qaoa_noisy_mitigated |     0.0047 |         0.9993 |          0         |
| qaoa_noisy_raw       |     0.0034 |         0.9995 |          0.157895  |
| sa_qubo              |     0.0063 |         0.9991 |          0.0526316 |

Fraction of rebalances where the exact QUBO optimum equals the hard-budget MIQP optimum: 1.00


## Error-mitigation ablation

| config                     |   est_err |   est_err_sd |   gap_norm |   hit_opt |   p_opt_model |     N99 |   overhead |   q_eff |
|:---------------------------|----------:|-------------:|-----------:|----------:|--------------:|--------:|-----------:|--------:|
| ideal                      |    0.0075 |       0.0043 |     0.0015 |    0.1146 |        0.0001 | 75275.8 |     1      |  0      |
| raw                        |    0.112  |       0.0523 |     0.0017 |    0.1042 |        0.0001 | 88575.4 |     1      |  0.0346 |
| DD                         |    0.112  |       0.054  |     0.0015 |    0.1146 |        0.0001 | 87975.5 |     1      |  0.0332 |
| ZNE                        |    0.0549 |       0.0347 |     0.0014 |    0.125  |        0.0001 | 88575.4 |    57      |  0.0346 |
| DD+ZNE                     |    0.0451 |       0.0248 |     0.0015 |    0.125  |        0.0001 | 87975.5 |    57      |  0.0332 |
| PEC                        |    0.0383 |       0.0156 |     0.0018 |    0.1146 |        0.0001 | 88575.4 |    12.0563 |  0.0346 |
| DD+PEC                     |    0.0381 |       0.0143 |     0.0015 |    0.125  |        0.0001 | 87975.5 |    10.7068 |  0.0332 |
| raw (noise-aware training) |    0.118  |       0.0491 |     0.0017 |    0.1042 |        0.0001 | 88494.9 |     1      |  0.0369 |
| logical (d=7)              |    0.0069 |       0.0033 |     0.0013 |    0.0938 |        0.0001 | 75281.3 |     1      |  0      |

## Shots/evaluations to optimum vs n

|   n |   p_layers |   p_opt_ideal |   N99_qaoa |   N99_qaoa_noisy_LB |   N99_random |   SA_evals_median |   grover_queries |
|----:|-----------:|--------------:|-----------:|--------------------:|-------------:|------------------:|-----------------:|
|   6 |          3 |          0.04 |     107.89 |              122.09 |       292.42 |              1200 |             6.28 |
|   8 |          3 |          0.01 |     413.87 |              486.88 |      1176.62 |              1600 |            12.57 |
|  10 |          3 |          0    |    1506.04 |             1843.73 |      4713.39 |              2000 |            25.13 |
|  12 |          3 |          0    |    3637.35 |             4635.88 |     18860.5  |              2400 |            50.27 |
|  14 |          1 |          0    |   14220    |            18869.1  |     75448.8  |              5600 |           100.53 |
|  16 |          3 |          0    |   43942.8  |            60712.3  |    301802    |             35200 |           201.06 |

Fitted exponents (cost ~ 2^(slope*n)):

|                   |   log2_slope_per_qubit |
|:------------------|-----------------------:|
| N99_qaoa          |                  0.856 |
| N99_qaoa_noisy_LB |                  0.885 |
| N99_random        |                  1.001 |
| SA_evals_median   |                  0.429 |
| grover_queries    |                  0.5   |

## Resource estimate corners

|    | circuit                    |   p_phys | scenario    |   distance |   physical_qubits |   time_per_shot_ms |   distill_levels |   note |
|---:|:---------------------------|---------:|:------------|-----------:|------------------:|-------------------:|-----------------:|-------:|
|  0 | small n=16 (A=4,T=2,p=3)   |    0.001 | optimistic  |          9 |             49232 |         5.122      |                1 |    nan |
|  1 | small n=16 (A=4,T=2,p=3)   |    0.001 | baseline    |         15 |             75576 |       326.618      |                1 |    nan |
|  2 | small n=16 (A=4,T=2,p=3)   |    0.001 | pessimistic |         19 |             88758 |      7394.65       |                1 |    nan |
|  3 | small n=16 (A=4,T=2,p=3)   |    0     | optimistic  |          5 |             15184 |         2.846      |                1 |    nan |
|  4 | small n=16 (A=4,T=2,p=3)   |    0     | baseline    |          7 |             16440 |       152.422      |                1 |    nan |
|  5 | small n=16 (A=4,T=2,p=3)   |    0     | pessimistic |          9 |             19878 |      3502.73       |                1 |    nan |
|  6 | large n=120 (A=20,T=3,p=3) |    0.001 | optimistic  |         15 |            183480 |       541.309      |                1 |    nan |
|  7 | large n=120 (A=20,T=3,p=3) |    0.001 | baseline    |         21 |            539604 |     57990.7        |                2 |    nan |
|  8 | large n=120 (A=20,T=3,p=3) |    0.001 | pessimistic |         27 |           1180620 |         1.9391e+06 |                2 |    nan |
|  9 | large n=120 (A=20,T=3,p=3) |    0     | optimistic  |          7 |             39864 |       252.611      |                1 |    nan |
| 10 | large n=120 (A=20,T=3,p=3) |    0     | baseline    |          9 |             52308 |     12426.6        |                1 |    nan |
| 11 | large n=120 (A=20,T=3,p=3) |    0     | pessimistic |         13 |            146670 |    311250          |                1 |    nan |