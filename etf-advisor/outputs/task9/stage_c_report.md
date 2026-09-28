# Task 9 Stage C report — allocators and interface

Four allocators behind one point-in-time interface; risk-budget solver per the pre-registered spec (L-BFGS-B, x0=1/8, bounds [1e-8, inf), gtol=1e-12, maxiter=500; analytic gradient; ftol=1e-18 so the registered gtol governs termination). Smoke test = 4 spot dates x 10 portfolios; not a backtest.

## Gates

**GATE C3 PASS** — rebalance grid derived from the prices.parquet index: first element 2012-01-03, 156 dates 2012-01..2024-12, contains all four smoke dates.
**GATE C1 PASS** — max|RC_i − b_i| < 1e-6 asserted inside every risk-budget solve of the smoke evaluation (24 production solves).
**GATE (variant difference) PASS** — rb_classical and rb_advisor weights differ at every date/profile; max|dw| ranges 0.0171 .. 0.0681.
**Mechanical gates PASS** — all 40 portfolios: |sum(w)−1| < 1e-9, all w ≥ 0.

## Smoke weight tables (10 portfolios per date)

### 2012-02-01

| allocator_id | profile | SPY | QQQ | EFA | EEM | AGG | IEF | GLD | VNQ |
|---|---|---|---|---|---|---|---|---|---|
| one_over_n |  | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 |
| static_bucket | conservative | 0.100000 | 0.100000 | 0.050000 | 0.050000 | 0.300000 | 0.300000 | 0.050000 | 0.050000 |
| static_bucket | balanced | 0.175000 | 0.175000 | 0.100000 | 0.100000 | 0.175000 | 0.175000 | 0.050000 | 0.050000 |
| static_bucket | growth | 0.275000 | 0.275000 | 0.125000 | 0.125000 | 0.050000 | 0.050000 | 0.050000 | 0.050000 |
| rb_classical | conservative | 0.113799 | 0.091245 | 0.022437 | 0.017385 | 0.393563 | 0.288717 | 0.039733 | 0.033121 |
| rb_classical | balanced | 0.132255 | 0.112763 | 0.032165 | 0.026076 | 0.310963 | 0.314042 | 0.043891 | 0.027846 |
| rb_classical | growth | 0.154109 | 0.134072 | 0.037451 | 0.030826 | 0.201327 | 0.365124 | 0.049656 | 0.027435 |
| rb_advisor | conservative | 0.087357 | 0.065839 | 0.021952 | 0.018370 | 0.419938 | 0.315132 | 0.044363 | 0.027050 |
| rb_advisor | balanced | 0.102733 | 0.082334 | 0.031844 | 0.027883 | 0.335752 | 0.346854 | 0.049589 | 0.023012 |
| rb_advisor | growth | 0.121277 | 0.099176 | 0.037563 | 0.033394 | 0.220225 | 0.408558 | 0.056837 | 0.022970 |

### 2020-03-02

| allocator_id | profile | SPY | QQQ | EFA | EEM | AGG | IEF | GLD | VNQ |
|---|---|---|---|---|---|---|---|---|---|
| one_over_n |  | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 |
| static_bucket | conservative | 0.100000 | 0.100000 | 0.050000 | 0.050000 | 0.300000 | 0.300000 | 0.050000 | 0.050000 |
| static_bucket | balanced | 0.175000 | 0.175000 | 0.100000 | 0.100000 | 0.175000 | 0.175000 | 0.050000 | 0.050000 |
| static_bucket | growth | 0.275000 | 0.275000 | 0.125000 | 0.125000 | 0.050000 | 0.050000 | 0.050000 | 0.050000 |
| rb_classical | conservative | 0.042587 | 0.035153 | 0.026066 | 0.027770 | 0.477510 | 0.333480 | 0.038997 | 0.018437 |
| rb_classical | balanced | 0.055713 | 0.046953 | 0.041230 | 0.041456 | 0.402668 | 0.336106 | 0.056116 | 0.019756 |
| rb_classical | growth | 0.072384 | 0.061617 | 0.053279 | 0.052744 | 0.301065 | 0.354173 | 0.081493 | 0.023244 |
| rb_advisor | conservative | 0.053598 | 0.045634 | 0.026555 | 0.027769 | 0.469758 | 0.316366 | 0.039627 | 0.020694 |
| rb_advisor | balanced | 0.069513 | 0.060427 | 0.041641 | 0.041097 | 0.392708 | 0.316103 | 0.056530 | 0.021983 |
| rb_advisor | growth | 0.089385 | 0.078484 | 0.053257 | 0.051750 | 0.290603 | 0.329673 | 0.081250 | 0.025598 |

### 2022-06-01

| allocator_id | profile | SPY | QQQ | EFA | EEM | AGG | IEF | GLD | VNQ |
|---|---|---|---|---|---|---|---|---|---|
| one_over_n |  | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 |
| static_bucket | conservative | 0.100000 | 0.100000 | 0.050000 | 0.050000 | 0.300000 | 0.300000 | 0.050000 | 0.050000 |
| static_bucket | balanced | 0.175000 | 0.175000 | 0.100000 | 0.100000 | 0.175000 | 0.175000 | 0.050000 | 0.050000 |
| static_bucket | growth | 0.275000 | 0.275000 | 0.125000 | 0.125000 | 0.050000 | 0.050000 | 0.050000 | 0.050000 |
| rb_classical | conservative | 0.055368 | 0.040718 | 0.033160 | 0.032785 | 0.357023 | 0.318381 | 0.124724 | 0.037840 |
| rb_classical | balanced | 0.078894 | 0.059335 | 0.055308 | 0.055389 | 0.275113 | 0.274912 | 0.163648 | 0.037400 |
| rb_classical | growth | 0.104752 | 0.079869 | 0.071710 | 0.072692 | 0.187038 | 0.224184 | 0.218085 | 0.041670 |
| rb_advisor | conservative | 0.055396 | 0.041741 | 0.029409 | 0.028195 | 0.425127 | 0.298615 | 0.087904 | 0.033613 |
| rb_advisor | balanced | 0.081338 | 0.062679 | 0.050545 | 0.049085 | 0.337570 | 0.265698 | 0.118851 | 0.034234 |
| rb_advisor | growth | 0.111912 | 0.087428 | 0.067911 | 0.066753 | 0.237819 | 0.224525 | 0.164128 | 0.039525 |

### 2024-12-02

| allocator_id | profile | SPY | QQQ | EFA | EEM | AGG | IEF | GLD | VNQ |
|---|---|---|---|---|---|---|---|---|---|
| one_over_n |  | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 |
| static_bucket | conservative | 0.100000 | 0.100000 | 0.050000 | 0.050000 | 0.300000 | 0.300000 | 0.050000 | 0.050000 |
| static_bucket | balanced | 0.175000 | 0.175000 | 0.100000 | 0.100000 | 0.175000 | 0.175000 | 0.050000 | 0.050000 |
| static_bucket | growth | 0.275000 | 0.275000 | 0.125000 | 0.125000 | 0.050000 | 0.050000 | 0.050000 | 0.050000 |
| rb_classical | conservative | 0.111771 | 0.091020 | 0.034175 | 0.032649 | 0.338753 | 0.302832 | 0.044093 | 0.044708 |
| rb_classical | balanced | 0.163411 | 0.128819 | 0.068276 | 0.063144 | 0.244397 | 0.229572 | 0.049420 | 0.052962 |
| rb_classical | growth | 0.217442 | 0.168714 | 0.096321 | 0.087549 | 0.151845 | 0.152192 | 0.059335 | 0.066601 |
| rb_advisor | conservative | 0.093289 | 0.080146 | 0.029251 | 0.027655 | 0.379495 | 0.299765 | 0.050356 | 0.040043 |
| rb_advisor | balanced | 0.141095 | 0.117342 | 0.060453 | 0.055330 | 0.283235 | 0.235085 | 0.058387 | 0.049072 |
| rb_advisor | growth | 0.194142 | 0.158916 | 0.088190 | 0.079329 | 0.181969 | 0.161155 | 0.072488 | 0.063812 |


- 2012-02-01 rb_classical conservative: bond sleeve 0.682, equity sleeve 0.245.
- 2012-02-01 rb_advisor conservative: bond sleeve 0.735, equity sleeve 0.194.
- 2020-03-02 rb_classical conservative: bond sleeve 0.811, equity sleeve 0.132.
- 2020-03-02 rb_advisor conservative: bond sleeve 0.786, equity sleeve 0.154.
- 2022-06-01 rb_classical conservative: bond sleeve 0.675, equity sleeve 0.162.
- 2022-06-01 rb_advisor conservative: bond sleeve 0.724, equity sleeve 0.155.
- 2024-12-02 rb_classical conservative: bond sleeve 0.642, equity sleeve 0.270.
- 2024-12-02 rb_advisor conservative: bond sleeve 0.679, equity sleeve 0.230.
- Balanced equity sleeve, 2012-02-01 vs 2020-03-02: rb_classical 0.303 -> 0.185, rb_advisor 0.245 -> 0.213.


Gate C2 evidence (scale invariance, all 24 rb solves at the smoke dates):

| date | variant | profile | max_abs_w_diff | pass_1e-06 |
|---|---|---|---|---|
| 2012-02-01 | advisor | conservative | 8.399e-10 | yes |
| 2012-02-01 | advisor | balanced | 1.099e-09 | yes |
| 2012-02-01 | advisor | growth | 2.104e-09 | yes |
| 2012-02-01 | classical | conservative | 1.404e-09 | yes |
| 2012-02-01 | classical | balanced | 3.874e-09 | yes |
| 2012-02-01 | classical | growth | 1.852e-09 | yes |
| 2020-03-02 | advisor | conservative | 3.309e-10 | yes |
| 2020-03-02 | advisor | balanced | 3.582e-09 | yes |
| 2020-03-02 | advisor | growth | 1.850e-09 | yes |
| 2020-03-02 | classical | conservative | 1.990e-09 | yes |
| 2020-03-02 | classical | balanced | 1.853e-09 | yes |
| 2020-03-02 | classical | growth | 1.502e-09 | yes |
| 2022-06-01 | advisor | conservative | 4.109e-09 | yes |
| 2022-06-01 | advisor | balanced | 2.494e-09 | yes |
| 2022-06-01 | advisor | growth | 1.108e-08 | yes |
| 2022-06-01 | classical | conservative | 2.909e-09 | yes |
| 2022-06-01 | classical | balanced | 7.411e-10 | yes |
| 2022-06-01 | classical | growth | 5.234e-09 | yes |
| 2024-12-02 | advisor | conservative | 2.440e-09 | yes |
| 2024-12-02 | advisor | balanced | 6.039e-09 | yes |
| 2024-12-02 | advisor | growth | 4.089e-09 | yes |
| 2024-12-02 | classical | conservative | 1.939e-09 | yes |
| 2024-12-02 | classical | balanced | 5.521e-09 | yes |
| 2024-12-02 | classical | growth | 7.910e-10 | yes |

**GATE C2 PASS** — worst disagreement 1.1076660033637609e-08 (= 1.108e-08) at ('2022-06-01', 'advisor', 'growth'), under the amended threshold 1e-06. Full precision printed for the determinism cross-check against the pre-amendment failure record.

## Checksum record — rb_advisor balanced @ 2024-12-02 (Task 11 anchor, full precision)

- SPY = `0.14109534938031104`
- QQQ = `0.11734152046106551`
- EFA = `0.060453491106364984`
- EEM = `0.05533039347054884`
- AGG = `0.28323468605737295`
- IEF = `0.23508534268634776`
- GLD = `0.05838695772086517`
- VNQ = `0.04907225911712361`

## Deliverables

- `src/task9/allocators.py`
- `outputs/task9/smoke_weights.csv` (40 rows)
- `outputs/task9/stage_c_report.md`

**STOP — Stage C boundary.** Gates C1–C3 and smoke tables reported above.