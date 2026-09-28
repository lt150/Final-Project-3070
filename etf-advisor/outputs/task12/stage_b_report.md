# Task 12 Stage B report -- main grid and tie-outs

Environment pin: everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0

12 registered cells (4 smoke dates x 3 profiles), one exact interventional Shapley enumeration each: 2^8 = 256 coalitions per cell, 3072 evaluations of `allocate` in total, each a distinct L-BFGS-B solve. Explicand d_forecast(t) = `forecast_vol(t)`; background d_classical(t) = `vol_trail_20(t)`, the ablation twin's D. Profile is a conditioning variable, never a feature.

All reference values are read from the on-disk files of record at execution time, never retyped: `smoke_recommendations.csv`, `smoke_weights.csv`, `task9_stage_b_report.md`, `task9_stage_c_report.md`.

## Gates

**GATE B1 PASS** -- v(N) -- the full coalition, every feature taken from the explicand -- at every cell against the 12 rows of `smoke_recommendations.csv`, the frozen rb_advisor recommendation. Stored precision inspected first: the file carries full round-trip precision (every cell satisfies `repr(float(cell)) == cell`), so the **repr-exact branch** applies. All 96 components compared:

| date | profile | components vs record |
|---|---|---|
| 2012-02-01 | conservative | 8/8 match |
| 2012-02-01 | balanced | 8/8 match |
| 2012-02-01 | growth | 8/8 match |
| 2020-03-02 | conservative | 8/8 match |
| 2020-03-02 | balanced | 8/8 match |
| 2020-03-02 | growth | 8/8 match |
| 2022-06-01 | conservative | 8/8 match |
| 2022-06-01 | balanced | 8/8 match |
| 2022-06-01 | growth | 8/8 match |
| 2024-12-02 | conservative | 8/8 match |
| 2024-12-02 | balanced | 8/8 match |
| 2024-12-02 | growth | 8/8 match |

**GATE B2 PASS** -- v(empty) -- the empty coalition, every feature taken from the background -- at every cell against the 12 `rb_classical` rows of `smoke_weights.csv`. This certifies that the registered background is EXACTLY the classical twin's input and hence that the attributions decompose w_advisor - w_classical. Stored precision inspected first: the file carries full round-trip precision (every cell satisfies `repr(float(cell)) == cell`), so the **repr-exact branch** applies. All 96 components compared:

| date | profile | components vs record |
|---|---|---|
| 2012-02-01 | conservative | 8/8 match |
| 2012-02-01 | balanced | 8/8 match |
| 2012-02-01 | growth | 8/8 match |
| 2020-03-02 | conservative | 8/8 match |
| 2020-03-02 | balanced | 8/8 match |
| 2020-03-02 | growth | 8/8 match |
| 2022-06-01 | conservative | 8/8 match |
| 2022-06-01 | balanced | 8/8 match |
| 2022-06-01 | growth | 8/8 match |
| 2024-12-02 | conservative | 8/8 match |
| 2024-12-02 | balanced | 8/8 match |
| 2024-12-02 | growth | 8/8 match |

**GATE B3 PASS** -- efficiency identity `|sum_i phi[i, j] - (v_j(N) - v_j(empty))|` at every cell and every output, tolerance 1e-12. This is linear algebra over the SAME cached coalition values appearing on both sides -- engine arithmetic, not a solve-vs-solve comparison:

| date | profile | worst residual | at output | verdict |
|---|---|---|---|---|
| 2012-02-01 | conservative | 1.388e-17 | QQQ | PASS |
| 2012-02-01 | balanced | 3.469e-17 | SPY | PASS |
| 2012-02-01 | growth | 2.776e-17 | SPY | PASS |
| 2020-03-02 | conservative | 6.939e-18 | QQQ | PASS |
| 2020-03-02 | balanced | 6.939e-18 | AGG | PASS |
| 2020-03-02 | growth | 6.939e-18 | AGG | PASS |
| 2022-06-01 | conservative | 6.939e-17 | AGG | PASS |
| 2022-06-01 | balanced | 1.388e-17 | AGG | PASS |
| 2022-06-01 | growth | 6.939e-18 | EFA | PASS |
| 2024-12-02 | conservative | 1.041e-17 | SPY | PASS |
| 2024-12-02 | balanced | 6.939e-18 | AGG | PASS |
| 2024-12-02 | growth | 1.388e-17 | AGG | PASS |

- worst residual over all 12 x 8 = 96 (cell, output) checks: `6.938893903907228e-17` (= 6.939e-17) at (('2022-06-01', 'conservative'), 'AGG').

## Tolerance doctrine as applied

| gate | comparison | standard |
|---|---|---|
| B1 | identical-trajectory reproduction of the frozen recommendation | repr-exact |
| B2 | identical-trajectory reproduction of the classical twin | repr-exact |
| B3 | efficiency identity over the same cached values | 1e-12 |

Each of the 3072 coalition evaluations is a distinct L-BFGS-B solve, but no two solves are ever compared for agreement inside the attribution arithmetic -- coalition values are data. The only solve-vs-solve agreement checks in this stage are B1 and B2, both identical-trajectory reproductions of a frozen record and therefore repr-exact. No unplanned solve-vs-solve comparison arose; nothing was gated at 1e-6 on that account.

## Deliverables (Stage B)

- `outputs/task12/attributions_main.csv` (768 rows, long format `date, profile, feature, output, phi`, full `repr()` precision)
- `outputs/task12/stage_b_report.md`

**STOP -- Stage B boundary.** Gates B1-B3 reported above. 