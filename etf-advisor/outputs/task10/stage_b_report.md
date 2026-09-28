# Task 10 Stage B report -- numerical tie-outs and smoke evidence

All reference values are read from the on-disk files of record at execution time, never retyped: `outputs/task9/task9_stage_b_report.md`, `outputs/task9/task9_stage_c_report.md`, `outputs/task9/smoke_weights.csv`.

## Gates

**GATE B1 PASS** -- shared covariance module at t = 2024-12-02, compared repr-exact against the Gate B2 checksum block of `task9_stage_b_report.md` (values read from that file at execution time). Sigma_classical is not on the recommendation path; this gate certifies the shared module is behaviourally intact.

| anchor | recorded (Task 9) | computed (Task 10 session) | verdict |
|---|---|---|---|
| `trace(Sigma_classical)` | `0.0006854447613844723` | `0.0006854447613844723` | MATCH |
| `Sigma_classical[SPY, AGG]` | `3.883729281316424e-06` | `3.883729281316424e-06` | MATCH |
| `R[SPY, QQQ]` | `0.9438862936732487` | `0.9438862936732487` | MATCH |

**GATE B2 PASS** -- `recommend("balanced", "2024-12-02").weights` vs the section "Checksum record -- rb_advisor balanced @ 2024-12-02" of `task9_stage_c_report.md`. Repr-exact per component, through the identical solver trajectory:

| ticker | recorded (Task 9) | computed (Task 10 session) | verdict |
|---|---|---|---|
| SPY | `0.14109534938031104` | `0.14109534938031104` | MATCH |
| QQQ | `0.11734152046106551` | `0.11734152046106551` | MATCH |
| EFA | `0.060453491106364984` | `0.060453491106364984` | MATCH |
| EEM | `0.05533039347054884` | `0.05533039347054884` | MATCH |
| AGG | `0.28323468605737295` | `0.28323468605737295` | MATCH |
| IEF | `0.23508534268634776` | `0.23508534268634776` | MATCH |
| GLD | `0.05838695772086517` | `0.05838695772086517` | MATCH |
| VNQ | `0.04907225911712361` | `0.04907225911712361` | MATCH |

**GATE B3 PASS** -- 12 `recommend` calls (4 smoke dates x 3 profiles) vs the 12 `rb_advisor` rows of `smoke_weights.csv`. Stored precision inspected first: the file carries full round-trip precision (every cell satisfies `repr(float(cell)) == cell`), so the **repr-exact branch** applies. All 96 components compared:

| date | profile | weights vs record |
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

**GATE B4 PASS** -- for all 12 (date, profile) pairs, `recommend(p, t)` vs `allocate(p, forecast_vol(t), t)`, field by field with exact float equality including nested fields, plus whole-dataclass `==`. Same session, same code path -- exact, not toleranced. The identity holds by construction: `recommend` computes the forecast and delegates in a single call, with no parallel numerical path.

| date | profile | fields exact | whole `==` |
|---|---|---|---|
| 2012-02-01 | conservative | 8/8 exact | yes |
| 2012-02-01 | balanced | 8/8 exact | yes |
| 2012-02-01 | growth | 8/8 exact | yes |
| 2020-03-02 | conservative | 8/8 exact | yes |
| 2020-03-02 | balanced | 8/8 exact | yes |
| 2020-03-02 | growth | 8/8 exact | yes |
| 2022-06-01 | conservative | 8/8 exact | yes |
| 2022-06-01 | balanced | 8/8 exact | yes |
| 2022-06-01 | growth | 8/8 exact | yes |
| 2024-12-02 | conservative | 8/8 exact | yes |
| 2024-12-02 | balanced | 8/8 exact | yes |
| 2024-12-02 | growth | 8/8 exact | yes |

**GATE B5 PASS** -- all 12 recommendations: `|sum(weights) - 1|` < 1e-09, `min(weights)` >= 0, `|sum(risk_contributions) - 1|` < 1e-12, `max_abs_rc_minus_budget` < 1e-06 (the inherited Gate C1 convention).

| date | profile | abs_sum_w_minus_1 | min_w | abs_sum_rc_minus_1 | max_abs_rc_minus_budget | verdict |
|---|---|---|---|---|---|---|
| 2012-02-01 | conservative | 0.000e+00 | 0.018370 | 0.000e+00 | 1.189e-09 | PASS |
| 2012-02-01 | balanced | 2.220e-16 | 0.023012 | 2.220e-16 | 5.297e-10 | PASS |
| 2012-02-01 | growth | 0.000e+00 | 0.022970 | 1.110e-16 | 2.746e-09 | PASS |
| 2020-03-02 | conservative | 0.000e+00 | 0.020694 | 0.000e+00 | 2.532e-10 | PASS |
| 2020-03-02 | balanced | 0.000e+00 | 0.021983 | 1.110e-16 | 3.036e-09 | PASS |
| 2020-03-02 | growth | 0.000e+00 | 0.025598 | 0.000e+00 | 1.804e-09 | PASS |
| 2022-06-01 | conservative | 0.000e+00 | 0.028195 | 1.110e-16 | 1.131e-09 | PASS |
| 2022-06-01 | balanced | 0.000e+00 | 0.034234 | 0.000e+00 | 6.841e-10 | PASS |
| 2022-06-01 | growth | 0.000e+00 | 0.039525 | 0.000e+00 | 2.499e-09 | PASS |
| 2024-12-02 | conservative | 0.000e+00 | 0.027655 | 0.000e+00 | 2.385e-10 | PASS |
| 2024-12-02 | balanced | 1.110e-16 | 0.049072 | 0.000e+00 | 5.864e-10 | PASS |
| 2024-12-02 | growth | 0.000e+00 | 0.063812 | 1.110e-16 | 2.508e-09 | PASS |

## Tolerance doctrine as applied

| gate | comparison | standard |
|---|---|---|
| B1 | identical-trajectory reproduction of the covariance anchors | repr-exact |
| B2 | identical-trajectory reproduction of the end-to-end anchor | repr-exact |
| B3 | 12 weight vectors vs a full-round-trip-precision CSV | repr-exact |
| B4 | within-session delegation identity | exact float equality |
| B5 | engine arithmetic (RC sum) | 1e-12 |
| B5 | inherited weight-sum check | 1e-09 |
| B5 | solver-residual convention (Gate C1) | 1e-06 |

No comparison between two distinct L-BFGS-B solves arose.

## Notes

- Gate-only import, not on the recommendation path: `task9.covariance.sigma` is imported inside `gate_b1` (and inside the Gate A1 evidence block) to build Sigma_classical and to certify the amended Sec. 1.5 item 5. The recommendation path itself reaches the covariance module only through `corr_matrix`, per the ratified Gate A1 import map.
- Gate B2's reference section is located by asserting that exactly one `## Checksum record ... rb_advisor balanced` header exists in `task9_stage_c_report.md`, so the verbatim pre-amendment failure record earlier in that file cannot be picked up by accident.
- `recommend` was called once per (date, profile); the same 12 objects feed Gates B2-B5 and the CSV, so the artifact is the evidence, not a second evaluation of it.
- Units: `forecast_vols` and the Sigma behind the weights are raw daily; nothing is annualised. The system is mu-free.
- `max_abs_rc_minus_budget` is recorded per row in the artifact as a health indicator; it acquires no evidentiary status by existing.

## Deliverables (Stage B)

- `outputs/task10/smoke_recommendations.csv` (12 rows, full `repr()` precision: 8 weights, 8 forecast_vols, the residual)
- `outputs/task10/stage_b_report.md`

**STOP -- Stage B boundary.** Gates B1-B5 reported above.