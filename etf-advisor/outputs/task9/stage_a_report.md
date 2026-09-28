# Task 9 Stage A report — forecaster deployment


Production forecaster: OLS-per-ticker `y_vol = a_i + b_i * vol_trail_20`, walk-forward refit, raw daily vol units.

## Gates

**GATE A1 PASS** — pooled train fit reproduces the Task 8 Stage C Gate C3 checksum to all printed digits: a=0.00293256 (full: 0.0029325640193698533), b=0.659492 (full: 0.6594918209185826). Fit basis: 21992 train rows (2749 per ticker), Stage C construction.
**GATE A2 PASS** — on-demand vol_trail_20 vs frozen baseline.parquet at 30 random (split, window_end, ticker) points (seed 42): max abs diff 1.102e-16 < 1e-10.
**GATE A3 PASS** — walk-forward ↔ frozen tie-out at derived t* = 2020-12-31: max elementwise |diff| 0.000e+00 < 1e-12. Eligible set at t*: 21992 rows, all split='train': True (train rows: 21992) — exactly the frozen fit basis. Earliest val label realises 2021-02-02.
**GATE A4 (stop-and-report, no acceptance criterion)** — continuity check on the 24 monthly rebalance dates 2023-01-03 .. 2024-12-02 (192 pairs). Frozen forecasts tie back to predictions_test.parquet pred_ols_ticker at all 192 points (max |diff| 7.112e-17 < 1e-10). Walk-forward vs frozen divergence and RMSE tables follow; numbers are documented, no action taken on them.

## Frozen per-ticker coefficients (full precision — Task 11 reproducibility anchors)

| ticker | a | b | n |
|---|---|---|---|
| SPY | 0.0044006132785179 | 0.5124582184065051 | 2749 |
| QQQ | 0.005515062385905128 | 0.4884047412334432 | 2749 |
| EFA | 0.004936253430417866 | 0.529086254568326 | 2749 |
| EEM | 0.006581811179233103 | 0.4858783580236274 | 2749 |
| AGG | 0.0015337434807462052 | 0.25989904877955666 | 2749 |
| IEF | 0.0019089957968220885 | 0.4661354646109362 | 2749 |
| GLD | 0.004937652884128399 | 0.466371748505788 | 2749 |
| VNQ | 0.004736119869146286 | 0.5697700908295656 | 2749 |

Pooled (full precision): a=0.0029325640193698533, b=0.6594918209185826.

## Continuity check (A.4): walk-forward vs frozen, 2023-01-03 .. 2024-12-02

Per ticker and pooled — |walk-forward − frozen| and RMSE vs realised y_vol (192 pairs, all labels realised):

| ticker | mean_abs_wf_minus_frozen | max_abs_wf_minus_frozen | rmse_wf | rmse_frozen |
|---|---|---|---|---|
| SPY | 0.000114624 | 0.000261179 | 0.00190157 | 0.00185251 |
| QQQ | 0.000332065 | 0.000681533 | 0.00210731 | 0.00205842 |
| EFA | 5.12905e-05 | 0.000109338 | 0.00191758 | 0.00193685 |
| EEM | 9.99909e-05 | 0.000219472 | 0.00217501 | 0.00223481 |
| AGG | 0.000591997 | 0.000972065 | 0.00119192 | 0.00164764 |
| IEF | 0.00027352 | 0.000630109 | 0.00137111 | 0.00146121 |
| GLD | 6.46616e-05 | 0.000178732 | 0.00221405 | 0.00222362 |
| VNQ | 8.24403e-05 | 0.000108992 | 0.00260949 | 0.0026051 |
| POOLED | 0.000201324 | 0.000972065 | 0.00198358 | 0.00203087 |

Coefficient drift (walk-forward fits at the first and last continuity dates vs the frozen train fit):

| ticker | a@2023-01-03 | b@2023-01-03 | a@2024-12-02 | b@2024-12-02 | a_frozen | b_frozen |
|---|---|---|---|---|---|---|
| SPY | 0.0043906 | 0.535146 | 0.00430029 | 0.533608 | 0.00440061 | 0.512458 |
| QQQ | 0.00519066 | 0.55008 | 0.00522302 | 0.543653 | 0.00551506 | 0.488405 |
| EFA | 0.00499347 | 0.527421 | 0.00481821 | 0.530081 | 0.00493625 | 0.529086 |
| EEM | 0.00677712 | 0.470285 | 0.00642674 | 0.481214 | 0.00658181 | 0.485878 |
| AGG | 0.00133961 | 0.421188 | 0.0012762 | 0.49488 | 0.00153374 | 0.259899 |
| IEF | 0.00168998 | 0.554361 | 0.00172793 | 0.560443 | 0.001909 | 0.466135 |
| GLD | 0.00506312 | 0.448663 | 0.00507946 | 0.443242 | 0.00493765 | 0.466372 |
| VNQ | 0.00479079 | 0.572008 | 0.00483494 | 0.565883 | 0.00473612 | 0.56977 |

Narrative: the walk-forward coefficients at 2023-01-03 already include the 2021-22 val-period samples (labels realised well before 2023), so they differ from the frozen train fit by construction; through 2024 they continue to absorb realised test-period samples. The divergence documented above is the small, deliberate difference between the deployment mode (expanding walk-forward refit) and Task 8's fixed train fit. Per Gate A4 these numbers are recorded, not acted on.

## Interpretive notes (recorded decisions)

- Fit universe = split rows (train/val/test) of the Task 8 construction; the 896 split='drop' rows (pre-2010 warm-up, boundary straddles) carry no y_vol in any frozen artifact and never enter a fit. 576 of them have vol_label_date <= t*; including them would break the Gate A3 tie-out identity, confirming the exclusion matches the intent.
- Gate A2 sampling: 30 row indices drawn without replacement from the 29712 split rows of baseline.parquet via numpy default_rng(42).
- Walk-forward fits and the frozen fit share one closed-form OLS code path and one row construction, so Gate A3 is an exact-identity check.

## Deliverables

- `src/task9/forecaster_deploy.py`
- `outputs/task9/forecaster_trainfit.json`
- `outputs/task9/continuity_check.csv` (192 rows)
- `outputs/task9/stage_a_report.md`

**STOP — Stage A boundary.** Gates A1–A4 reported above.