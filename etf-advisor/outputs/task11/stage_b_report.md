# Task 11 Stage B report — engine and accounting on fixtures

This stage proves the accounting on two registered fixtures, against an independently implemented reconciliation path; `src/task11/reconcile.py` implements the same accounting by daily compounding with a one-day weight-drift recursion. **No full-period run happens in this stage.** Engine-side tolerances are registered values (1e-12 elementwise / 1e-10 relative); the one quantity that passes through the registered L-BFGS-B solve (Gate B6) is matched at printed precision instead, never inside the solver band.

Headline cost rate in force: 0.001 (10 bps one-way, §0.1).

## Gates

### GATE B1 — Fixture 1 (synthetic, exact expressions)

Prices A (100, 110, 121) · B (100, 100, 100) · C (100, 90, 81); targets w(D0) = (0.5, 0.3, 0.2), w(D1) = (0.4, 0.4, 0.2). Expected values are held as `fractions.Fraction` and converted to float only at the comparison. The registered expressions were additionally re-derived from the fixture inputs through the §1.2 formulae and found exactly equal as rationals (asserted inside `fixture1_expected`), so the fixture is self-consistent before the gate is evaluated.

Grid reading: {D0, D1}, with D2 the terminal valuation day — no w(D2) and registers V(D2) as the un-haircut mark of the units bought at D1. See `fixtures.py`.

| quantity | engine | registered (exact -> float) | abs diff |
|---|---|---|---|
| T(D0) | 1.0 | 1.0 | 0.000e+00 |
| cost(D0) | 0.001 | 0.001 | 0.000e+00 |
| V_post(D0) | 0.999 | 0.999 | 0.000e+00 |
| V(D0) on path | 0.999 | 0.999 | 0.000e+00 |
| w_drift(D1)[A] | 0.5339805825242718 | 0.5339805825242718 | 0.000e+00 |
| w_drift(D1)[B] | 0.2912621359223301 | 0.2912621359223301 | 0.000e+00 |
| w_drift(D1)[C] | 0.17475728155339806 | 0.17475728155339806 | 0.000e+00 |
| T(D1) | 0.2679611650485437 | 0.26796116504854367 | 5.551e-17 |
| cost(D1) | 0.0002679611650485437 | 0.00026796116504854367 | 5.421e-20 |
| V_pre(D1) | 1.02897 | 1.02897 | 0.000e+00 |
| V_post(D1) | 1.028694276 | 1.028694276 | 0.000e+00 |
| V(D1) on path | 1.028694276 | 1.028694276 | 0.000e+00 |
| V(D2) | 1.0492681615200001 | 1.04926816152 | 2.220e-16 |

**GATE B1 PASS** — all 13 registered quantities reproduced; worst |diff| = 2.220e-16 (at V(D2)) < 1e-12.

### GATE B2 — independent reconciliation, Fixture 2

Engine: fixed unit counts marked at adjusted closes, drift from one two-point price ratio. Reconciliation: no units and no price level anywhere — daily compounding of the portfolio return with a one-day weight-drift recursion, so a grid-date drift vector is a product of ~21 daily steps. Same accounting, disjoint arithmetic.

| portfolio | max rel ΔV (daily path) | max abs Δw_drift |
|---|---|---|
| one_over_n | 1.453e-15 | 1.388e-16 |
| static_bucket:balanced | 8.559e-16 | 1.665e-16 |
| rb_classical:balanced | 1.083e-15 | 2.220e-16 |
| rb_advisor:balanced | 2.165e-15 | 3.331e-16 |

**GATE B2 PASS** — worst relative |ΔV| = 2.165e-15 < 1e-10; worst |Δw_drift| = 3.331e-16 < 1e-12, across all four portfolios.

### GATE B3 — cost identity

**GATE B3 PASS** — worst |rate·T − charged cost| = 0.000e+00 ≤ 1e-15 over all rebalances and portfolios (bit-exact identity at every rebalance, so no worst case exists).

### GATE B4 — post-cost weights equal the allocator target

Recomputed from the holdings actually recorded (units · P / V_post), not from the stored target frame, so the check has content.

**GATE B4 PASS** — worst |units·P/V_post − w_target| = 5.551e-17 < 1e-12 (worst at rb_classical:balanced).

### GATE B5 — completeness, no NaN, T ≥ 0

| portfolio | value days | span days | path complete | NaN in weights/turnover/costs | min T | max T |
|---|---|---|---|---|---|---|
| one_over_n | 126 | 126 | yes | 0 | 0.018149 | 1.000000 |
| static_bucket:balanced | 126 | 126 | yes | 0 | 0.016648 | 1.000000 |
| rb_classical:balanced | 126 | 126 | yes | 0 | 0.038494 | 1.000000 |
| rb_advisor:balanced | 126 | 126 | yes | 0 | 0.022710 | 1.000000 |

**GATE B5 PASS** — every portfolio's value path covers all 126 trading days in the window with no NaN anywhere in values, weights, turnover or costs; min T over all grid dates and portfolios = 0.016648 ≥ 0.

### Registered closed-form check — 1/N drift

With equal-weight targets, w_drift at each grid date is exactly the per-ticker gross return since the previous grid date, normalised. Checked at all 6 post-inception grid dates: worst |diff| = 0.000e+00 < 1e-12. PASS.

### GATE B6 — Task 9 Stage C smoke tie-out at 2012-02-01 (6 dp)

Solver-band gate: this quantity passes through the registered L-BFGS-B solve, so it is matched at the Task 9 report's printed precision rather than at repr level.

| allocator | ticker | engine (6 dp) | Task 9 smoke (6 dp) | match | abs diff |
|---|---|---|---|---|---|
| rb_classical | SPY | 0.132255 | 0.132255 | yes | 2.776e-17 |
| rb_classical | QQQ | 0.112763 | 0.112763 | yes | 2.776e-17 |
| rb_classical | EFA | 0.032165 | 0.032165 | yes | 1.388e-17 |
| rb_classical | EEM | 0.026076 | 0.026076 | yes | 0.000e+00 |
| rb_classical | AGG | 0.310963 | 0.310963 | yes | 5.551e-17 |
| rb_classical | IEF | 0.314042 | 0.314042 | yes | 0.000e+00 |
| rb_classical | GLD | 0.043891 | 0.043891 | yes | 1.388e-17 |
| rb_classical | VNQ | 0.027846 | 0.027846 | yes | 3.816e-17 |
| rb_advisor | SPY | 0.102733 | 0.102733 | yes | 0.000e+00 |
| rb_advisor | QQQ | 0.082334 | 0.082334 | yes | 6.939e-17 |
| rb_advisor | EFA | 0.031844 | 0.031844 | yes | 6.939e-18 |
| rb_advisor | EEM | 0.027883 | 0.027883 | yes | 6.939e-18 |
| rb_advisor | AGG | 0.335752 | 0.335752 | yes | 5.551e-17 |
| rb_advisor | IEF | 0.346854 | 0.346854 | yes | 5.551e-17 |
| rb_advisor | GLD | 0.049589 | 0.049589 | yes | 4.857e-17 |
| rb_advisor | VNQ | 0.023012 | 0.023012 | yes | 2.776e-17 |

**GATE B6 PASS** — all 16 weights (rb_classical and rb_advisor, balanced) agree with the Task 9 Stage C smoke table to 6 dp; the engine consumes the allocator interface without perturbing it.

## Fixture 2 tables (real data, 2012-01-03 .. 2012-07-02, 7 grid dates, 126 trading days)

Per-rebalance turnover T (traded fraction; inception = 1.0, excluded from turnover statistics in Stage C):

| date | one_over_n | static_bucket:balanced | rb_classical:balanced | rb_advisor:balanced |
|---|---|---|---|---|
| 2012-01-03 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2012-02-01 | 0.026953 | 0.026138 | 0.26438 | 0.085958 |
| 2012-03-01 | 0.028362 | 0.027161 | 0.151035 | 0.053188 |
| 2012-04-02 | 0.027156 | 0.025865 | 0.125148 | 0.028192 |
| 2012-05-01 | 0.018149 | 0.016648 | 0.147702 | 0.044883 |
| 2012-06-01 | 0.058835 | 0.064434 | 0.080203 | 0.02271 |
| 2012-07-02 | 0.040828 | 0.038909 | 0.038494 | 0.05104 |

Terminal net value at 2012-07-02 (all paths funded at 1.0 and haircut to 0.999 at inception):

| portfolio | V(inception) | V(end) |
|---|---|---|
| one_over_n | 0.999 | 1.0587845289343112 |
| static_bucket:balanced | 0.999 | 1.0618641430971032 |
| rb_classical:balanced | 0.999 | 1.046486960017306 |
| rb_advisor:balanced | 0.999 | 1.045789127275129 |

Inception values are bit-identical across portfolios: **True**; common value `0.999`.

## Deliverables

- `src/task11/stage_b.py`
- `outputs/task11/stage_b_report.md`

**STOP — Stage B boundary.** Gates B1, B2, B3, B4, B5, B6 all PASS. 