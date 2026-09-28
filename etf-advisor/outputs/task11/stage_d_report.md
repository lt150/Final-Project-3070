# Task 11 Stage D report — sensitivity appendix

Executed after the Stage C freeze. Five one-factor-at-a-time deviations from the headline configuration; scope is the **balanced** profile and the three portfolios `one_over_n`, `rb_classical`, `rb_advisor`. All variant artifacts are quarantined to `outputs/task11/appendix_sensitivity/`.

**Stage D feeds nothing back.** No headline number, sentence, exhibit or verdict has been edited in response to anything in this report; the frozen Stage C body and its seven CSVs are untouched. Nothing here participates in any selection. The headline rows reproduced below are read from the frozen `metrics_full_period.csv`, for reference only.

## Gates and validation

### GATE D1 — parallel path validated at the headline window

At window = 250 the parallel implementation must reproduce the registered path. Arithmetic quantities are gated at 1e-12; the weight vectors pass through the registered L-BFGS-B solve and are gated at 1e-06, the solver's terminal-accuracy band — no gate tolerance sits inside it.

| quantity | registered | parallel | abs diff |
|---|---|---|---|
| R[SPY, QQQ] | 0.9438862936732487 | 0.9438862936732487 | 0.000e+00 |
| trace(Σ_classical) | 0.0006854447613844723 | 0.0006854447613844723 | 0.000e+00 |

Balanced weight vectors at 2024-12-02, all eight tickers:

| allocator | max abs elementwise diff | within 1e-6 |
|---|---|---|
| rb_advisor | 0.000e+00 | yes |
| rb_classical | 0.000e+00 | yes |

**GATE D1 PASS** — worst arithmetic |diff| 0.000e+00 ≤ 1e-12; worst weight |diff| 0.000e+00 < 1e-06. The parallel path differs from the registered one in the correlation window and in nothing else; V1 and V2 may run.

### Reuse validation — frozen weights rebuild the frozen paths

Before V4/V5 deviate the cost rate, the reuse machinery is run *at* the headline rate (0.001) from the frozen `grid_weights_target.csv` and `rebalance_turnover_costs.csv`, and checked against the frozen `value_paths_daily.csv`. Worst relative |diff| = 5.142e-14 < 1e-12 across the three in-scope portfolios. **PASS** — V4/V5 differ from the headline in the cost rate and in nothing else.

### Recorded check — cost-invariance of weights

Because the haircut is proportional and uniform, target weights are cost-invariant; V4/V5 rest on it. Re-running the engine end to end at 0.0025 (25 bps) and comparing the target-weight frames against the frozen ones gives worst elementwise |diff| = 9.714e-17 < 1e-12 over both risk-budget allocators × 156 grid dates × 8 tickers. The claim holds exactly. Recorded, not a gate.

### GATE D2 — mechanical accounting, per variant

| variant | worst abs(rate·T − cost) | worst abs(units·P/V_post − w_target) | min T | NaN | result |
|---|---|---|---|---|---|
| V1 | 0.000e+00 | 5.551e-17 | 0.003585 | 0 | PASS |
| V2 | 0.000e+00 | 1.110e-16 | 0.003585 | 0 | PASS |
| V3 | 0.000e+00 | 1.110e-16 | 0.014433 | 0 | PASS |
| V4 | 0.000e+00 | 1.110e-16 | 0.003585 | 0 | PASS |
| V5 | 0.000e+00 | 1.110e-16 | 0.003585 | 0 | PASS |

**GATE D2 PASS** — B3/B4/B5-style accounting re-asserted for all five variants (cost identity ≤ 1e-15, post-cost weights vs target < 1e-12, T ≥ 0, no NaN, path complete over all 3270 trading days).

## Results

### Headline reference (frozen, reproduced — not recomputed)

R = 250 · monthly · 10 bps, read from `metrics_full_period.csv`:

| portfolio | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|
| one_over_n | 7.5% | 10.7% | 0.73 | 1.02 | -23.9% | 2.4% | 2.8 |
| rb_classical:balanced | 4.9% | 6.3% | 0.79 | 1.12 | -20.0% | 17.6% | 21.0 |
| rb_advisor:balanced | 5.0% | 6.2% | 0.82 | 1.16 | -20.0% | 8.4% | 10.0 |

### V1 — R = 60 d

| portfolio | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|
| one_over_n | 7.5% | 10.7% | 0.73 | 1.02 | -23.9% | 2.4% | 2.8 |
| rb_classical:balanced | 4.8% | 6.3% | 0.78 | 1.10 | -20.0% | 18.7% | 22.3 |
| rb_advisor:balanced | 4.9% | 6.2% | 0.80 | 1.13 | -20.0% | 10.6% | 12.7 |

CSV: `outputs/task11/appendix_sensitivity/v1_R60_metrics.csv` (full precision).

### V2 — R = 120 d

| portfolio | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|
| one_over_n | 7.5% | 10.7% | 0.73 | 1.02 | -23.9% | 2.4% | 2.8 |
| rb_classical:balanced | 5.0% | 6.3% | 0.80 | 1.13 | -20.1% | 17.8% | 21.3 |
| rb_advisor:balanced | 5.0% | 6.2% | 0.83 | 1.16 | -20.1% | 9.0% | 10.8 |

CSV: `outputs/task11/appendix_sensitivity/v2_R120_metrics.csv` (full precision).

### V3 — quarterly rebalance

| portfolio | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|
| one_over_n | 7.5% | 10.6% | 0.74 | 1.03 | -24.0% | 4.2% | 1.6 |
| rb_classical:balanced | 4.5% | 6.8% | 0.69 | 0.96 | -20.4% | 21.3% | 8.4 |
| rb_advisor:balanced | 4.8% | 6.4% | 0.77 | 1.07 | -20.1% | 11.6% | 4.5 |

CSV: `outputs/task11/appendix_sensitivity/v3_quarterly_metrics.csv` (full precision).

### V4 — costs 5 bps

| portfolio | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|
| one_over_n | 7.5% | 10.7% | 0.73 | 1.02 | -23.9% | 2.4% | 1.4 |
| rb_classical:balanced | 5.0% | 6.3% | 0.81 | 1.15 | -19.9% | 17.6% | 10.5 |
| rb_advisor:balanced | 5.1% | 6.2% | 0.83 | 1.17 | -19.9% | 8.4% | 5.0 |

CSV: `outputs/task11/appendix_sensitivity/v4_cost5bps_metrics.csv` (full precision).

### V5 — costs 25 bps

| portfolio | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|
| one_over_n | 7.4% | 10.7% | 0.73 | 1.01 | -23.9% | 2.4% | 7.1 |
| rb_classical:balanced | 4.6% | 6.3% | 0.74 | 1.05 | -20.2% | 17.6% | 52.6 |
| rb_advisor:balanced | 4.8% | 6.2% | 0.80 | 1.12 | -20.1% | 8.4% | 25.0 |

CSV: `outputs/task11/appendix_sensitivity/v5_cost25bps_metrics.csv` (full precision).

### Cross-variant summary

The two quantities the sensitivity grid bears on most. Headline = R 250 · monthly · 10 bps.

| metric | portfolio | headline | V1 | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|---|---|
| Sharpe | one_over_n | 0.73 | 0.73 | 0.73 | 0.74 | 0.73 | 0.73 |
| Sharpe | rb_classical:balanced | 0.79 | 0.78 | 0.80 | 0.69 | 0.81 | 0.74 |
| Sharpe | rb_advisor:balanced | 0.82 | 0.80 | 0.83 | 0.77 | 0.83 | 0.80 |
| T̄ | one_over_n | 2.4% | 2.4% | 2.4% | 4.2% | 2.4% | 2.4% |
| T̄ | rb_classical:balanced | 17.6% | 18.7% | 17.8% | 21.3% | 17.6% | 17.6% |
| T̄ | rb_advisor:balanced | 8.4% | 10.6% | 9.0% | 11.6% | 8.4% | 8.4% |

T̄ under V4 and V5 is identical to the headline to the printed precision, which is the cost-invariance of weights showing up directly in the table rather than only in the check above.

## Deliverables

- `outputs/task11/appendix_sensitivity/v1_R60_metrics.csv`
- `outputs/task11/appendix_sensitivity/v2_R120_metrics.csv`
- `outputs/task11/appendix_sensitivity/v3_quarterly_metrics.csv`
- `outputs/task11/appendix_sensitivity/v4_cost5bps_metrics.csv`
- `outputs/task11/appendix_sensitivity/v5_cost25bps_metrics.csv`
- `outputs/task11/stage_d_report.md`

**STOP — Stage D boundary.** Gates D1 and D2 PASS; five variants reported. Headline results remain frozen and unedited. Awaiting task close from planning.