# Task 11 Stage C report — headline backtest (one-shot)

Executed **once**, reported regardless of outcome. No performance gate exists in this stage; the only gates are the three mechanical completion checks, which are accounting rather than performance. H1 and portfolio performance are reported as separate claims and are never traded off against each other.

## Method disclosures (registered, stated once)

1. **Execution timing.** Rebalancing trades are assumed executed at the closing prices that terminate the estimation window — a market-on-close idealisation; no return earned by any weight vector precedes the data used to compute it.
2. **Inception.** At the first grid date (2012-01-03) every portfolio is funded from cash: the traded fraction is T = 1.0 exactly, the cost is one-way 10 bps, and V_post(2012-01-03) = 0.999 identically for all ten portfolios. The inception date is excluded from all per-rebalance turnover statistics — it measures funding, not rebalancing behaviour — leaving 155 turnover observations per portfolio.
3. **Risk-free and MAR.** Sharpe uses rf = 0 and Sortino MAR = 0; no risk-free series exists in the pre-registered universe and adding one post hoc would be a universe change.
4. **Benchmark cost symmetry (§1.6).** Every portfolio, including 1/N and static_bucket, trades to its target on the same monthly grid with the same traded fraction and the same 10 bps: 1/N must trade to remain 1/N.

## Pre-flight and mechanical gates

**Pre-flight PASS (mechanical)** — `stage_a_report.md` records A1–A4 PASS, `stage_b_report.md` records B1–B6 PASS, neither contains a gate failure; the exhibit registration in this module is unchanged (7 CSVs, 4 body figures, 2 appendix figures, 10 portfolios).

**GATE C-M1 PASS** — all 10 value paths defined on every one of the 3270 trading days in 2012-01-03 .. 2024-12-31, no NaN.

**GATE C-M2 PASS** — B3/B4/B5 re-asserted over all 156 grid dates × 10 portfolios (1560 rebalances). Cost identity worst |rate·T − charged| = 0.000e+00 ≤ 1e-15; post-cost weights vs allocator target worst |diff| = 1.110e-16 < 1e-12; min T = 0.003021 ≥ 0; NaN in grid records = 0. Re-implemented in this module rather than imported from `stage_b`.

## Results

### T1 — full-period metrics, all 10 portfolios (2012-01-03 → 2024-12-31)

Net of costs on the daily value path; 3269 return days over 12.97 years. T is the mean traded fraction over the 155 post-inception grid dates.

| portfolio | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|
| one_over_n | 7.5% | 10.7% | 0.73 | 1.02 | -23.9% | 2.4% | 2.8 |
| static_bucket:conservative | 5.5% | 6.8% | 0.82 | 1.15 | -20.8% | 1.9% | 2.3 |
| static_bucket:balanced | 8.2% | 10.2% | 0.82 | 1.15 | -23.9% | 2.2% | 2.7 |
| static_bucket:growth | 11.4% | 14.4% | 0.82 | 1.15 | -27.2% | 1.9% | 2.3 |
| rb_classical:conservative | 3.9% | 5.5% | 0.72 | 1.01 | -19.0% | 15.1% | 18.0 |
| rb_classical:balanced | 4.9% | 6.3% | 0.79 | 1.12 | -20.0% | 17.6% | 21.0 |
| rb_classical:growth | 6.0% | 7.4% | 0.83 | 1.18 | -21.3% | 19.3% | 23.1 |
| rb_advisor:conservative | 3.9% | 5.4% | 0.74 | 1.03 | -19.0% | 7.0% | 8.4 |
| rb_advisor:balanced | 5.0% | 6.2% | 0.82 | 1.16 | -20.0% | 8.4% | 10.0 |
| rb_advisor:growth | 6.2% | 7.3% | 0.87 | 1.23 | -21.3% | 9.6% | 11.4 |

*Hindsight disclosure: the forecaster family was selected on 2021–24 evidence, so pre-2021 exhibits apply a method chosen with hindsight — universal to retrospective backtests, low severity for a 16-parameter regression, dissolved at coefficient level by walk-forward refit.*

### T2 — sub-period metrics, balanced profile

The COVID row reports cumulative return and max drawdown only — an 8-week window is not annualised, and the annualised fields are left empty in `metrics_subperiods.csv` rather than filled, so no unregistered number can be quoted from the artifacts.

| window | portfolio | cumulative return | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|---|---|
| COVID crash | one_over_n | -11.8% | — | — | — | — | -21.3% | — | — |
| COVID crash | static_bucket:balanced | -10.2% | — | — | — | — | -19.3% | — | — |
| COVID crash | rb_classical:balanced | -1.9% | — | — | — | — | -10.0% | — | — |
| COVID crash | rb_advisor:balanced | -2.5% | — | — | — | — | -10.3% | — | — |
| 2022 drawdown | one_over_n | -17.4% | -17.6% | 15.9% | -1.14 | -1.58 | -23.7% | 2.9% | 3.2 |
| 2022 drawdown | static_bucket:balanced | -18.6% | -18.7% | 15.9% | -1.22 | -1.68 | -23.7% | 2.8% | 3.1 |
| 2022 drawdown | rb_classical:balanced | -14.7% | -14.9% | 11.5% | -1.34 | -1.86 | -19.6% | 14.2% | 15.8 |
| 2022 drawdown | rb_advisor:balanced | -14.9% | -15.1% | 11.1% | -1.41 | -1.94 | -19.5% | 8.0% | 8.9 |
| 2023–24 | one_over_n | 30.3% | 14.2% | 9.4% | 1.46 | 2.18 | -8.5% | 2.0% | 2.4 |
| 2023–24 | static_bucket:balanced | 32.0% | 15.0% | 9.1% | 1.59 | 2.40 | -8.4% | 1.9% | 2.2 |
| 2023–24 | rb_classical:balanced | 25.7% | 12.2% | 8.0% | 1.48 | 2.23 | -7.5% | 14.7% | 17.0 |
| 2023–24 | rb_advisor:balanced | 24.0% | 11.4% | 7.6% | 1.45 | 2.18 | -7.4% | 7.4% | 8.5 |

### T3 — H1 quantities, all three profiles

rb_advisor vs rb_classical within profile, on target-vs-drift weights at grid dates; cost-invariant. Weight quantities at the registered 3 dp. 'advisor smoother' records the registered direction (advisor < classical).

| profile | quantity | rb_advisor | rb_classical | advisor − classical | advisor smoother |
|---|---|---|---|---|---|
| conservative | Q1 mean turnover T̄ | 0.070 | 0.151 | -0.081 | yes |
| conservative | Q2 mean max single-asset shift | 0.025 | 0.051 | -0.027 | yes |
| conservative | Q2 full-period max shift | 0.198 | 0.273 | -0.075 | yes |
| conservative | Q3 abs ΔE(Feb→Mar) 2020 | 0.033 | 0.039 | -0.006 | yes |
| balanced | Q1 mean turnover T̄ | 0.084 | 0.176 | -0.092 | yes |
| balanced | Q2 mean max single-asset shift | 0.027 | 0.055 | -0.028 | yes |
| balanced | Q2 full-period max shift | 0.219 | 0.276 | -0.057 | yes |
| balanced | Q3 abs ΔE(Feb→Mar) 2020 | 0.042 | 0.049 | -0.007 | yes |
| growth | Q1 mean turnover T̄ | 0.096 | 0.193 | -0.098 | yes |
| growth | Q2 mean max single-asset shift | 0.030 | 0.058 | -0.028 | yes |
| growth | Q2 full-period max shift | 0.205 | 0.242 | -0.036 | yes |
| growth | Q3 abs ΔE(Feb→Mar) 2020 | 0.052 | 0.062 | -0.010 | yes |

Equity-sleeve levels behind H1-Q3 (grid dates 2020-02-03, 2020-03-02, 2020-04-01). The Feb to Apr leg is descriptive only — recovery dynamics confound its direction and it is not a registered prediction:

| profile | allocator | E(Feb) | E(Mar) | E(Apr) | abs ΔE Feb→Mar (registered) | E(Apr) − E(Feb) (descriptive) |
|---|---|---|---|---|---|---|
| conservative | rb_advisor | 0.186 | 0.154 | 0.150 | 0.033 | -0.037 |
| conservative | rb_classical | 0.171 | 0.132 | 0.148 | 0.039 | -0.023 |
| balanced | rb_advisor | 0.255 | 0.213 | 0.213 | 0.042 | -0.042 |
| balanced | rb_classical | 0.235 | 0.185 | 0.206 | 0.049 | -0.029 |
| growth | rb_advisor | 0.325 | 0.273 | 0.262 | 0.052 | -0.063 |
| growth | rb_classical | 0.302 | 0.240 | 0.247 | 0.062 | -0.054 |

#### H1 verdict

H1 is **supported** on the balanced profile: rb_advisor is the smoother allocator on every registered component (H1-Q1 turnover, H1-Q2 mean allocation shift, H1-Q2 max allocation shift, H1-Q3 crash response).

- Components holding in the registered direction: H1-Q1 turnover, H1-Q2 mean allocation shift, H1-Q2 max allocation shift, H1-Q3 crash response.
- Components running the other way: none.
- Robustness rows (conservative, growth) are in T3 above; the verdict itself is evaluated on the balanced profile.

H1 concerns smoothness, not risk-adjusted performance. The two claims are reported independently and neither is traded off against the other.

## Registered exhibits

| exhibit | file | content |
|---|---|---|
| F1 | `f1_equity_balanced.png` | net-of-cost equity curves, balanced, log y-axis, full span (1/N is profile-less and appears in every profile's figure) |
| F2 | `f2_drawdown_balanced.png` | drawdown curves, balanced, full span |
| F3 | `f3_weights_rb_balanced.png` | stacked-area target-weight trajectories at grid dates, rb_advisor and rb_classical, two panels |
| F4 | `f4_crash_equity_sleeve.png` | equity-sleeve target weight at grid dates 2019-12-02 → 2020-06-01, rb_advisor vs rb_classical |
| A1 | `a1_equity_conservative.png` | appendix — conservative equity curves |
| A2 | `a2_equity_growth.png` | appendix — growth equity curves |

## Appendix — per-profile sub-period tables

### conservative

| window | portfolio | cumulative return | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|---|---|
| COVID crash | one_over_n | -11.8% | — | — | — | — | -21.3% | — | — |
| COVID crash | static_bucket:conservative | -4.6% | — | — | — | — | -12.6% | — | — |
| COVID crash | rb_classical:conservative | -0.6% | — | — | — | — | -9.4% | — | — |
| COVID crash | rb_advisor:conservative | -1.1% | — | — | — | — | -9.7% | — | — |
| 2022 drawdown | one_over_n | -17.4% | -17.6% | 15.9% | -1.14 | -1.58 | -23.7% | 2.9% | 3.2 |
| 2022 drawdown | static_bucket:conservative | -16.2% | -16.3% | 11.6% | -1.48 | -2.02 | -20.3% | 2.4% | 2.7 |
| 2022 drawdown | rb_classical:conservative | -14.2% | -14.3% | 10.0% | -1.49 | -2.03 | -18.5% | 11.7% | 13.0 |
| 2022 drawdown | rb_advisor:conservative | -14.3% | -14.4% | 9.7% | -1.55 | -2.10 | -18.4% | 6.2% | 6.8 |
| 2023–24 | one_over_n | 30.3% | 14.2% | 9.4% | 1.46 | 2.18 | -8.5% | 2.0% | 2.4 |
| 2023–24 | static_bucket:conservative | 20.4% | 9.8% | 7.3% | 1.31 | 1.96 | -7.5% | 1.6% | 1.8 |
| 2023–24 | rb_classical:conservative | 18.4% | 8.9% | 7.1% | 1.23 | 1.84 | -7.2% | 12.7% | 14.6 |
| 2023–24 | rb_advisor:conservative | 16.9% | 8.2% | 7.0% | 1.16 | 1.74 | -7.1% | 6.3% | 7.3 |

### growth

| window | portfolio | cumulative return | ann. return | ann. vol | Sharpe | Sortino | max DD | T̄ | cost drag (bps/yr) |
|---|---|---|---|---|---|---|---|---|---|
| COVID crash | one_over_n | -11.8% | — | — | — | — | -21.3% | — | — |
| COVID crash | static_bucket:growth | -15.5% | — | — | — | — | -27.2% | — | — |
| COVID crash | rb_classical:growth | -3.2% | — | — | — | — | -11.1% | — | — |
| COVID crash | rb_advisor:growth | -3.8% | — | — | — | — | -11.9% | — | — |
| 2022 drawdown | one_over_n | -17.4% | -17.6% | 15.9% | -1.14 | -1.58 | -23.7% | 2.9% | 3.2 |
| 2022 drawdown | static_bucket:growth | -21.4% | -21.6% | 21.5% | -1.02 | -1.42 | -27.1% | 2.3% | 2.6 |
| 2022 drawdown | rb_classical:growth | -15.4% | -15.5% | 13.4% | -1.19 | -1.67 | -21.0% | 15.9% | 17.6 |
| 2022 drawdown | rb_advisor:growth | -15.7% | -15.8% | 13.1% | -1.25 | -1.74 | -21.0% | 9.5% | 10.6 |
| 2023–24 | one_over_n | 30.3% | 14.2% | 9.4% | 1.46 | 2.18 | -8.5% | 2.0% | 2.4 |
| 2023–24 | static_bucket:growth | 47.5% | 21.6% | 11.8% | 1.72 | 2.58 | -9.7% | 1.7% | 2.0 |
| 2023–24 | rb_classical:growth | 34.0% | 15.9% | 9.2% | 1.64 | 2.47 | -8.0% | 15.1% | 17.5 |
| 2023–24 | rb_advisor:growth | 32.5% | 15.2% | 8.9% | 1.64 | 2.47 | -8.0% | 7.8% | 9.0 |


## Freeze

Headline results computed once on 2026-07-27 (2026-07-27T08:56:58+00:00), frozen; no re-run without logged defect and planning ruling.

## Deliverables

- `outputs/task11/value_paths_daily.csv`
- `outputs/task11/grid_weights_target.csv`
- `outputs/task11/grid_weights_drift.csv`
- `outputs/task11/rebalance_turnover_costs.csv`
- `outputs/task11/metrics_full_period.csv`
- `outputs/task11/metrics_subperiods.csv`
- `outputs/task11/h1_quantities.csv`
- `outputs/task11/f1_equity_balanced.png`
- `outputs/task11/f2_drawdown_balanced.png`
- `outputs/task11/f3_weights_rb_balanced.png`
- `outputs/task11/f4_crash_equity_sleeve.png`
- `outputs/task11/a1_equity_conservative.png`
- `outputs/task11/a2_equity_growth.png`
- `outputs/task11/stage_c_report.md`
- `src/task11/metrics.py`, `src/task11/h1.py`, `src/task11/exhibits.py`, `src/task11/stage_c.py`

**STOP — Stage C boundary.** Mechanical gates C-M1 and C-M2 all PASS; results are reported above regardless of direction. 
