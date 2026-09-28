# Exhibit N1 -- anchor-cell narration with claim spans annotated

Cell: **2024-12-02 / balanced** (the development cell, flagged and outside the headline counts). Evaluated transcript = repeat 1, `outputs/task13/narrations/2024-12-02_balanced_rep1.txt`.

Rendered from the frozen transcript and the frozen grounding CSVs only. ASCII-escaped for the Sec. 8 rule; the raw bytes are the file named above.

## Narration, verbatim

```
For the balanced profile on 2024-12-02, the advisor allocates 14.1% to SPY, 11.7% to QQQ, 6.0% to EFA, 5.5% to EEM, 28.3% to AGG, 23.5% to IEF, 5.8% to GLD and 4.9% to VNQ. AGG is the largest position in the portfolio. The model forecasts a daily volatility of 0.32% for AGG, the lowest in the universe. Compared with the classical benchmark, the advisor overweights AGG and reduces its exposure to SPY. The attributions decompose the advisor-versus-classical difference. The largest single contribution to the AGG weight difference comes from the AGG forecast, at 3.7 percentage points. This explanation is generated automatically and is not financial advice.
```

## Narration, annotated with claim spans (type : outcome)

```
[T6:PASS]For the balanced profile[/T6:PASS] on [T6:PASS]2024-12-02[/T6:PASS], the advisor allocates [T1:PASS]14.1%[/T1:PASS] to SPY, [T1:PASS]11.7%[/T1:PASS] to QQQ, [T1:PASS]6.0%[/T1:PASS] to EFA, [T1:PASS]5.5%[/T1:PASS] to EEM, [T1:PASS]28.3%[/T1:PASS] to AGG, [T1:PASS]23.5%[/T1:PASS] to IEF, [T1:PASS]5.8%[/T1:PASS] to GLD and [T1:PASS]4.9%[/T1:PASS] to VNQ. AGG is the [T5:PASS]largest[/T5:PASS] position in the portfolio. The model forecasts a daily volatility of [T2:PASS]0.32%[/T2:PASS] for AGG, the [T5:PASS]lowest[/T5:PASS] in the universe. Compared with the classical benchmark, the advisor [T4:PASS]overweights[/T4:PASS] AGG and [T4:PASS]reduces[/T4:PASS] its exposure to SPY. The attributions decompose the advisor-versus-classical difference. The [T5:PASS]largest[/T5:PASS] single contribution to the AGG weight difference comes from the AGG forecast, at [T3:PASS]3.7 percentage points[/T3:PASS]. This explanation is generated automatically and is not financial advice.
```

Spans annotated: 17 of 17 claims (0 skipped as partially overlapping another span; every claim appears in the table below regardless).

## Claim table

| # | type | subject | basis | claimed | precision | span | span text | outcome | frozen reference |
|---|---|---|---|---|---|---|---|---|---|
| 0 | T6 | profile | - | `balanced` | - | 0-24 | `For the balanced profile` | PASS | `balanced` |
| 1 | T6 | date | - | `2024-12-02` | - | 28-38 | `2024-12-02` | PASS | `2024-12-02` |
| 2 | T1 | SPY | advisor_weight | `14.1` | pct.1 | 62-67 | `14.1%` | PASS | `0.14109534938031104` |
| 3 | T1 | QQQ | advisor_weight | `11.7` | pct.1 | 76-81 | `11.7%` | PASS | `0.11734152046106551` |
| 4 | T1 | EFA | advisor_weight | `6.0` | pct.1 | 90-94 | `6.0%` | PASS | `0.060453491106364984` |
| 5 | T1 | EEM | advisor_weight | `5.5` | pct.1 | 103-107 | `5.5%` | PASS | `0.05533039347054884` |
| 6 | T1 | AGG | advisor_weight | `28.3` | pct.1 | 116-121 | `28.3%` | PASS | `0.28323468605737295` |
| 7 | T1 | IEF | advisor_weight | `23.5` | pct.1 | 130-135 | `23.5%` | PASS | `0.23508534268634776` |
| 8 | T1 | GLD | advisor_weight | `5.8` | pct.1 | 144-148 | `5.8%` | PASS | `0.05838695772086517` |
| 9 | T1 | VNQ | advisor_weight | `4.9` | pct.1 | 160-164 | `4.9%` | PASS | `0.04907225911712361` |
| 10 | T5 | AGG | largest | `advisor_weights` | - | 184-191 | `largest` | PASS | `{SPY=0.14109534938031104, QQQ=0.11734152046106551, EFA=0.060` |
| 11 | T2 | AGG | forecast_vol | `0.32` | pct.2 | 261-266 | `0.32%` | PASS | `0.003186394614814109` |
| 12 | T5 | AGG | lowest | `forecast_vols` | - | 280-286 | `lowest` | PASS | `{SPY=0.008295212531746234, QQQ=0.010827919422911062, EFA=0.0` |
| 13 | T4 | AGG | w_adv_minus_w_cls | `positive` | - | 355-366 | `overweights` | PASS | `0.03883774741979065` |
| 14 | T4 | SPY | w_adv_minus_w_cls | `negative` | - | 375-382 | `reduces` | PASS | `-0.02231586518295496` |
| 15 | T3 | AGG | AGG | `3.7` | pp.1 | 565-586 | `3.7 percentage points` | PASS | `0.03719255208767154` |
| 16 | T5 | AGG | largest_abs_contribution | `attributions[output=AGG]` | - | 476-483 | `largest` | PASS | `{SPY=0.00410491752539386, QQQ=0.001588970876679392, EFA=0.00` |

## Sentence classification

| # | class | subclass | claims | T8 flagged | T8 violation | sentence |
|---|---|---|---|---|---|---|
| 0 | claim-bearing | - | 10 | False | False | For the balanced profile on 2024-12-02, the advisor allocates 14.1% to SPY, 11.7% to QQQ, 6.0% to EFA, 5.5% to EEM, 28.3% to AGG, 23.5% to IEF, 5.8% to GLD and 4.9% to VNQ. |
| 1 | claim-bearing | - | 1 | False | False | AGG is the largest position in the portfolio. |
| 2 | claim-bearing | - | 2 | False | False | The model forecasts a daily volatility of 0.32% for AGG, the lowest in the universe. |
| 3 | claim-bearing | - | 2 | False | False | Compared with the classical benchmark, the advisor overweights AGG and reduces its exposure to SPY. |
| 4 | no-claim | registered-verbatim | 0 | False | False | The attributions decompose the advisor-versus-classical difference. |
| 5 | claim-bearing | - | 2 | False | False | The largest single contribution to the AGG weight difference comes from the AGG forecast, at 3.7 percentage points. |
| 6 | no-claim | boilerplate | 0 | False | False | This explanation is generated automatically and is not financial advice. |

Non-ASCII characters escaped in this exhibit: 0.
