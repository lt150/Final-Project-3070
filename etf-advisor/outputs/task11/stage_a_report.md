# Task 11 Stage A report — determinism tie-in

A fresh session reproduces the Task 9 anchors to full precision **before any Task 11 engine code exists**. Acceptance is repr-exactness: a mismatch agreeing to ≤ 1e-6 is still a STOP, because the object under test is cross-session determinism, not numerical adequacy. `src/task9/` is consumed as imported, unmodified.

## Gates

### GATE A1 — pooled and per-ticker train fit

- pooled a = `0.0029325640193698533` — matches registered `0.0029325640193698533` (repr-exact).
- pooled b = `0.6594918209185826` — matches registered `0.6594918209185826` (repr-exact).

Per-ticker cross-check against `outputs/task9/forecaster_trainfit.json` (repr-exact, all eight tickers):

| ticker | a | b | n | repr_match |
|---|---|---|---|---|
| SPY | 0.0044006132785179 | 0.5124582184065051 | 2749 | yes |
| QQQ | 0.005515062385905128 | 0.4884047412334432 | 2749 | yes |
| EFA | 0.004936253430417866 | 0.529086254568326 | 2749 | yes |
| EEM | 0.006581811179233103 | 0.4858783580236274 | 2749 | yes |
| AGG | 0.0015337434807462052 | 0.25989904877955666 | 2749 | yes |
| IEF | 0.0019089957968220885 | 0.4661354646109362 | 2749 | yes |
| GLD | 0.004937652884128399 | 0.466371748505788 | 2749 | yes |
| VNQ | 0.004736119869146286 | 0.5697700908295656 | 2749 | yes |

Max elementwise |diff| vs the frozen artifact: 0.0e+00 

**GATE A1 PASS** — pooled and per-ticker train-fit coefficients reproduce the registered Task 9 anchors repr-exact in this session.

Recorded (not a gate): per-ticker slope range b ∈ [0.2599, 0.5698] — the shrinkage premise of hypothesis H1 (b ∈ [0.26, 0.57]).

### GATE A2 — covariance anchors at t = 2024-12-02

Correlation window = 250 trading days (Task 9 module constant).

- trace(Sigma_classical) = `0.0006854447613844723` — matches registered `0.0006854447613844723` (repr-exact).
- Sigma_classical[SPY, AGG] = `3.883729281316424e-06` — matches registered `3.883729281316424e-06` (repr-exact).
- R[SPY, QQQ] = `0.9438862936732487` — matches registered `0.9438862936732487` (repr-exact).

**GATE A2 PASS** — all three Task 9 Stage B covariance anchors reproduce repr-exact.

### GATE A3 — rb_advisor balanced weights at t = 2024-12-02

Strictest anchor in Stage A: this vector is the output of the registered L-BFGS-B solve, so repr-exactness here asserts an identical solver trajectory across sessions, not merely an equivalent optimum.

- w[SPY] = `0.14109534938031104` — matches registered `0.14109534938031104` (repr-exact).
- w[QQQ] = `0.11734152046106551` — matches registered `0.11734152046106551` (repr-exact).
- w[EFA] = `0.060453491106364984` — matches registered `0.060453491106364984` (repr-exact).
- w[EEM] = `0.05533039347054884` — matches registered `0.05533039347054884` (repr-exact).
- w[AGG] = `0.28323468605737295` — matches registered `0.28323468605737295` (repr-exact).
- w[IEF] = `0.23508534268634776` — matches registered `0.23508534268634776` (repr-exact).
- w[GLD] = `0.05838695772086517` — matches registered `0.05838695772086517` (repr-exact).
- w[VNQ] = `0.04907225911712361` — matches registered `0.04907225911712361` (repr-exact).

|sum(w) − 1| = 1.110e-16; all weights ≥ 0: True.

**GATE A3 PASS** — all eight weights reproduce repr-exact.

### GATE A4 — rebalance grid

- `rebalance_grid()` length = **156** (registered 156).
- first = **2012-01-03** (registered 2012-01-03).
- last = **2024-12-02** (registered 2024-12-02).
- contains all four Task 9 smoke dates ('2012-02-01', '2020-03-02', '2022-06-01', '2024-12-02'): yes.

Recorded verbatim — grid dates 2019-12 through 2020-06 (the exhibit F4 x-axis, seven dates):

  1. `2019-12-02`
  2. `2020-01-02`
  3. `2020-02-03`
  4. `2020-03-02`
  5. `2020-04-01`
  6. `2020-05-01`
  7. `2020-06-01`

Recorded verbatim — the named H1-Q3 grid dates:

- Feb 2020 grid date = **`2020-02-03`** (expected 2020-02-03).
- Mar 2020 grid date = **`2020-03-02`** (expected 2020-03-02).
- Apr 2020 grid date = **`2020-04-01`** (expected 2020-04-01).

**GATE A4 PASS** — grid length, endpoints and smoke-date membership all match the registered values; the F4 window and the H1-Q3 dates are named above and are now fixed for Stage C.

## Environment fingerprint

- platform: `Windows-11-10.0.26200-SP0`
- python: `3.13.0 (tags/v3.13.0:60403a5, Oct  7 2024, 09:38:07) [MSC v.1941 64 bit (AMD64)]`
- executable: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- numpy: `2.4.6`
- pandas: `3.0.3`
- scipy: `1.17.1`
- matplotlib: `3.11.0`

## Registered anchor values

```
A1  pooled a = 0.0029325640193698533
A1  pooled b = 0.6594918209185826
A2  trace(Sigma_classical) @ 2024-12-02 = 0.0006854447613844723
A2  Sigma_classical[SPY, AGG] @ 2024-12-02 = 3.883729281316424e-06
A2  R[SPY, QQQ] @ 2024-12-02             = 0.9438862936732487
A3  rb_advisor balanced SPY @ 2024-12-02 = 0.14109534938031104
A3  rb_advisor balanced QQQ @ 2024-12-02 = 0.11734152046106551
A3  rb_advisor balanced EFA @ 2024-12-02 = 0.060453491106364984
A3  rb_advisor balanced EEM @ 2024-12-02 = 0.05533039347054884
A3  rb_advisor balanced AGG @ 2024-12-02 = 0.28323468605737295
A3  rb_advisor balanced IEF @ 2024-12-02 = 0.23508534268634776
A3  rb_advisor balanced GLD @ 2024-12-02 = 0.05838695772086517
A3  rb_advisor balanced VNQ @ 2024-12-02 = 0.04907225911712361
A4  grid: 156 dates, 2012-01-03 .. 2024-12-02
```

## Deliverables

- `src/task11/__init__.py`, `src/task11/stage_a.py`
- `outputs/task11/stage_a_report.md`

**STOP — Stage A boundary.** Gates A1–A4 all PASS. 