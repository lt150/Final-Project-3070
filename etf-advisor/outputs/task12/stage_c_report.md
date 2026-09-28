# Task 12 Stage C report -- stability probes, verdict, exhibits

Environment pin: everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0

Perturbation: eps = 0.01, multiplicative, on the EXPLICAND entry only; the background is a constant of the method and is never perturbed. Each cell runs one base enumeration plus one enumeration per distinct registered probe (p1 belongs to both probe lists and is enumerated once), i.e. 12 x 8 x 256 = 24576 evaluations of `allocate`.

Scope boundary (binding): this stage evaluates the faithfulness of the *attributions*. Narration faithfulness is Task 13's evaluation and appears nowhere in this report, its exhibits or its verdict.

## Gates

**GATE C1 PASS** -- completeness: 96 of 96 registered (cell, probe, pair) combinations computed -- 12 cells x 4 probes for the monitored pair SPY-QQQ and 12 x 4 for the control pair SPY-AGG. No cell or probe was skipped or excluded; no unregistered probe appears.

| check | result |
|---|---|
| rows computed | 96 (expected 96) |
| missing combinations | 0 |
| unregistered combinations | 0 |
| monitored probes | p1, p2, p3, p4 |
| control probes | p1, k1, k2, k3 |
| eps | 0.01 (multiplicative, explicand only) |
| rho denominator floor | 1e-06 |

**GATE C2 PASS** -- the efficiency identity holds inside EVERY probe enumeration: for each of the 12 x 7 = 84 distinct perturbed enumerations, `|sum_i phi'[i, j] - (v'_j(N) - v'_j(empty))| < 1e-12` at every output j. This is engine arithmetic over each probe's own cached values on both sides of its own identity.

| quantity | value |
|---|---|
| enumerations checked | 84 |
| worst residual | `5.551115123125783e-17` (= 5.551e-17) |
| at | 2022-06-01 / conservative / probe p2 |
| tolerance | 1e-12 |

**Right-hand side, stated precisely.** The background is never perturbed -- it is a constant of the method -- so each probe's own empty coalition evaluates `allocate` at the identical background vector, and 'perturbed v(N) minus unperturbed v(empty)' is exactly the probe's own identity as gated above.

**A solve-vs-solve comparison arose and is reported.** Confirming that reading required comparing each probe enumeration's v'(empty) against the base enumeration's v(empty) -- two distinct L-BFGS-B solves. Per the tolerance doctrine such a comparison is gated at 1e-06, never tighter, and its appearance is itself reported. Worst observed gap `0.0` (= 0.000e+00) at 2012-02-01 / conservative / probe p1. This comparison is a recorded observation about the machinery; it is NOT the C2 gate, which is the engine-arithmetic identity above.

**GATE C3 PASS (procedural)** -- rho_max computed over the 48 monitored-pair (cell, probe) combinations and the verdict emitted in the registered Sec. 1.5 vocabulary with brackets filled. The stability OUTCOME is a finding, not a gate: this gate checks the table, the probe set and the vocabulary, not the number.

| quantity | value |
|---|---|
| rho_max (monitored pair, 48 combinations) | `1.1508644016123368` (= 1.15 to 3 s.f.) |
| attained at | 2022-06-01 / conservative / probe p2 |
| registered threshold | 10 |
| stability component | holds |
| efficiency component (Gate B3 + Gate C2) | holds |
| closed-form component (Gate A3) | holds |

## Verdict

> Attribution faithfulness is supported: the exact-Shapley attributions satisfy the efficiency identity to engine precision at every cell, the linear forecaster's attributions equal their closed form, and SPY-QQQ attribution stability holds under every registered perturbation probe (rho_max = 1.15, registered threshold 10).


## Exhibits

- **X1** `X1_attribution_heatmap_anchor.png` -- the 8x8 attribution matrix at the anchor cell 2024-12-02 / balanced, diverging colormap symmetric about zero, every cell also carrying its value as text, with the column sums annotated and labelled `w_advisor - w_classical`.
- **X2** `X2_stability.png` -- rho for ALL 12 cells x 4 monitored probes (48 points) with the control-pair distribution alongside and the threshold line at 10. All cells shown regardless of outcome. Log y axis with `NullFormatter` on the y minor ticks (the DC-1 lesson).

## Tolerance doctrine as applied

| gate | comparison | standard |
|---|---|---|
| C1 | completeness of the registered table | exact set equality |
| C2 | efficiency identity inside each probe enumeration | 1e-12 |
| C2 (reported) | v'(empty) vs v(empty) -- two distinct solves | 1e-06 |
| C3 | vocabulary and probe-set conformance | procedural |
| -- | dphi, dw, rho | measurements, not gates |


## Deliverables (Stage C)

- `outputs/task12/stability_probes.csv` (96 rows, full `repr()` precision)
- `outputs/task12/X1_attribution_heatmap_anchor.png`
- `outputs/task12/X2_stability.png`
- `outputs/task12/stage_c_report.md`

**STOP -- Stage C boundary.** Gates C1-C3 reported above. 