# Task 9 Stage B report — shared covariance module

Sigma = D·R·D, raw daily variance. R: Pearson correlation over the 250 trading days ending at and including t (pre-registered headline constant, not parameterised; {60, 120} sensitivity is Task 11 appendix only). D_advisor = walk-forward forecast vols (Stage A); D_classical = trailing vol_trail_20. R identical for both variants — any advisor edge enters only through D.

## Gates

**GATE B1 PASS** — at all four smoke dates, both variants: max asymmetry < 1e-14, min eigenvalue > 1e-10. No regularisation applied anywhere.

| date | variant | max_asymmetry | min_eigenvalue |
|---|---|---|---|
| 2012-02-01 | advisor | 0.000e+00 | 7.462671e-07 |
| 2012-02-01 | classical | 0.000e+00 | 4.701086e-07 |
| 2020-03-02 | advisor | 0.000e+00 | 1.954271e-07 |
| 2020-03-02 | classical | 0.000e+00 | 2.253410e-07 |
| 2022-06-01 | advisor | 0.000e+00 | 3.390976e-07 |
| 2022-06-01 | classical | 0.000e+00 | 7.549502e-07 |
| 2024-12-02 | advisor | 0.000e+00 | 1.638681e-07 |
| 2024-12-02 | classical | 0.000e+00 | 2.194368e-07 |

**GATE B2 (checksum record)** — anchors at t = 2024-12-02, full precision, recorded (no prior reference exists); fixed reproducibility anchors for Task 11:

- trace(Sigma_classical) = `0.0006854447613844723`
- Sigma_classical[SPY, AGG] = `3.883729281316424e-06`
- R[SPY, QQQ] = `0.9438862936732487`

## Notes

- Sigma is assembled as R_ij * d_i * d_j (elementwise outer-product scaling), which preserves R's exact symmetry; eigenvalues via `numpy.linalg.eigvalsh`.
- Both variants share one `corr_matrix(t)` code path and the Stage A vol functions; no covariance-specific data loading exists.
- Units: raw daily variance throughout; nothing is annualised.

## Deliverables

- `src/task9/covariance.py`
- `outputs/task9/stage_b_report.md`

**STOP — Stage B boundary.** Gates B1–B2 reported above.