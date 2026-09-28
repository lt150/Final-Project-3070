# Task 12 Stage D2 report -- LSTM post-mortem by Integrated Gradients

Environment pin: everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0

**DIAGNOSTIC EXHIBIT ONLY; FEEDS NOTHING BACK.**

Method: Integrated Gradients (amendment T12-A3; DeepSHAP inapplicable -- see the Stage D failure record). Registered after the Stage D measurement established that shap 0.52.0's PyTorch `DeepExplainer` has no op handler for the fused `nn.LSTM` and fails local accuracy structurally. `stage_d_report.md` is that failure record and is NOT modified by this stage; `stage_d.py` carries an overwrite guard refusing to regenerate it.

Quarantine unchanged: nothing computed here feeds back into the Task 12 verdict, into any Task 8-11 result, or into any chapter claim beyond the single permitted sentence.

## Gates

**GATE D2-1 PASS** -- posture affirmations.

| affirmation | evidence |
|---|---|
| `predictions_test.parquet` not opened | never referenced in `src/task12`; this stage IMPORTS Stage D's `_load_split`, the structural data door that raises on any split but train/val -- the firewall is retained, not restated |
| no test-period windows used anywhere | `C:\Users\tianc\Desktop\Final Project\etf-advisor\data\processed\train.npz`<br>`C:\Users\tianc\Desktop\Final Project\etf-advisor\data\processed\val.npz` |
| checkpoint path | `C:\Users\tianc\Desktop\Final Project\etf-advisor\outputs\task8\models\T1_A1_seed42.pt` |
| checkpoint SHA-256 (recomputed) | `21a6fb501d790830ff2e9167d18ec513a80d13673575b7e81d337c5e7b3adef0` |
| Stage D anchor (read from `stage_d_report.md`) | `21a6fb501d790830ff2e9167d18ec513a80d13673575b7e81d337c5e7b3adef0` |
| anchor comparison | **MATCH** |
| checkpoint metadata | target='T1', arch_id='A1', seed=42, best_epoch=1 |
| model eval mode asserted | `model.training` = False (False required) |
| parameter grads disabled | 0 of 6 parameters require grad (0 required: gradients are taken w.r.t. the INPUT only) |
| background index set identical to Stage D's | True -- same `numpy.random.default_rng(42)`, same size 100, same population; reproduced with a fresh generator and compared |
| output stamp | "diagnostic exhibit only; feeds nothing back." |

This is the checkpoint hash's FIRST USE AS A COMPARISON. Stage D created the anchor (no prior hash existed anywhere in the repository); Stage D2 reads it back from the file of record at execution time and compares, so the two stages are certified to have explained the same frozen weights.

**GATE D2-2 PASS** -- environment: NO new installs were made for this stage. The four registered pins verified by import:

| pinned package | registered version | imported version | verdict |
|---|---|---|---|
| numpy | 2.4.6 | 2.4.6 | UNDISTURBED |
| pandas | 3.0.3 | 3.0.3 | UNDISTURBED |
| scipy | 1.17.1 | 1.17.1 | UNDISTURBED |
| pyarrow | 25.0.0 | 25.0.0 | UNDISTURBED |

- torch 2.12.0+cpu, CPU execution, `torch.manual_seed(42)`.
- **The shap library is not imported, referenced or used in this module.** Integrated Gradients is computed with `torch.autograd.grad` alone. shap remains installed from Stage D's ratified attempt but plays no part here.

**GATE D2-3 PASS** -- completeness: per explicand, `|sum(IG) - (f(x) - f(x_baseline))|`, gated **ABSOLUTE** at 0.001 on the standardized output scale (amended criterion, registered by T12-A3 BEFORE any IG value existed):

| quantity | value |
|---|---|
| explicands checked | 3864 |
| worst ABSOLUTE deviation (gated) | `3.2988776895503946e-05` (= 3.299e-05) |
| registered threshold | 0.001 absolute |
| at explicand index | 2741 |
|   its sum(IG) | `2.3878493136069507` |
|   its f(x) - f(x_baseline) | `2.3878163248300552` |
| median ABSOLUTE deviation | 1.64e-06 |
| fraction over the absolute threshold | 0.0000 |
| worst RELATIVE deviation (descriptive, NOT gated) | 0.001409 |
| median RELATIVE deviation (descriptive, NOT gated) | 3.396e-06 |
| smallest |f(x) - f(x_baseline)| over explicands | 0.0004947 |
| f(x_baseline) | `-0.08964718878269196` |

**Why absolute, recorded.** Stage D's pure-relative criterion was pathological under near-zero denominators -- its worst case had `|f(x) - mean f(bg)| = 8.937932550907163e-05`, so the relative form partly measured denominator size rather than attribution error. The explained output is the standardized z, which is O(1), so an absolute 0.001 is the meaningful float32-class criterion here. The relative figures are reported above for continuity with the Stage D record; they are descriptive and carry no gate.

## Registered summary -- computed, not pre-committed

Mean absolute IG attribution aggregated by lag position and by input channel, over all 3864 validation explicands. The frozen schema is multichannel (30 lags x 6 channels), so both aggregations are registered and both are reported.

| lag | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean abs IG | 1.130e-04 | 1.427e-04 | 1.759e-04 | 2.184e-04 | 2.592e-04 | 3.130e-04 | 3.757e-04 | 4.519e-04 | 5.378e-04 | 6.664e-04 | 7.826e-04 | 9.373e-04 | 1.135e-03 | 1.355e-03 | 1.622e-03 | 1.948e-03 | 2.338e-03 | 2.825e-03 | 3.424e-03 | 4.140e-03 | 5.032e-03 | 6.180e-03 | 7.498e-03 | 9.157e-03 | 1.118e-02 | 1.380e-02 | 1.671e-02 | 2.037e-02 | 2.455e-02 | 3.158e-02 |
| share | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.003 | 0.003 | 0.004 | 0.005 | 0.006 | 0.007 | 0.008 | 0.010 | 0.011 | 0.014 | 0.017 | 0.020 | 0.024 | 0.030 | 0.036 | 0.044 | 0.054 | 0.066 | 0.081 | 0.098 | 0.120 | 0.145 | 0.186 |

| channel | ret | vol_10 | mom_5 | mom_20 | px_ma20 | px_ma50 |
|---|---|---|---|---|---|---|
| mean abs IG | 4.862e-03 | 1.609e-02 | 3.563e-03 | 2.926e-03 | 4.028e-03 | 2.492e-03 |
| share | 0.143 | 0.474 | 0.105 | 0.086 | 0.119 | 0.073 |

| scope | dominant | mean abs IG | share of total |
|---|---|---|---|
| lag position | lag 29 (window_end) | 0.0315798 | 0.1860 |
| input channel | `vol_10` | 0.0160908 | 0.4738 |

- Concentration, recorded: the most recent 8 of 30 lag positions carry 0.7940 of the total mean absolute attribution; a flat profile would put 0.2667 there.
- Dominant lag is index 29 of 29; dominant channel is `vol_10`.

## Deliverables (Stage D2, appendix placement)

- `C:\Users\tianc\Desktop\Final Project\etf-advisor\outputs\task12\appendix_lstm\attribution_by_lag.csv`
- `C:\Users\tianc\Desktop\Final Project\etf-advisor\outputs\task12\appendix_lstm\attribution_by_lag.png` (two panels: by lag position, by input channel)
- `outputs/task12/stage_d2_report.md`

Method label carried on every deliverable: "Integrated Gradients (amendment T12-A3; DeepSHAP inapplicable -- see the Stage D failure record)".

**STOP -- Stage D2 boundary. Task 12 execution complete.**