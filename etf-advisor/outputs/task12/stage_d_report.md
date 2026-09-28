# Task 12 Stage D report -- LSTM post-mortem (quarantined diagnostic)

Environment pin: everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0

**DIAGNOSTIC EXHIBIT ONLY; FEEDS NOTHING BACK.**

DeepSHAP over the frozen Task 8 LSTM checkpoin. This stage is quarantined: nothing computed here feeds back into the Task 12 verdict, into any Task 8-11 result, or into any chapter claim beyond the single permitted sentence.

## Frozen input schema (read from the arrays, reported before any attribution is computed)

| array | shape | dtype | meaning |
|---|---|---|---|
| `train.npz['X']` | (21992, 30, 6) | float64 | background universe |
| `val.npz['X']` | (3864, 30, 6) | float64 | explicand universe (ALL validation windows) |

The frozen input is **multichannel**: each sample is a 30 x 6 window -- 30 lag positions by 6 input channels. The registered summary is to be aggregated BOTH by lag position AND by input channel. Channel order (the stored column order of every window):

| channel index | name |
|---|---|
| 0 | `ret` |
| 1 | `vol_10` |
| 2 | `mom_5` |
| 3 | `mom_20` |
| 4 | `px_ma20` |
| 5 | `px_ma50` |

Lag convention: index 0 is the OLDEST step in the window and index 29 is the step at `window_end`, i.e. the most recent observation. The frozen architecture reads the last timestep's hidden state (`out[:, -1, :]`), so lag index 29 is the one the head sees directly.

Multichannel: True. Model output explained: the network's own scalar output, i.e. the standardised T1 prediction z before the checkpoint's raw-vol inversion. The inversion is a post-hoc affine map with a floor and is not part of the network.

## Gates

**GATE D1 PASS** -- posture affirmations.

| affirmation | evidence |
|---|---|
| `predictions_test.parquet` not opened | never referenced in `src/task12`; the only data door in this module is `_load_split`, which reads `data/processed/{train,val}.npz` and raises on any other split |
| no test-period windows used anywhere | background drawn from `train.npz`, explicands from `val.npz`; the files actually opened are recorded below |
| files opened by this stage | `C:\Users\tianc\Desktop\Final Project\etf-advisor\data\processed\train.npz`<br>`C:\Users\tianc\Desktop\Final Project\etf-advisor\data\processed\val.npz` |
| checkpoint path | `C:\Users\tianc\Desktop\Final Project\etf-advisor\outputs\task8\models\T1_A1_seed42.pt` |
| checkpoint SHA-256 | `21a6fb501d790830ff2e9167d18ec513a80d13673575b7e81d337c5e7b3adef0` |
| checkpoint metadata | target='T1', arch_id='A1', seed=42, best_epoch=1 |
| output stamp | "diagnostic exhibit only; feeds nothing back." |

No prior SHA-256 of this checkpoint exists anywhere in the repository, so this recording CREATES the anchor, per the Task 9 Gate B2 precedent. It is a record, not a comparison.

Rationale for explicands = validation, not test: the attribution profile is a property of the frozen weights and is visible on any samples; val is already spent for selection, so no further contact with test occurs.

**GATE D2 PASS** -- single dependency install attempt into the venv, followed by pin verification BY IMPORT:

| pinned package | registered version | imported version | verdict |
|---|---|---|---|
| numpy | 2.4.6 | 2.4.6 | UNDISTURBED |
| pandas | 3.0.3 | 3.0.3 | UNDISTURBED |
| scipy | 1.17.1 | 1.17.1 | UNDISTURBED |
| pyarrow | 25.0.0 | 25.0.0 | UNDISTURBED |

- installed this stage: shap 0.52.0, numba 0.66.0, llvmlite 0.48.0, plus `slicer`, `cloudpickle`, `tqdm` (pure additions).
- torch 2.12.0+cpu, CPU execution, `torch.manual_seed(42)`.
- The install added packages only: no pinned package was upgraded, downgraded or removed. `shap` declares `numpy>=2`, which the registered numpy 2.4.6 already satisfies.

**Registered computation.** Background: 100 training windows drawn WITHOUT replacement via `numpy.random.default_rng(42)` (the Task 9 Gate A2 sampling convention). Explicands: ALL 3864 validation windows -- deterministic, no sampling choice. CPU execution, `torch.manual_seed(42)`.

**Warnings raised by the explainer, recorded verbatim:**

- `UserWarning: unrecognized nn.Module: LSTM`

**GATE D3 FAIL** -- local accuracy: DeepSHAP attribution sums against `f(x) - mean over background of f`, per explicand. **This is a float32-class check, not an engine-precision gate** -- the network is float32, so the registered threshold is 0.001 RELATIVE.

| quantity | value |
|---|---|
| explicands checked | 3864 |
| worst relative deviation | `2378.6995032506907` (= 2379) |
| registered threshold | 0.001 relative |
| at explicand index | 234 |
|   its sum(phi) | `-0.21269593651481955` |
|   its f(x) - mean f(bg) | `-8.937932550907163e-05` |
| median relative deviation | 0.5481 |
| fraction of explicands over threshold | 0.9995 |
| worst absolute deviation | 1.79664 |
| mean f over background | `-0.013484919145703315` |

## Diagnosis (Gate D3 FAIL -- halt)

The failure is structural, not marginal. `shap` 0.52.0's PyTorch `DeepExplainer` walks the module graph and dispatches each module to an op handler; the frozen architecture's `nn.LSTM` is a fused recurrent module for which that handler table has **no entry**, which is exactly what the recorded `unrecognized nn.Module: LSTM` warning reports. The explainer does not fail loudly on the unknown module -- it falls through to a default treatment that does not implement the rescale/reveal-cancel rule for a recurrent cell, so the returned values are not DeepSHAP attributions for this network and do not decompose its output. The measured worst relative deviation above is orders of magnitude beyond the registered 0.001; a float32 rounding story cannot account for it.

**Nothing is patched forward.** No appendix deliverable is written on a failed stage -- publishing a by-lag profile computed from values that fail local accuracy would present an artifact of the explainer as a property of the model.

**Scope of the failure, stated precisely.** This is a finding about the DeepSHAP method's applicability to a fused LSTM under the pinned library, not about the frozen model and not about Stages A-C. The Stage C verdict is untouched: Stages A-C use the own-implementation exact enumeration, share no code with this stage, and never invoke the shap library.

**GATE FAILURE -- STOP**: GATE D3: worst relative deviation exceeds 0.001 -- DeepSHAP does not satisfy local accuracy on the frozen LSTM (unrecognized nn.Module: LSTM) -- STOP

Diagnosis recorded above; nothing patched. No appendix deliverable is written on a failed stage.

**STOP -- Stage D boundary.**