# Task 12 Stage A report -- engine, imports, closed-form validation

Environment pin: everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0

Explanation method: exact interventional Shapley, full 2^8 coalition enumeration, single-reference background. No sampling, no regression approximation, no shap-library estimator on the production attribution path.

Scope boundary: this task evaluates the faithfulness of the *attributions* -- efficiency, exactness, stability. Narration faithfulness is Task 13's evaluation and appears nowhere in this report.

## Gates

**GATE A1 PASS (with one declared item, below)** -- import map verified by object identity (each symbol used by `src/task12` `is` the attribute of a freshly imported module of record):

| symbol | module of record | purpose |
|---|---|---|
| `allocate` | `src.task10.recommend` | the explained function f(d) = allocate(p, d, t).weights |
| `TICKERS` | `src.task10.recommend` | canonical ticker ordering (task10's own source) |
| `PROFILES` | `src.task9.allocators` | the three registered profile strings |
| `forecast_vol` | `src.task9.forecaster_deploy` | explicand d_forecast(t) -- walk-forward OLS forecast vols |
| `vol_trail_20` | `src.task9.forecaster_deploy` | background d_classical(t) -- the ablation twin's D |
| `fit_coeffs` | `src.task9.forecaster_deploy` | walk-forward per-ticker (a_i, b_i) for the Sec. 1.3 check |
| `ROOT` | `src.data.config` | repo root for output paths |

Static audit of the package source (every `from ..x import y` statement in `src/task12/*.py`, parsed with `ast`; each resolved target was checked against the map above and no unregistered upstream symbol exists):

| file | statement | resolved module |
|---|---|---|
| `fixtures.py` | `from ..data.config import ROOT` | `src.data.config` |
| `fixtures.py` | `from ..task9.allocators import PROFILES` | `src.task9.allocators` |
| `fixtures.py` | `from ..task9.forecaster_deploy import fit_coeffs, forecast_vol, vol_trail_20` | `src.task9.forecaster_deploy` |
| `fixtures.py` | `from ..task10.recommend import TICKERS, allocate` | `src.task10.recommend` |

**Affirmations.**

1. `allocate` is reached only via `src.task10.recommend`; the identity check above is `is`, not equality. No other route into the recommendation layer exists in this package: `src.task9.allocators.solve_risk_budget`, `src.task9.covariance.sigma` / `corr_matrix` and `src.task9.allocators.RISK_BUDGETS` are NOT imported anywhere in `src/task12` -- the covariance assembly, the budget tables and the L-BFGS-B solve are reached only through `allocate`, as Task 10 composed them.
2. `vol_trail_20` and the walk-forward coefficients are reached only via the imported Task 9 code paths: the background d_classical(t) is `vol_trail_20(t)` -- the same callable, with the same 20-day sample-std window and the same returns panel, that `sigma(t, "classical")` calls to build the ablation twin's D -- and the coefficients are `fit_coeffs(t)`, i.e. the walk-forward closed-form OLS with its `vol_label_date <= t` label-realisation cutoff. Neither is restated.
3. No reimplementation of any Task 9 / Task 10 numeric: this package contains no OLS fit, no trailing-vol computation, no correlation, no Sigma assembly, no risk-budget objective / gradient / solver call, no risk-contribution normalisation, and no window arithmetic of any kind. The window/data semantics of every coalition value are entirely those of `allocate` at t.

**New numerics, closed list.**

1. The Shapley enumeration and its coalition weights `s! (n - s - 1)! / n!` -- `engine.py`.
2. Hybrid construction `h_i = x_i if i in S else b_i` -- `engine.py` (selection only; every entry is a bit-identical copy of an entry of x or of b).
3. `dphi` / `dw` / `rho` arithmetic -- `metrics.py`, and the multiplicative probe application in `probes.py`.
4. Summary and exhibit code -- `stage_b.py`, `stage_c.py`, `exhibits.py`, `reporting.py`.
5. Stage D's DeepSHAP driver -- not yet written (see the note below).

**DECLARED ITEM -- the linear map, planning-level.** Gate A1's closed list as written does not name the evaluation of the Stage A explained function itself. Register it verbatim -- `g(x)_i = a_i(t*) + b_i(t*) * x_i` with the coefficients obtained through the imported Task 9 fit path -- and it cannot be reached through any Task 9 callable: `forecast_vol(t)` evaluates exactly this map but fixes x = `vol_trail_20(t)`, whereas the exactness check must evaluate it at 2^8 hybrid inputs, and `src/task9` is read-only so lifting x to a parameter there is out of scope. `fixtures.linear_map` is therefore one expression, `a + b * x`, with (a, b) imported and never retyped. Gate A1 as written asks for an unqualified 'nothing else'; that statement cannot be made without this qualification, so it is reported here rather than absorbed silently. In-repo precedent for the posture: the Task 10 Gate A1 declared exception for `_sigma_from_d`.


**GATE A2 PASS** -- the same enumeration engine, run on the registered toy `f(x1, x2, x3) = 2*x1 + 3*x2*x3` with explicand (1, 1, 1) and background (0, 0, 0); 2^3 = 8 coalitions evaluated once each:

| attribution | exact Shapley value | engine output | abs dev |
|---|---|---|---|
| phi1 | `2.0` | `2.0` | 0.000e+00 |
| phi2 | `1.5` | `1.5` | 0.000e+00 |
| phi3 | `1.5` | `1.5` | 0.000e+00 |

- max abs deviation from the closed form: `0.0` (= 0.000e+00), tolerance 1e-12.
- engine efficiency residual on the toy: `0.0` (= 0.000e+00), tolerance 1e-12; v(N) = `5.0`, v(empty) = `0.0`.
- coalition weights for n = 3: ['0.3333333333333333', '0.16666666666666666', '0.3333333333333333'].

**GATE A3 PASS** -- the same enumeration engine, run on the forecaster map `g(x)_i = a_i(t*) + b_i(t*) * x_i` at the fixed date t* = 2024-12-02; 2^8 = 256 coalitions evaluated once each. Explicand: `vol_trail_20(2024-12-02)`. Background: `vol_trail_20(2020-03-02)` -- calm vs crisis, the maximally separated registered inputs, so the check is non-trivial.

Coefficient provenance: `src.task9.forecaster_deploy.fit_coeffs(2024-12-02)` -> `_fit_coeffs_cached` -> closed-form per-ticker OLS on every sample whose 20-day label window is realised by t*. Imported and used as returned; never retyped.

| ticker | a_i(t*) | b_i(t*) | explicand x_i | background bg_i |
|---|---|---|---|---|
| SPY | `0.0043002861653971556` | `0.533607738585915` | `0.0074866349894696345` | `0.01862563317329489` |
| QQQ | `0.005223015902966071` | `0.5436533509162731` | `0.010309701044789826` | `0.021855054904872568` |
| EFA | `0.004818206182858383` | `0.5300811009665563` | `0.008753980502820082` | `0.013434216542026775` |
| EEM | `0.006426735204731586` | `0.4812142795679762` | `0.010525692366487422` | `0.015436870635731242` |
| AGG | `0.0012761974091441885` | `0.4948797402646781` | `0.0038599220179195122` | `0.0022433367211735` |
| IEF | `0.0017279337056796897` | `0.5604430099079967` | `0.004622490579018556` | `0.004000724306070118` |
| GLD | `0.005079459817691734` | `0.4432423710706346` | `0.013858532247756872` | `0.011563098568362214` |
| VNQ | `0.004834940637330621` | `0.5658827308787332` | `0.010361342318862113` | `0.019277620119825957` |

Full 8x8 attribution matrix at repr precision (rows = input features, columns = output components of g):

| phi[i, j] | SPY | QQQ | EFA | EEM | AGG | IEF | GLD | VNQ |
|---|---|---|---|---|---|---|---|---|
| SPY | `-0.005943855630983606` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| QQQ | `0.0` | `-0.006276670313548123` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| EFA | `0.0` | `0.0` | `-0.0024809046724460354` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| EEM | `0.0` | `0.0` | `0.0` | `-0.0023633291126640655` | `0.0` | `0.0` | `0.0` | `0.0` |
| AGG | `0.0` | `0.0` | `0.0` | `0.0` | `0.0008000153117693641` | `0.0` | `0.0` | `0.0` |
| IEF | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.00034846456147050024` | `0.0` | `0.0` |
| GLD | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.001017433466690283` | `0.0` |
| VNQ | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `-0.005045567631282849` |

Diagonal vs closed form `phi[i, i] = b_i(t*) * (x_i - bg_i)`:

| ticker | closed form | engine phi[i, i] | abs dev |
|---|---|---|---|
| SPY | `-0.0059438556309836105` | `-0.005943855630983606` | 4.337e-18 |
| QQQ | `-0.006276670313548111` | `-0.006276670313548123` | 1.214e-17 |
| EFA | `-0.002480904672446038` | `-0.0024809046724460354` | 2.602e-18 |
| EEM | `-0.002363329112664065` | `-0.0023633291126640655` | 4.337e-19 |
| AGG | `0.0008000153117693641` | `0.0008000153117693641` | 0.000e+00 |
| IEF | `0.0003484645614704998` | `0.00034846456147050024` | 4.337e-19 |
| GLD | `0.0010174334666902792` | `0.001017433466690283` | 3.903e-18 |
| VNQ | `-0.005045567631282846` | `-0.005045567631282849` | 2.602e-18 |

- max abs diagonal deviation: `1.214306433183765e-17` (= 1.214e-17), tolerance 1e-12.
- off-diagonal entries exactly 0.0: 56/56 (nonzero: 0); max abs off-diagonal `0.0`. Each output of g depends on one input, so every off-diagonal marginal difference is a subtraction of bit-identical floats and every weighted sum of those is a sum of exact zeros.
- engine efficiency residual on g (recorded, engine arithmetic): `1.0408340855860843e-17`.

## Tolerance doctrine as applied

| gate | comparison | standard |
|---|---|---|
| A2 | engine output vs a closed-form stub | 1e-12 |
| A2 | efficiency identity over the same cached values | 1e-12 |
| A3 | engine diagonal vs closed form | 1e-12 |
| A3 | engine off-diagonal | exactly 0.0 |

No comparison between two *distinct* L-BFGS-B solves arose in Stage A. None was planned; none appeared; nothing was gated at 1e-6 on that account. Stage A's explained functions are the registered toy and the linear map; neither invokes the solver.

## Deliverables (Stage A)

- `src/task12/__init__.py`, `engine.py`, `fixtures.py`, `probes.py`, `metrics.py`, `exhibits.py`, `reporting.py`, `stage_a.py`, `stage_b.py`, `stage_c.py`
- `outputs/task12/stage_a_report.md`

**STOP -- Stage A boundary.** Gates A1-A3 reported above.