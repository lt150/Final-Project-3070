# Task 10 Stage A report -- recommendation-layer module, contract, validation

Deliverable module: `src/task10/recommend.py`. Exposes `allocate(profile, d_vector, as_of)` and `recommend(profile, as_of)`, both returning a frozen `Recommendation`. Allocator exposed: `rb_advisor` only (Sec. A7). Units: raw daily vol / raw daily variance; nothing annualised; the system is mu-free.

## Gates

**GATE A1 PASS (with one declared exception, below)** -- import map verified by object identity (each local symbol `is` the attribute of a freshly imported module of record):

| symbol | module of record | purpose |
|---|---|---|
| `UNIVERSE` | `src.data.config` | canonical ticker ordering (task9's own source) |
| `ROOT` | `src.data.config` | repo root for output paths |
| `PROFILES` | `src.task9.allocators` | the three registered profile strings |
| `RISK_BUDGETS` | `src.task9.allocators` | registered risk-budget tables |
| `solve_risk_budget` | `src.task9.allocators` | registered objective/gradient/L-BFGS-B solve |
| `CORR_WINDOW` | `src.task9.covariance` | 250-day correlation window constant |
| `corr_matrix` | `src.task9.covariance` | Pearson R over the 250 days ending at t |
| `forecast_vol` | `src.task9.forecaster_deploy` | walk-forward OLS forecast vols at t |
| `_prices_index` | `src.task9.forecaster_deploy` | trading-day index (prices.parquet) |

Affirmative statement: `src/task10/recommend.py` contains no reimplementation of OLS fitting, trailing-vol computation, correlation, or the risk-budget objective / gradient / solver call. The walk-forward fit and its label-realisation cutoff are reached only through `forecast_vol`; R only through `corr_matrix`; the objective, analytic gradient, x0 = 1/8, bounds [1e-8, inf), gtol = 1e-12, ftol = 1e-18, maxiter = 500 and the Gate C1 assertion only through `solve_risk_budget`; the budget vectors only through `RISK_BUDGETS`.

New arithmetic, closed list (Sec. 1.5):

1. RC normalisation: `rc = w * (S w) / (w' S w)` -- `allocate`.
2. Max-abs residual: `max_i |rc_i - b_i|` -- `allocate`.
3. Tuple packing / dataclass construction.
4. Input validation, including `earliest_valid_as_of()`, which is derived by walking the trading-day index and probing the imported `corr_matrix` / `forecast_vol` -- no window arithmetic is restated.

**DECLARED EXCEPTION -- covariance assembly.** `_sigma_from_d(t, d)` evaluates `corr_matrix(t) * np.outer(d, d)`: one expression, character-identical to the assembly line in `src/task9/covariance.py`. It exists because Sec. 1.3 requires `allocate` to build Sigma from a *supplied* d-vector while no Task 9 function accepts one -- `sigma(t, variant)` selects D internally from `variant` -- and `src/task9` is read-only (Sec. 0.2), so lifting D to a parameter there is out of scope. Bit-identity to the Task 9 path is verified below and is certified by Gates B2/B3/B4.

Exception evidence -- `_sigma_from_d(t, forecast_vol(t))` vs `task9.covariance.sigma(t, "advisor")`, exact float equality:

| date | bit-identical |
|---|---|
| 2012-02-01 | yes (all 64 entries bit-identical) |
| 2020-03-02 | yes (all 64 entries bit-identical) |
| 2022-06-01 | yes (all 64 entries bit-identical) |
| 2024-12-02 | yes (all 64 entries bit-identical) |

**GATE A2 PASS** -- every probe raised `ValueError`; the raised text is reproduced verbatim:

| call | exception | message |
|---|---|---|
| `recommend("aggressive", "2024-12-02")` | ValueError | `profile must be one of ('conservative', 'balanced', 'growth'), got 'aggressive'` |
| `recommend("balanced", "2024-12-01")  # Sunday` | ValueError | `as_of 2024-12-01 is not a trading day on the price index (no snapping to a neighbouring date is performed)` |
| `recommend("balanced", "2010-05-26")  # < earliest valid date` | ValueError | `as_of 2010-05-26 precedes the earliest valid date 2010-05-27 (the walk-forward forecaster and the 250-day correlation window are not both defined before it)` |
| `allocate("balanced", d[:7], "2024-12-02")  # 7 elements` | ValueError | `d_vector must have shape (8,), got (7,)` |
| `allocate("balanced", d with AGG -> 0.0, "2024-12-02")` | ValueError | `d_vector entries must be strictly positive: [0.008295212531746234, 0.010827919422911062, 0.009458525805633021, 0.011491848673824977, 0.0, 0.004318576239056208, 0.011222148510746342, 0.010698245324297698]` |
| `allocate("balanced", d with AGG -> nan, "2024-12-02")` | ValueError | `d_vector contains a non-finite entry: [0.008295212531746234, 0.010827919422911062, 0.009458525805633021, 0.011491848673824977, nan, 0.004318576239056208, 0.011222148510746342, 0.010698245324297698]` |

Derived earliest valid `as_of`: **2010-05-27**. Derivation (documented, not gated -- no registered reference exists): the trading-day index from `prices.parquet` is walked in order and the first date at which both imported Task 9 callables return without raising is taken; `corr_matrix` needs 250 clean return rows ending at t and the panel's first return row is NaN by construction (pct_change), so the bound is the 251st row of `returns.parquet`, which is later than the walk-forward fit's own bound (the fit needs only two realised samples per ticker). The preceding trading day 2010-05-26 is shown rejected above.

**GATE A3 PASS** -- contract verified programmatically on a live `recommend("balanced", "2024-12-02")` instance:

| object | check | result |
|---|---|---|
| Diagnostics | field names | exact match to Sec. 1.2 |
| Metadata | field names | exact match to Sec. 1.2 |
| Recommendation | field names | exact match to Sec. 1.2 |
| Recommendation | nesting | `.diagnostics` -> Diagnostics, `.metadata` -> Metadata |
| Recommendation | frozen | `weights = None` raises FrozenInstanceError |
| Diagnostics | frozen | `risk_budget = None` raises FrozenInstanceError |
| Metadata | frozen | `profile = None` raises FrozenInstanceError |
| Recommendation.weights | tuple[float] x 8 | yes |
| Recommendation.forecast_vols | tuple[float] x 8 | yes |
| Diagnostics.risk_contributions | tuple[float] x 8 | yes |
| Diagnostics.risk_budget | tuple[float] x 8 | yes |
| Diagnostics.max_abs_rc_minus_budget | float | yes |
| Metadata.covariance_source | == COVARIANCE_SOURCE | `Sigma = D_advisor(walk-forward OLS) * R_250d * D_advisor` |

## Leakage rule (Sec. 1.4)

At `as_of` = t the forecaster coefficients are fit only on samples whose 20-day label windows are fully realised by t (the label-realisation cutoff of the Task 9 Stage A contract, with its fit-universe clause: split rows only -- the 896 `split='drop'` rows never enter any fit); R is the Pearson correlation over the 250 trading days ending at and including t.

In code: both rules are enforced inside the imported `src/task9` functions -- `forecast_vol` -> `fit_coeffs` filters the fit frame on `vol_label_date <= t`, and `corr_matrix` takes `returns.loc[:t].tail(250)`. `src/task10` introduces no window arithmetic of its own; `earliest_valid_as_of()` is derived by probing those same callables rather than by restating their windows.

Window span at t: forecaster fit basis = all eligible labelled samples with `vol_label_date <= t`; correlation basis = trading days t-249 .. t inclusive.

## TICKERS rule (Sec. 1.1) -- branch applied

**Branch (a): imported, not redefined.** `src/task9` already carries a canonical ordering -- every Task 9 module does `from ..data.config import UNIVERSE`, and `RISK_BUDGETS` / `STATIC_BUCKET` are indexed by it -- so `TICKERS = tuple(UNIVERSE)` is taken from that same source.

## Deliverables (Stage A)

- `src/task10/__init__.py`
- `src/task10/recommend.py`
- `outputs/task10/stage_a_report.md`

**STOP -- Stage A boundary.** Gates A1-A3 reported above.