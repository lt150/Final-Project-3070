"""Task 10: the recommendation layer as one callable object.

This exposes ``profile -> forecasts -> covariance -> allocator -> weights``
as two callables returning one frozen dataclass. It is the object Task 12's SHAP
evaluation explains. Only the ``rb_advisor`` allocator is exposed.

Every numeric on the recommendation path is imported from ``src/task9`` and
called, not restated: the walk-forward OLS forecaster, the 250-day correlation,
the registered risk-budget objective/gradient/solver, and the registered budget
tables.  The arithmetic new to this module is:

  1. risk-contribution normalisation   rc = w * (S w) / (w' S w)
  2. the max-abs residual              max_i |rc_i - b_i|
  3. tuple packing / dataclass construction
  4. input validation (the earliest valid ``as_of`` is derived by probing the
     imported Task 9 callables, so no window arithmetic is restated here)
  5. Sigma assembly from a *supplied* d-vector, one expression:
     ``corr_matrix(t) * np.outer(d, d)``.

Leakage rule 

At ``as_of`` = t the forecaster coefficients are fit only on samples
whose 20-day label windows are fully realised by t (label-realisation cutoff,
per the Task 9 Stage A and its fit-universe clause: split rows only,
the 896 split='drop' rows never enter any fit); R is the Pearson correlation
over the 250 trading days ending at and including t.

The system is mu-free: nothing anywhere forecasts returns.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import re
import sys
from dataclasses import FrozenInstanceError, dataclass, fields
from datetime import datetime, timezone
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

from ..data.config import ROOT, UNIVERSE
from ..task9.allocators import PROFILES, RISK_BUDGETS, solve_risk_budget
from ..task9.covariance import CORR_WINDOW, corr_matrix
from ..task9.forecaster_deploy import _prices_index, forecast_vol

OUT_DIR = ROOT / "outputs" / "task10"

# Gate A1 verifies each entry by identity against a fresh import.
IMPORT_MAP: tuple[tuple[str, str, str], ...] = (
    ("UNIVERSE", "src.data.config", "canonical ticker ordering (task9's own source)"),
    ("ROOT", "src.data.config", "repo root for output paths"),
    ("PROFILES", "src.task9.allocators", "the three registered profile strings"),
    ("RISK_BUDGETS", "src.task9.allocators", "registered risk-budget tables"),
    ("solve_risk_budget", "src.task9.allocators", "registered objective/gradient/L-BFGS-B solve"),
    ("CORR_WINDOW", "src.task9.covariance", "250-day correlation window constant"),
    ("corr_matrix", "src.task9.covariance", "Pearson R over the 250 days ending at t"),
    ("forecast_vol", "src.task9.forecaster_deploy", "walk-forward OLS forecast vols at t"),
    ("_prices_index", "src.task9.forecaster_deploy", "trading-day index (prices.parquet)"),
)


# --------------------------------------------------------------------------- #
# Sec. 1.1 canonical constants
# --------------------------------------------------------------------------- #
# Branch (a) of Sec. 1.1: src/task9 already carries a canonical ordering -- every
# task9 module imports UNIVERSE from src/data/config and RISK_BUDGETS is indexed
# by it.
TICKERS: tuple[str, ...] = tuple(UNIVERSE)

# Provenance stamp for downstream readers. Not a parameter: nothing accepts it
# as an input and nothing can vary it.
COVARIANCE_SOURCE = "Sigma = D_advisor(walk-forward OLS) * R_250d * D_advisor"

ALLOCATOR_ID = "rb_advisor"  # Sec. A7: the only allocator this layer exposes


# --------------------------------------------------------------------------- #
# Sec. 1.2 return type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Diagnostics:
    risk_contributions: tuple[float, ...]  # length 8, canonical order, normalised
    risk_budget: tuple[float, ...]         # length 8, registered budget for the profile
    max_abs_rc_minus_budget: float         # achieved solver residual, diagnostic only


@dataclass(frozen=True)
class Metadata:
    profile: str                           # conservative | balanced | growth
    as_of: str                             # ISO date "YYYY-MM-DD"
    covariance_source: str                 # == COVARIANCE_SOURCE, always


@dataclass(frozen=True)
class Recommendation:
    weights: tuple[float, ...]             # length 8, canonical order, sums to 1, all >= 0
    forecast_vols: tuple[float, ...]       # length 8, canonical order, RAW DAILY vol units
    diagnostics: Diagnostics
    metadata: Metadata


# --------------------------------------------------------------------------- #
# Input validation (Sec. 1.3)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def earliest_valid_as_of() -> pd.Timestamp:
    """First trading day at which the imported Task 9 machinery is defined.

    Derived, not assumed: the trading-day index is walked in order and the first
    date at which both ``corr_matrix`` (250-day window) and ``forecast_vol``
    (walk-forward fit) return without raising is taken.  No window arithmetic is
    restated here -- the bound is whatever the registered code path admits.
    """
    for t in pd.DatetimeIndex(_prices_index()):
        try:
            corr_matrix(t)
            forecast_vol(t)
        except ValueError:
            continue
        return t
    raise RuntimeError("no trading day satisfies the Task 9 preconditions")


def _validate_profile(profile: str) -> str:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    return profile


def _validate_as_of(as_of: str) -> pd.Timestamp:
    """Parse an ISO date and check it against the trading grid. Never snaps."""
    if not isinstance(as_of, str):
        raise ValueError(f"as_of must be an ISO date string, got {type(as_of).__name__}")
    try:
        day = _dt.date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError(f"as_of {as_of!r} is not an ISO date (YYYY-MM-DD): {exc}") from exc

    t = pd.Timestamp(day)
    if t not in pd.DatetimeIndex(_prices_index()):
        raise ValueError(
            f"as_of {as_of} is not a trading day on the price index "
            "(no snapping to a neighbouring date is performed)"
        )
    first = earliest_valid_as_of()
    if t < first:
        raise ValueError(
            f"as_of {as_of} precedes the earliest valid date {first.date()} "
            "(the walk-forward forecaster and the 250-day correlation window "
            "are not both defined before it)"
        )
    return t


def _validate_d_vector(d_vector: Sequence[float]) -> np.ndarray:
    """Deliberately minimal (Sec. 1.3): length 8, finite, strictly positive.

    No range or plausibility checks -- Task 12's perturbed d-vectors must pass.
    """
    try:
        d = np.asarray(d_vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"d_vector is not numeric: {exc}") from exc
    if d.ndim != 1 or d.shape[0] != len(TICKERS):
        raise ValueError(
            f"d_vector must have shape ({len(TICKERS)},), got {d.shape}"
        )
    shown = [float(x) for x in d]  # plain-float repr, full precision
    if not np.isfinite(d).all():
        raise ValueError(f"d_vector contains a non-finite entry: {shown}")
    if not (d > 0.0).all():
        raise ValueError(f"d_vector entries must be strictly positive: {shown}")
    return d


# --------------------------------------------------------------------------- #
# Sigma from a supplied d (declared exception to Sec. 1.5)
# --------------------------------------------------------------------------- #
def _sigma_from_d(t: pd.Timestamp, d: np.ndarray) -> pd.DataFrame:
    """Sigma = D.R.D with R from the imported Task 9 correlation code path.

    Character-identical to the assembly line in src/task9/covariance.py; the
    ``reindex`` that sigma() performs there is a verified no-op here because d
    is supplied positionally in canonical order, which is R's own order.
    """
    R = corr_matrix(t)
    if tuple(R.index) != TICKERS or tuple(R.columns) != TICKERS:
        raise RuntimeError(
            f"correlation matrix axis order {tuple(R.index)} != canonical {TICKERS}"
        )
    return R * np.outer(d, d)


# --------------------------------------------------------------------------- #
# Sec. 1.3 callables
# --------------------------------------------------------------------------- #
def allocate(profile: str, d_vector: Sequence[float], as_of: str) -> Recommendation:
    """Risk-budget weights for ``profile`` from a supplied daily-vol vector.

    Task 12's SHAP explanation target: ``d_vector`` is the perturbable input.
    """
    _validate_profile(profile)
    t = _validate_as_of(as_of)
    d = _validate_d_vector(d_vector)

    S = _sigma_from_d(t, d)
    budgets = RISK_BUDGETS[profile]
    w = solve_risk_budget(S, budgets)

    # --- new arithmetic 1 and 2: RC normalisation and the max-abs residual
    S_arr = S.loc[list(TICKERS), list(TICKERS)].to_numpy()
    w_arr = w.reindex(list(TICKERS)).to_numpy()
    b_arr = budgets.reindex(list(TICKERS)).to_numpy()
    Sw = S_arr @ w_arr
    rc = w_arr * Sw / (w_arr @ Sw)
    resid = float(np.abs(rc - b_arr).max())

    return Recommendation(
        weights=tuple(float(x) for x in w_arr),
        forecast_vols=tuple(float(x) for x in d),
        diagnostics=Diagnostics(
            risk_contributions=tuple(float(x) for x in rc),
            risk_budget=tuple(float(x) for x in b_arr),
            max_abs_rc_minus_budget=resid,
        ),
        metadata=Metadata(
            profile=profile,
            as_of=as_of,
            covariance_source=COVARIANCE_SOURCE,
        ),
    )


def recommend(profile: str, as_of: str) -> Recommendation:
    """Walk-forward forecast at ``as_of``, then delegate to ``allocate``.

    Single delegation, no parallel path: ``recommend(p, t)`` is
    ``allocate(p, forecast_vol(t), t)`` by construction (Gate B4).
    """
    _validate_profile(profile)
    t = _validate_as_of(as_of)
    d = forecast_vol(t)
    if tuple(d.index) != TICKERS:
        raise RuntimeError(
            f"forecast_vol axis order {tuple(d.index)} != canonical {TICKERS}"
        )
    return allocate(profile, d.to_numpy(), as_of)


# --------------------------------------------------------------------------- #
# Report helpers (ASCII-only output, Sec. 5)
# --------------------------------------------------------------------------- #
def _md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "| " + " | ".join(str(h) for h in header) + " |"
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _write_report(name: str, lines: Sequence[str]) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (OUT_DIR / name).write_text(report, encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
# Stage A gates
# --------------------------------------------------------------------------- #
def gate_a1(lines: list[str]) -> None:
    """Composition audit: the import map, verified by identity."""
    rows = []
    for name, module, purpose in IMPORT_MAP:
        obj = getattr(importlib.import_module(module), name)
        local = globals()[name]
        assert obj is local, (
            f"GATE A1: local {name} is not {module}.{name} -- import drift, STOP"
        )
        rows.append([f"`{name}`", f"`{module}`", purpose])
    lines += [
        "**GATE A1 PASS (with one declared exception, below)** -- import map "
        "verified by object identity (each local symbol `is` the attribute of a "
        "freshly imported module of record):",
        "",
        _md_table(["symbol", "module of record", "purpose"], rows),
        "",
        "Affirmative statement: `src/task10/recommend.py` contains no "
        "reimplementation of OLS fitting, trailing-vol computation, correlation, "
        "or the risk-budget objective / gradient / solver call. The walk-forward "
        "fit and its label-realisation cutoff are reached only through "
        "`forecast_vol`; R only through `corr_matrix`; the objective, analytic "
        "gradient, x0 = 1/8, bounds [1e-8, inf), gtol = 1e-12, ftol = 1e-18, "
        "maxiter = 500 and the Gate C1 assertion only through "
        "`solve_risk_budget`; the budget vectors only through `RISK_BUDGETS`.",
        "",
        "New arithmetic, closed list (Sec. 1.5):",
        "",
        "1. RC normalisation: `rc = w * (S w) / (w' S w)` -- `allocate`.",
        "2. Max-abs residual: `max_i |rc_i - b_i|` -- `allocate`.",
        "3. Tuple packing / dataclass construction.",
        "4. Input validation, including `earliest_valid_as_of()`, which is "
        "derived by walking the trading-day index and probing the imported "
        "`corr_matrix` / `forecast_vol` -- no window arithmetic is restated.",
        "",
        "**DECLARED EXCEPTION -- covariance assembly.** "
        "`_sigma_from_d(t, d)` evaluates `corr_matrix(t) * np.outer(d, d)`: one "
        "expression, character-identical to the assembly line in "
        "`src/task9/covariance.py`. It exists because Sec. 1.3 requires `allocate` to "
        "build Sigma from a *supplied* d-vector while no Task 9 function accepts "
        "one -- `sigma(t, variant)` selects D internally from `variant` -- and "
        "`src/task9` is read-only (Sec. 0.2), so lifting D to a parameter there is "
        "out of scope."
        "Bit-identity to the Task 9 path is verified below "
        "and is certified by Gates B2/B3/B4."
    ]

    # Evidence for the exception: bit-identity against sigma(t, "advisor").
    from ..task9.covariance import sigma as _t9_sigma

    ev = []
    for date in ("2012-02-01", "2020-03-02", "2022-06-01", "2024-12-02"):
        t = pd.Timestamp(date)
        mine = _sigma_from_d(t, forecast_vol(t).to_numpy()).to_numpy()
        ref = _t9_sigma(t, "advisor").to_numpy()
        identical = bool((mine == ref).all())
        assert identical, f"GATE A1: assembly differs from task9 sigma at {date} -- STOP"
        ev.append([date, "yes (all 64 entries bit-identical)"])
    lines += [
        "",
        "Exception evidence -- `_sigma_from_d(t, forecast_vol(t))` vs "
        "`task9.covariance.sigma(t, \"advisor\")`, exact float equality:",
        "",
        _md_table(["date", "bit-identical"], ev),
    ]


def gate_a2(lines: list[str]) -> None:
    """Validation probes: five raises, with the actual exception text."""
    first = earliest_valid_as_of()
    good_d = forecast_vol(pd.Timestamp("2024-12-02")).to_numpy()

    probes = [
        (
            'recommend("aggressive", "2024-12-02")',
            lambda: recommend("aggressive", "2024-12-02"),
        ),
        (
            'recommend("balanced", "2024-12-01")  # Sunday',
            lambda: recommend("balanced", "2024-12-01"),
        ),
        (
            'recommend("balanced", "2010-05-26")  # < earliest valid date',
            lambda: recommend("balanced", "2010-05-26"),
        ),
        (
            'allocate("balanced", d[:7], "2024-12-02")  # 7 elements',
            lambda: allocate("balanced", good_d[:7], "2024-12-02"),
        ),
        (
            'allocate("balanced", d with AGG -> 0.0, "2024-12-02")',
            lambda: allocate("balanced", _replace(good_d, 4, 0.0), "2024-12-02"),
        ),
        (
            'allocate("balanced", d with AGG -> nan, "2024-12-02")',
            lambda: allocate("balanced", _replace(good_d, 4, float("nan")), "2024-12-02"),
        ),
    ]
    rows = []
    for label, fn in probes:
        try:
            fn()
        except ValueError as exc:
            rows.append([f"`{label}`", "ValueError", f"`{exc}`"])
        else:
            raise AssertionError(f"GATE A2: {label} did not raise -- STOP")
    lines += [
        "**GATE A2 PASS** -- every probe raised `ValueError`; the raised text is "
        "reproduced verbatim:",
        "",
        _md_table(["call", "exception", "message"], rows),
        "",
        f"Derived earliest valid `as_of`: **{first.date()}**. Derivation "
        "(documented, not gated -- no registered reference exists): the "
        "trading-day index from `prices.parquet` is walked in order and the "
        "first date at which both imported Task 9 callables return without "
        f"raising is taken; `corr_matrix` needs {CORR_WINDOW} clean return rows "
        "ending at t and the panel's first return row is NaN by construction "
        "(pct_change), so the bound is the 251st row of `returns.parquet`, which "
        "is later than the walk-forward fit's own bound (the fit needs only two "
        "realised samples per ticker). The preceding trading day 2010-05-26 is "
        "shown rejected above.",
    ]


def _replace(arr: np.ndarray, i: int, value: float) -> np.ndarray:
    out = arr.copy()
    out[i] = value
    return out


def _ticker_order_crosscheck() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Sec. 1.1 cross-check: TICKERS vs the smoke CSV column order and RISK_BUDGETS."""
    header = (ROOT / "outputs" / "task9" / "smoke_weights.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0].split(",")
    csv_order = tuple(header[3:])
    budget_order = tuple(RISK_BUDGETS.index)
    assert csv_order == TICKERS, (
        f"Sec. 1.1 cross-check: smoke_weights.csv ticker columns {csv_order} != "
        f"canonical {TICKERS} -- STOP"
    )
    assert budget_order == TICKERS, (
        f"Sec. 1.1 cross-check: RISK_BUDGETS.index {budget_order} != canonical "
        f"{TICKERS} -- STOP"
    )
    return csv_order, budget_order


def gate_a3(lines: list[str]) -> None:
    """Contract gate: field names, nesting, frozenness, tuple lengths, stamp."""
    schema = {
        "Diagnostics": {"risk_contributions", "risk_budget", "max_abs_rc_minus_budget"},
        "Metadata": {"profile", "as_of", "covariance_source"},
        "Recommendation": {"weights", "forecast_vols", "diagnostics", "metadata"},
    }
    checks = []
    for cls in (Diagnostics, Metadata, Recommendation):
        got = {f.name for f in fields(cls)}
        assert got == schema[cls.__name__], (
            f"GATE A3: {cls.__name__} fields {sorted(got)} != registered "
            f"{sorted(schema[cls.__name__])} -- STOP"
        )
        checks.append([cls.__name__, "field names", "exact match to Sec. 1.2"])

    rec = recommend("balanced", "2024-12-02")

    assert type(rec.diagnostics) is Diagnostics and type(rec.metadata) is Metadata, (
        "GATE A3: nesting is not Recommendation -> (Diagnostics, Metadata) -- STOP"
    )
    checks.append(["Recommendation", "nesting", "`.diagnostics` -> Diagnostics, `.metadata` -> Metadata"])

    for obj, attr in ((rec, "weights"), (rec.diagnostics, "risk_budget"), (rec.metadata, "profile")):
        try:
            setattr(obj, attr, None)
        except FrozenInstanceError:
            checks.append([type(obj).__name__, "frozen", f"`{attr} = None` raises FrozenInstanceError"])
        else:
            raise AssertionError(f"GATE A3: {type(obj).__name__} is not frozen -- STOP")

    seqs = [
        ("Recommendation.weights", rec.weights),
        ("Recommendation.forecast_vols", rec.forecast_vols),
        ("Diagnostics.risk_contributions", rec.diagnostics.risk_contributions),
        ("Diagnostics.risk_budget", rec.diagnostics.risk_budget),
    ]
    for label, value in seqs:
        assert type(value) is tuple and len(value) == len(TICKERS), (
            f"GATE A3: {label} is {type(value).__name__} of length "
            f"{len(value)}, expected tuple of length {len(TICKERS)} -- STOP"
        )
        assert all(type(x) is float for x in value), f"GATE A3: {label} is not all float -- STOP"
        checks.append([label, "tuple[float] x 8", "yes"])

    assert type(rec.diagnostics.max_abs_rc_minus_budget) is float, "GATE A3: residual not float -- STOP"
    checks.append(["Diagnostics.max_abs_rc_minus_budget", "float", "yes"])

    assert rec.metadata.covariance_source == COVARIANCE_SOURCE, (
        f"GATE A3: covariance_source {rec.metadata.covariance_source!r} != "
        f"{COVARIANCE_SOURCE!r} -- STOP"
    )
    checks.append(["Metadata.covariance_source", "== COVARIANCE_SOURCE", f"`{COVARIANCE_SOURCE}`"])

    lines += [
        "**GATE A3 PASS** -- contract verified programmatically on a live "
        '`recommend("balanced", "2024-12-02")` instance:',
        "",
        _md_table(["object", "check", "result"], checks),
    ]


def main_stage_a() -> None:
    existing = OUT_DIR / "stage_a_report.md"

    lines = [
        "# Task 10 Stage A report -- recommendation-layer module, contract, validation",
        "",
        "Deliverable module: `src/task10/recommend.py`. Exposes "
        "`allocate(profile, d_vector, as_of)` and `recommend(profile, as_of)`, "
        "both returning a frozen `Recommendation`. Allocator exposed: "
        f"`{ALLOCATOR_ID}` only (Sec. A7). Units: raw daily vol / raw daily variance; "
        "nothing annualised; the system is mu-free.",
        "",
        "## Gates",
        "",
    ]
    try:
        gate_a1(lines)
        lines.append("")
        gate_a2(lines)
        lines.append("")
        gate_a3(lines)
        csv_order, budget_order = _ticker_order_crosscheck()
    except AssertionError as exc:
        lines += ["", f"**GATE FAILURE -- STOP**: {exc}", "",
                  "Diagnosis recorded."]
        print(_write_report("stage_a_report.md", lines))
        sys.exit(1)

    lines += [
        "",
        "## Leakage rule (Sec. 1.4)",
        "",
        "At `as_of` = t the forecaster coefficients are fit only on "
        "samples whose 20-day label windows are fully realised by t (the "
        "label-realisation cutoff of the Task 9 Stage A contract, with its "
        "fit-universe clause: split rows only -- the 896 `split='drop'` rows "
        "never enter any fit); R is the Pearson correlation over the 250 trading "
        "days ending at and including t.",
        "",
        "In code: both rules are enforced inside the imported `src/task9` "
        "functions -- `forecast_vol` -> `fit_coeffs` filters the fit frame on "
        "`vol_label_date <= t`, and `corr_matrix` takes "
        "`returns.loc[:t].tail(250)`. `src/task10` introduces no window "
        "arithmetic of its own; `earliest_valid_as_of()` is derived by probing "
        "those same callables rather than by restating their windows.",
        "",
        "Window span at t: forecaster fit basis = all eligible labelled samples "
        "with `vol_label_date <= t`; correlation basis = trading days t-249 .. t "
        "inclusive.",
        "",
        "## TICKERS rule (Sec. 1.1) -- branch applied",
        "",
        "**Branch (a): imported, not redefined.** `src/task9` already carries a "
        "canonical ordering -- every Task 9 module does "
        "`from ..data.config import UNIVERSE`, and `RISK_BUDGETS` / "
        "`STATIC_BUCKET` are indexed by it -- so `TICKERS = tuple(UNIVERSE)` is "
        "taken from that same source.",
        "",
        "## Deliverables (Stage A)",
        "",
        "- `src/task10/__init__.py`",
        "- `src/task10/recommend.py`",
        "- `outputs/task10/stage_a_report.md`",
        "",
        "**STOP -- Stage A boundary.** Gates A1-A3 reported above. "
    ]
    print(_write_report("stage_a_report.md", lines))


# --------------------------------------------------------------------------- #
# Stage B
# --------------------------------------------------------------------------- #
TASK9_DIR = ROOT / "outputs" / "task9"
RECORD_STAGE_B = TASK9_DIR / "stage_b_report.md"
RECORD_STAGE_C = TASK9_DIR / "stage_c_report.md"
RECORD_SMOKE = TASK9_DIR / "smoke_weights.csv"

SMOKE_DATES = ("2012-02-01", "2020-03-02", "2022-06-01", "2024-12-02")

B5_SUM_TOL = 1e-9        # inherited weight-sum convention
B5_RC_SUM_TOL = 1e-12    # engine-arithmetic band
B5_RESID_TOL = 1e-6      # inherited Gate C1 convention


def _read_gate_b2_anchors() -> dict[str, str]:
    """Gate B2 checksum block of the Task 9 Stage B report, as stored strings."""
    text = RECORD_STAGE_B.read_text(encoding="utf-8")
    wanted = ("trace(Sigma_classical)", "Sigma_classical[SPY, AGG]", "R[SPY, QQQ]")
    found: dict[str, str] = {}
    for key in wanted:
        hits = re.findall(
            r"^- " + re.escape(key) + r" = `([^`]+)`\s*$", text, flags=re.MULTILINE
        )
        assert len(hits) == 1, (
            f"GATE B1: '{key}' appears {len(hits)} times in {RECORD_STAGE_B.name}, "
            "expected exactly 1 -- cannot identify the reference, STOP"
        )
        found[key] = hits[0]
    return found


def _read_checksum_record() -> dict[str, str]:
    text = RECORD_STAGE_C.read_text(encoding="utf-8")
    heads = [
        i for i, line in enumerate(text.splitlines())
        if line.startswith("## Checksum record") and "rb_advisor balanced" in line
    ]
    assert len(heads) == 1, (
        f"GATE B2: {len(heads)} 'Checksum record -- rb_advisor balanced' sections "
        f"in {RECORD_STAGE_C.name}, expected exactly 1 -- STOP"
    )
    out: dict[str, str] = {}
    for line in text.splitlines()[heads[0] + 1:]:
        if line.startswith("## "):
            break
        m = re.match(r"^- ([A-Z]+) = `([^`]+)`\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    assert tuple(out) == TICKERS, (
        f"GATE B2: checksum record lists {tuple(out)}, expected {TICKERS} -- STOP"
    )
    return out


def _read_smoke_record() -> tuple[dict[tuple[str, str], list[str]], bool]:
    """rb_advisor rows of smoke_weights.csv as stored strings + a precision verdict."""
    rows = RECORD_SMOKE.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    assert tuple(header[3:]) == TICKERS, (
        f"GATE B3: smoke CSV ticker columns {tuple(header[3:])} != {TICKERS} -- STOP"
    )
    out: dict[tuple[str, str], list[str]] = {}
    full_precision = True
    for line in rows[1:]:
        cells = line.split(",")
        if cells[1] != ALLOCATOR_ID:
            continue
        out[(cells[0], cells[2])] = cells[3:]
        for c in cells[3:]:
            if repr(float(c)) != c:
                full_precision = False
    assert len(out) == 12, f"GATE B3: {len(out)} rb_advisor rows, expected 12 -- STOP"
    return out, full_precision


def gate_b1(lines: list[str]) -> None:
    """Covariance module integrity at 2024-12-02 (Sigma_classical is off-path)."""
    from ..task9.covariance import sigma as _t9_sigma  # gate-only, not on the path

    t = pd.Timestamp("2024-12-02")
    S_cl = _t9_sigma(t, "classical")
    R = corr_matrix(t)
    got = {
        "trace(Sigma_classical)": float(np.trace(S_cl.to_numpy())),
        "Sigma_classical[SPY, AGG]": float(S_cl.at["SPY", "AGG"]),
        "R[SPY, QQQ]": float(R.at["SPY", "QQQ"]),
    }
    ref = _read_gate_b2_anchors()

    rows, bad = [], []
    for key, value in got.items():
        stored = ref[key]
        ok = repr(value) == stored
        rows.append([f"`{key}`", f"`{stored}`", f"`{value!r}`", "MATCH" if ok else "**FAIL**"])
        if not ok:
            bad.append((key, stored, repr(value)))
    lines += [
        ("**GATE B1 PASS** -- " if not bad else "**GATE B1 FAIL** -- ")
        + "shared covariance module at t = 2024-12-02, compared repr-exact "
        f"against the Gate B2 checksum block of `{RECORD_STAGE_B.name}` (values "
        "read from that file at execution time). Sigma_classical is not on the "
        "recommendation path; this gate certifies the shared module is "
        "behaviourally intact.",
        "",
        _md_table(["anchor", "recorded (Task 9)", "computed (Task 10 session)", "verdict"], rows),
    ]
    assert not bad, f"GATE B1: repr mismatch {bad} -- STOP"


def gate_b2(lines: list[str], recs: dict[tuple[str, str], Recommendation]) -> None:
    """End-to-end anchor: recommend('balanced', '2024-12-02').weights."""
    ref = _read_checksum_record()
    rec = recs[("2024-12-02", "balanced")]
    rows, bad = [], []
    for tk, value in zip(TICKERS, rec.weights):
        stored = ref[tk]
        ok = repr(value) == stored
        rows.append([tk, f"`{stored}`", f"`{value!r}`", "MATCH" if ok else "**FAIL**"])
        if not ok:
            bad.append((tk, stored, repr(value)))
    lines += [
        ("**GATE B2 PASS** -- " if not bad else "**GATE B2 FAIL** -- ")
        + '`recommend("balanced", "2024-12-02").weights` vs the section '
        f'"Checksum record -- rb_advisor balanced @ 2024-12-02" of '
        f"`{RECORD_STAGE_C.name}`. "
        "Repr-exact per component, through the identical solver trajectory:",
        "",
        _md_table(["ticker", "recorded (Task 9)", "computed (Task 10 session)", "verdict"], rows),
    ]
    assert not bad, f"GATE B2: repr mismatch {bad} -- STOP"


def gate_b3(lines: list[str], recs: dict[tuple[str, str], Recommendation]) -> None:
    """Smoke tie-out: 12 weight vectors vs the 12 rb_advisor rows of the CSV."""
    ref, full_precision = _read_smoke_record()
    branch = (
        "full round-trip precision (every cell satisfies `repr(float(cell)) == "
        "cell`), so the **repr-exact branch** applies"
        if full_precision
        else "less than full round-trip precision, so the **stored-precision "
        "branch** applies"
    )
    rows, bad = [], []
    for date in SMOKE_DATES:
        for profile in PROFILES:
            stored = ref[(date, profile)]
            got = recs[(date, profile)].weights
            mism = [
                (tk, s, repr(g))
                for tk, s, g in zip(TICKERS, stored, got)
                if (repr(g) != s if full_precision else float(s) != g)
            ]
            rows.append([date, profile, "8/8 match" if not mism else f"**{len(mism)} MISMATCH**"])
            bad += [(date, profile, *m) for m in mism]
    lines += [
        ("**GATE B3 PASS** -- " if not bad else "**GATE B3 FAIL** -- ")
        + f"12 `recommend` calls (4 smoke dates x 3 profiles) vs the 12 "
        f"`{ALLOCATOR_ID}` rows of `{RECORD_SMOKE.name}`. Stored precision "
        f"inspected first: the file carries {branch}. All 96 components compared:",
        "",
        _md_table(["date", "profile", "weights vs record"], rows),
    ]
    assert not bad, f"GATE B3: mismatches {bad} -- STOP"


def gate_b4(lines: list[str], recs: dict[tuple[str, str], Recommendation]) -> None:
    """Delegation identity: recommend(p, t) == allocate(p, forecast(t), t)."""
    rows, bad = [], []
    for date in SMOKE_DATES:
        d = forecast_vol(pd.Timestamp(date))
        for profile in PROFILES:
            lhs = recs[(date, profile)]
            rhs = allocate(profile, d.to_numpy(), date)
            per_field = {
                "weights": lhs.weights == rhs.weights,
                "forecast_vols": lhs.forecast_vols == rhs.forecast_vols,
                "diagnostics.risk_contributions":
                    lhs.diagnostics.risk_contributions == rhs.diagnostics.risk_contributions,
                "diagnostics.risk_budget":
                    lhs.diagnostics.risk_budget == rhs.diagnostics.risk_budget,
                "diagnostics.max_abs_rc_minus_budget":
                    lhs.diagnostics.max_abs_rc_minus_budget
                    == rhs.diagnostics.max_abs_rc_minus_budget,
                "metadata.profile": lhs.metadata.profile == rhs.metadata.profile,
                "metadata.as_of": lhs.metadata.as_of == rhs.metadata.as_of,
                "metadata.covariance_source":
                    lhs.metadata.covariance_source == rhs.metadata.covariance_source,
            }
            failed = [k for k, v in per_field.items() if not v]
            whole = lhs == rhs
            rows.append([
                date, profile,
                "8/8 exact" if not failed else "**" + ", ".join(failed) + "**",
                "yes" if whole else "**no**",
            ])
            if failed or not whole:
                bad.append((date, profile, failed, whole))
    lines += [
        ("**GATE B4 PASS** -- " if not bad else "**GATE B4 FAIL** -- ")
        + 'for all 12 (date, profile) pairs, `recommend(p, t)` vs '
        "`allocate(p, forecast_vol(t), t)`, field by field with exact float "
        "equality including nested fields, plus whole-dataclass `==`. Same "
        "session, same code path -- exact, not toleranced. The identity holds by "
        "construction: `recommend` computes the forecast and delegates in a "
        "single call, with no parallel numerical path.",
        "",
        _md_table(["date", "profile", "fields exact", "whole `==`"], rows),
    ]
    assert not bad, f"GATE B4: identity broken at {bad} -- STOP"


def gate_b5(lines: list[str], recs: dict[tuple[str, str], Recommendation]) -> None:
    """Mechanical + diagnostic checks on all 12 recommendations."""
    rows, bad = [], []
    for date in SMOKE_DATES:
        for profile in PROFILES:
            rec = recs[(date, profile)]
            sum_gap = abs(sum(rec.weights) - 1.0)
            w_min = min(rec.weights)
            rc_gap = abs(sum(rec.diagnostics.risk_contributions) - 1.0)
            resid = rec.diagnostics.max_abs_rc_minus_budget
            checks = (
                sum_gap < B5_SUM_TOL, w_min >= 0.0,
                rc_gap < B5_RC_SUM_TOL, resid < B5_RESID_TOL,
            )
            rows.append([
                date, profile, f"{sum_gap:.3e}", f"{w_min:.6f}",
                f"{rc_gap:.3e}", f"{resid:.3e}",
                "PASS" if all(checks) else "**FAIL**",
            ])
            if not all(checks):
                bad.append((date, profile, sum_gap, w_min, rc_gap, resid))
    lines += [
        ("**GATE B5 PASS** -- " if not bad else "**GATE B5 FAIL** -- ")
        + f"all 12 recommendations: `|sum(weights) - 1|` < {B5_SUM_TOL:g}, "
        f"`min(weights)` >= 0, `|sum(risk_contributions) - 1|` < "
        f"{B5_RC_SUM_TOL:g}, `max_abs_rc_minus_budget` < {B5_RESID_TOL:g} (the "
        "inherited Gate C1 convention).",
        "",
        # Column names avoid literal pipes, which would break the table.
        _md_table(
            ["date", "profile", "abs_sum_w_minus_1", "min_w", "abs_sum_rc_minus_1",
             "max_abs_rc_minus_budget", "verdict"],
            rows,
        ),
    ]
    assert not bad, f"GATE B5: {bad} -- STOP"


def write_smoke_csv(recs: dict[tuple[str, str], Recommendation]) -> int:
    """12 rows at full repr() precision -- Task 12's reference artifact."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        ["date", "profile"]
        + [f"w_{tk}" for tk in TICKERS]
        + [f"d_{tk}" for tk in TICKERS]
        + ["max_abs_rc_minus_budget"]
    )
    out = [",".join(header)]
    for date in SMOKE_DATES:
        for profile in PROFILES:
            rec = recs[(date, profile)]
            out.append(",".join(
                [date, profile]
                + [repr(x) for x in rec.weights]
                + [repr(x) for x in rec.forecast_vols]
                + [repr(rec.diagnostics.max_abs_rc_minus_budget)]
            ))
    (OUT_DIR / "smoke_recommendations.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(out) - 1


def main_stage_b() -> None:
    import pyarrow
    import scipy

    lines = [
        "# Task 10 Stage B report -- numerical tie-outs and smoke evidence",
        "",
        "All reference values are read from the on-disk files of record at "
        "execution time, never retyped: "
        f"`outputs/task9/{RECORD_STAGE_B.name}`, "
        f"`outputs/task9/{RECORD_STAGE_C.name}`, "
        f"`outputs/task9/{RECORD_SMOKE.name}`.",
        "",
        "## Gates",
        "",
    ]
    try:
        recs = {
            (date, profile): recommend(profile, date)
            for date in SMOKE_DATES
            for profile in PROFILES
        }
        gate_b1(lines)
        lines.append("")
        gate_b2(lines, recs)
        lines.append("")
        gate_b3(lines, recs)
        lines.append("")
        gate_b4(lines, recs)
        lines.append("")
        gate_b5(lines, recs)
        n_rows = write_smoke_csv(recs)
    except AssertionError as exc:
        lines += ["", f"**GATE FAILURE -- STOP**: {exc}", "",
                  "Diagnosis recorded. "
                  "smoke_recommendations.csv is NOT written on a failed stage."]
        print(_write_report("stage_b_report.md", lines))
        sys.exit(1)

    lines += [
        "",
        "## Tolerance doctrine as applied",
        "",
        "| gate | comparison | standard |",
        "|---|---|---|",
        "| B1 | identical-trajectory reproduction of the covariance anchors | repr-exact |",
        "| B2 | identical-trajectory reproduction of the end-to-end anchor | repr-exact |",
        "| B3 | 12 weight vectors vs a full-round-trip-precision CSV | repr-exact |",
        "| B4 | within-session delegation identity | exact float equality |",
        f"| B5 | engine arithmetic (RC sum) | {B5_RC_SUM_TOL:g} |",
        f"| B5 | inherited weight-sum check | {B5_SUM_TOL:g} |",
        f"| B5 | solver-residual convention (Gate C1) | {B5_RESID_TOL:g} |",
        "",
        "No comparison between two distinct L-BFGS-B solves arose. "
        "",
        "## Notes",
        "",
        "- Gate-only import, not on the recommendation path: "
        "`task9.covariance.sigma` is imported inside `gate_b1` (and inside the "
        "Gate A1 evidence block) to build Sigma_classical and to certify the "
        "amended Sec. 1.5 item 5. The recommendation path itself reaches the "
        "covariance module only through `corr_matrix`, per the ratified Gate A1 "
        "import map.",
        "- Gate B2's reference section is located by asserting that exactly one "
        "`## Checksum record ... rb_advisor balanced` header exists in "
        f"`{RECORD_STAGE_C.name}`, so the verbatim pre-amendment failure record "
        "earlier in that file cannot be picked up by accident.",
        "- `recommend` was called once per (date, profile); the same 12 objects "
        "feed Gates B2-B5 and the CSV, so the artifact is the evidence, not a "
        "second evaluation of it.",
        "- Units: `forecast_vols` and the Sigma behind the weights are raw daily; "
        "nothing is annualised. The system is mu-free.",
        "- `max_abs_rc_minus_budget` is recorded per row in the artifact as a "
        "health indicator; it acquires no evidentiary status by existing.",
        "",
        "## Deliverables (Stage B)",
        "",
        f"- `outputs/task10/smoke_recommendations.csv` ({n_rows} rows, full "
        "`repr()` precision: 8 weights, 8 forecast_vols, the residual)",
        "- `outputs/task10/stage_b_report.md`",
        "",
        "**STOP -- Stage B boundary.** Gates B1-B5 reported above. "
    ]
    print(_write_report("stage_b_report.md", lines))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = argv[0] if argv else "stage_b"
    if stage == "stage_a":
        main_stage_a()
        return
    if stage == "stage_b":
        main_stage_b()
        return
    raise SystemExit(f"unknown stage {stage!r}; expected 'stage_a' or 'stage_b'")


if __name__ == "__main__":
    main()
