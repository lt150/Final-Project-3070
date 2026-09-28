"""Task 12 Stage A driver: engine, imports, closed-form validation.

Gates A1 (import posture), A2 (engine validation on a registered toy), A3
(linear-layer closed form). On any gate failure: the diagnosis is written to the
stage report verbatim with full repr values, nothing is patched.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np

from . import fixtures
from .engine import enumerate_exact, shapley_weights
from .fixtures import (
    IMPORT_MAP,
    OUT_DIR,
    LINEAR_BACKGROUND_DATE,
    LINEAR_EXPLICAND_DATE,
    TICKERS,
    TOL_CLOSED_FORM_DIAG,
    TOL_TOY,
    TOY_BACKGROUND,
    TOY_EXACT_PHI,
    TOY_EXPLICAND,
    linear_map,
    toy_function,
)
from .reporting import (
    env_header,
    md_table,
    package_imports,
    registered_upstream,
    write_report,
)


# --------------------------------------------------------------------------- #
# GATE A1 -- import posture
# --------------------------------------------------------------------------- #
def gate_a1(lines: list[str]) -> None:
    rows = []
    for name, module, purpose in IMPORT_MAP:
        obj = getattr(importlib.import_module(module), name)
        local = getattr(fixtures, name)
        assert obj is local, (
            f"GATE A1: local {name} is not {module}.{name} -- import drift, STOP"
        )
        rows.append([f"`{name}`", f"`{module}`", purpose])

    audit, upstream = package_imports()
    registered = registered_upstream()
    unregistered = [s for s in upstream if s not in registered]
    assert not unregistered, (
        f"GATE A1: src/task12 reaches upstream symbols outside the registered "
        f"import map: {unregistered} -- STOP"
    )
    upstream_rows = [
        [f"`{f}`", f"`{stmt}`", f"`{target}`"]
        for f, stmt, kind, target in audit
        if kind == "upstream"
    ]

    lines += [
        "**GATE A1 PASS (with one declared item, below)** -- import map verified "
        "by object identity (each symbol used by `src/task12` `is` the attribute "
        "of a freshly imported module of record):",
        "",
        md_table(["symbol", "module of record", "purpose"], rows),
        "",
        "Static audit of the package source (every `from ..x import y` statement "
        "in `src/task12/*.py`, parsed with `ast`; each resolved target was "
        "checked against the map above and no unregistered upstream symbol "
        "exists):",
        "",
        md_table(["file", "statement", "resolved module"], upstream_rows),
        "",
        "**Affirmations.**",
        "",
        "1. `allocate` is reached only via `src.task10.recommend`; the identity "
        "check above is `is`, not equality. No other route into the "
        "recommendation layer exists in this package: `src.task9.allocators."
        "solve_risk_budget`, `src.task9.covariance.sigma` / `corr_matrix` and "
        "`src.task9.allocators.RISK_BUDGETS` are NOT imported anywhere in "
        "`src/task12` -- the covariance assembly, the budget tables and the "
        "L-BFGS-B solve are reached only through `allocate`, as Task 10 "
        "composed them.",
        "2. `vol_trail_20` and the walk-forward coefficients are reached only "
        "via the imported Task 9 code paths: the background d_classical(t) is "
        "`vol_trail_20(t)` -- the same callable, with the same 20-day "
        "sample-std window and the same returns panel, that `sigma(t, "
        "\"classical\")` calls to build the ablation twin's D -- and the "
        "coefficients are `fit_coeffs(t)`, i.e. the walk-forward closed-form OLS "
        "with its `vol_label_date <= t` label-realisation cutoff. Neither is "
        "restated.",
        "3. No reimplementation of any Task 9 / Task 10 numeric: this package "
        "contains no OLS fit, no trailing-vol computation, no correlation, no "
        "Sigma assembly, no risk-budget objective / gradient / solver call, no "
        "risk-contribution normalisation, and no window arithmetic of any kind. "
        "The window/data semantics of every coalition value are entirely those "
        "of `allocate` at t.",
        "",
        "**New numerics, closed list.**",
        "",
        "1. The Shapley enumeration and its coalition weights "
        "`s! (n - s - 1)! / n!` -- `engine.py`.",
        "2. Hybrid construction `h_i = x_i if i in S else b_i` -- `engine.py` "
        "(selection only; every entry is a bit-identical copy of an entry of x "
        "or of b).",
        "3. `dphi` / `dw` / `rho` arithmetic -- `metrics.py`, and the "
        "multiplicative probe application in `probes.py`.",
        "4. Summary and exhibit code -- `stage_b.py`, `stage_c.py`, "
        "`exhibits.py`, `reporting.py`.",
        "5. Stage D's DeepSHAP driver -- not yet written (see the note below).",
        "",
        "**DECLARED ITEM -- the linear map, planning-level.** Gate A1's "
        "closed list as written does not name the evaluation of the Stage A "
        "explained function itself. Register it verbatim -- "
        "`g(x)_i = a_i(t*) + b_i(t*) * x_i` with the coefficients obtained "
        "through the imported Task 9 fit path -- and it cannot be reached "
        "through any Task 9 callable: `forecast_vol(t)` evaluates exactly this "
        "map but fixes x = `vol_trail_20(t)`, whereas the exactness check must "
        "evaluate it at 2^8 hybrid inputs, and `src/task9` is read-only so "
        "lifting x to a parameter there is out of scope. `fixtures.linear_map` "
        "is therefore one expression, `a + b * x`, with (a, b) imported and "
        "never retyped. Gate A1 as written asks for an unqualified "
        "'nothing else'; that statement cannot be made without this "
        "qualification, so it is reported here rather than absorbed silently. "
        "In-repo precedent for the posture: the Task 10 Gate A1 declared "
        "exception for `_sigma_from_d`.",
    ]


# --------------------------------------------------------------------------- #
# GATE A2 -- engine validation on a registered toy
# --------------------------------------------------------------------------- #
def gate_a2(lines: list[str]) -> None:
    enum = enumerate_exact(toy_function, TOY_EXPLICAND, TOY_BACKGROUND)
    got = enum.phi[:, 0]
    exact = np.array(TOY_EXACT_PHI, dtype=float)
    dev = np.abs(got - exact)
    worst = float(dev.max())
    resid = enum.efficiency_residual()

    rows = [
        [f"phi{i + 1}", f"`{float(exact[i])!r}`", f"`{float(got[i])!r}`",
         f"{dev[i]:.3e}"]
        for i in range(3)
    ]
    ok = worst < TOL_TOY and resid < TOL_TOY
    lines += [
        ("**GATE A2 PASS** -- " if ok else "**GATE A2 FAIL** -- ")
        + "the same enumeration engine, run on the registered toy "
        "`f(x1, x2, x3) = 2*x1 + 3*x2*x3` with explicand (1, 1, 1) and "
        "background (0, 0, 0); 2^3 = 8 coalitions evaluated once each:",
        "",
        md_table(["attribution", "exact Shapley value", "engine output", "abs dev"], rows),
        "",
        f"- max abs deviation from the closed form: `{worst!r}` "
        f"(= {worst:.3e}), tolerance {TOL_TOY:g}.",
        f"- engine efficiency residual on the toy: `{resid!r}` "
        f"(= {resid:.3e}), tolerance {TOL_TOY:g}; "
        f"v(N) = `{float(enum.v_full[0])!r}`, v(empty) = "
        f"`{float(enum.v_empty[0])!r}`.",
        f"- coalition weights for n = 3: "
        f"{[repr(w) for w in shapley_weights(3)]}.",
    ]
    assert worst < TOL_TOY, (
        f"GATE A2: max abs deviation {worst!r} >= {TOL_TOY:g} vs the registered "
        f"exact values {TOY_EXACT_PHI} -- STOP"
    )
    assert resid < TOL_TOY, (
        f"GATE A2: toy efficiency residual {resid!r} >= {TOL_TOY:g} -- STOP"
    )


# --------------------------------------------------------------------------- #
# GATE A3 -- linear-layer closed form
# --------------------------------------------------------------------------- #
def gate_a3(lines: list[str]) -> None:
    g, a, b = linear_map(LINEAR_EXPLICAND_DATE)
    x = fixtures.trailing_vol(LINEAR_EXPLICAND_DATE)     # vol_trail_20(t*)
    bg = fixtures.trailing_vol(LINEAR_BACKGROUND_DATE)   # vol_trail_20(2020-03-02)

    enum = enumerate_exact(g, x, bg)
    phi = enum.phi

    closed_diag = b * (x - bg)
    diag = np.diag(phi)
    diag_dev = np.abs(diag - closed_diag)
    worst_diag = float(diag_dev.max())

    off_mask = ~np.eye(len(TICKERS), dtype=bool)
    off = phi[off_mask]
    n_nonzero_off = int((off != 0.0).sum())
    worst_off = float(np.abs(off).max())

    lines += [
        ("**GATE A3 PASS** -- " if (worst_diag < TOL_CLOSED_FORM_DIAG and n_nonzero_off == 0)
         else "**GATE A3 FAIL** -- ")
        + "the same enumeration engine, run on the forecaster map "
        f"`g(x)_i = a_i(t*) + b_i(t*) * x_i` at the fixed date t* = "
        f"{LINEAR_EXPLICAND_DATE}; 2^8 = 256 coalitions evaluated once each. "
        f"Explicand: `vol_trail_20({LINEAR_EXPLICAND_DATE})`. Background: "
        f"`vol_trail_20({LINEAR_BACKGROUND_DATE})` -- calm vs crisis, the "
        "maximally separated registered inputs, so the check is non-trivial.",
        "",
        "Coefficient provenance: `src.task9.forecaster_deploy.fit_coeffs("
        f"{LINEAR_EXPLICAND_DATE})` -> `_fit_coeffs_cached` -> closed-form "
        "per-ticker OLS on every sample whose 20-day label window is realised "
        "by t*. Imported and used as returned; never retyped.",
        "",
        md_table(
            ["ticker", "a_i(t*)", "b_i(t*)", "explicand x_i", "background bg_i"],
            [
                [tk, f"`{float(a[i])!r}`", f"`{float(b[i])!r}`",
                 f"`{float(x[i])!r}`", f"`{float(bg[i])!r}`"]
                for i, tk in enumerate(TICKERS)
            ],
        ),
        "",
        "Full 8x8 attribution matrix at repr precision (rows = input features, "
        "columns = output components of g):",
        "",
        md_table(
            ["phi[i, j]", *TICKERS],
            [
                [TICKERS[i], *[f"`{float(phi[i, j])!r}`" for j in range(len(TICKERS))]]
                for i in range(len(TICKERS))
            ],
        ),
        "",
        "Diagonal vs closed form `phi[i, i] = b_i(t*) * (x_i - bg_i)`:",
        "",
        md_table(
            ["ticker", "closed form", "engine phi[i, i]", "abs dev"],
            [
                [tk, f"`{float(closed_diag[i])!r}`", f"`{float(diag[i])!r}`",
                 f"{diag_dev[i]:.3e}"]
                for i, tk in enumerate(TICKERS)
            ],
        ),
        "",
        f"- max abs diagonal deviation: `{worst_diag!r}` (= {worst_diag:.3e}), "
        f"tolerance {TOL_CLOSED_FORM_DIAG:g}.",
        f"- off-diagonal entries exactly 0.0: {56 - n_nonzero_off}/56 "
        f"(nonzero: {n_nonzero_off}); max abs off-diagonal `{worst_off!r}`. "
        "Each output of g depends on one input, so every off-diagonal marginal "
        "difference is a subtraction of bit-identical floats and every weighted "
        "sum of those is a sum of exact zeros.",
        f"- engine efficiency residual on g (recorded, engine arithmetic): "
        f"`{enum.efficiency_residual()!r}`.",
    ]
    assert worst_diag < TOL_CLOSED_FORM_DIAG, (
        f"GATE A3: max abs diagonal deviation {worst_diag!r} >= "
        f"{TOL_CLOSED_FORM_DIAG:g} vs the closed form b_i * (x_i - bg_i) -- STOP"
    )
    assert n_nonzero_off == 0, (
        f"GATE A3: {n_nonzero_off} off-diagonal entries are not exactly 0.0 "
        f"(max abs {worst_off!r}) -- the engine or the map is not what we think, "
        "STOP"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    lines = env_header(
        "Task 12 Stage A report -- engine, imports, closed-form validation",
    )
    lines += [
        "Explanation method: exact interventional Shapley, full 2^8 "
        "coalition enumeration, single-reference background. No sampling, no "
        "regression approximation, no shap-library estimator on the production "
        "attribution path.",
        "",
        "Scope boundary: this task evaluates the faithfulness of the "
        "*attributions* -- efficiency, exactness, stability. Narration "
        "faithfulness is Task 13's evaluation and appears nowhere in this "
        "report.",
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
    except AssertionError as exc:
        lines += [
            "",
            f"**GATE FAILURE -- STOP**: {exc}",
            "",
            "Diagnosis recorded.",
        ]
        report = write_report("stage_a_report.md", lines)
        print(report)
        sys.exit(1)

    lines += [
        "",
        "## Tolerance doctrine as applied",
        "",
        md_table(
            ["gate", "comparison", "standard"],
            [
                ["A2", "engine output vs a closed-form stub", f"{TOL_TOY:g}"],
                ["A2", "efficiency identity over the same cached values", f"{TOL_TOY:g}"],
                ["A3", "engine diagonal vs closed form", f"{TOL_CLOSED_FORM_DIAG:g}"],
                ["A3", "engine off-diagonal", "exactly 0.0"],
            ],
        ),
        "",
        "No comparison between two *distinct* L-BFGS-B solves arose in Stage A. "
        "None was planned; none appeared; nothing was gated at 1e-6 on that "
        "account. Stage A's explained functions are the registered toy and the "
        "linear map; neither invokes the solver.",
        "",
        "## Deliverables (Stage A)",
        "",
        "- `src/task12/__init__.py`, `engine.py`, `fixtures.py`, `probes.py`, "
        "`metrics.py`, `exhibits.py`, `reporting.py`, `stage_a.py`, "
        "`stage_b.py`, `stage_c.py`",
        "- `outputs/task12/stage_a_report.md`",
        "",
        "**STOP -- Stage A boundary.** Gates A1-A3 reported above."
    ]
    print(write_report("stage_a_report.md", lines))


if __name__ == "__main__":
    main()
