"""Task 11 Stage A: cross-session determinism tie-in.

This stage writes no backtest code and consumes no engine. Its only job is to
prove the ability to reproduce the Task 9 anchors to full precision 
before Task 11 builds anything on top of them.

  * GATE A1 -- pooled train fit (a, b) repr-exact, plus the full per-ticker
               table cross-checked against outputs/task9/forecaster_trainfit.json.
  * GATE A2 -- covariance anchors at t = 2024-12-02: trace(Sigma_classical),
               Sigma_classical[SPY, AGG], R[SPY, QQQ], repr-exact.
  * GATE A3 -- rb_advisor balanced weights at t = 2024-12-02, all eight
               tickers, repr-exact. This one passes through the registered
               L-BFGS-B solve, so it is the strictest statement available:
               same session-independent solver trajectory, not merely the same
               optimum.
  * GATE A4 -- rebalance_grid(): 156 dates, first 2012-01-03, last 2024-12-02,
               contains the four Task 9 smoke dates. Records verbatim the grid
               dates 2019-12 .. 2020-06 (exhibit F4) and the named Feb/Mar/Apr
               2020 dates (H1-Q3).

On any gate failure: STOP
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

from ..data.config import ROOT, UNIVERSE
from ..task9.allocators import SMOKE_DATES, get_allocator, rebalance_grid
from ..task9.covariance import CORR_WINDOW, corr_matrix, sigma
from ..task9.forecaster_deploy import train_fit

TASK9_DIR = ROOT / "outputs" / "task9"
OUT_DIR = ROOT / "outputs" / "task11"

# --------------------------------------------------------------------------- #
# Registered anchors. These string literals are the Task 9 values;
# they are compared against repr(float(...)), so any difference fails.
# --------------------------------------------------------------------------- #
A1_POOLED_A = "0.0029325640193698533"
A1_POOLED_B = "0.6594918209185826"

A2_DATE = "2024-12-02"
A2_TRACE_SIGMA_CLASSICAL = "0.0006854447613844723"
A2_SIGMA_CLASSICAL_SPY_AGG = "3.883729281316424e-06"
A2_R_SPY_QQQ = "0.9438862936732487"

A3_DATE = "2024-12-02"
A3_RB_ADVISOR_BALANCED = {
    "SPY": "0.14109534938031104",
    "QQQ": "0.11734152046106551",
    "EFA": "0.060453491106364984",
    "EEM": "0.05533039347054884",
    "AGG": "0.28323468605737295",
    "IEF": "0.23508534268634776",
    "GLD": "0.05838695772086517",
    "VNQ": "0.04907225911712361",
}

A4_GRID_COUNT = 156
A4_GRID_FIRST = "2012-01-03"
A4_GRID_LAST = "2024-12-02"
A4_F4_WINDOW = ("2019-12-01", "2020-06-30")  # calendar bounds

_PASS: list[str] = []


def _repr_check(label: str, value: float, expected: str) -> str:
    """repr-exact comparison; returns a report line, raises AssertionError on miss."""
    got = repr(float(value))
    assert got == expected, (
        f"GATE mismatch on {label}: got `{got}`, registered `{expected}` -- "
        "cross-session determinism failure, STOP."
    )
    return f"- {label} = `{got}` — matches registered `{expected}` (repr-exact)."


# --------------------------------------------------------------------------- #
# GATE A1
# --------------------------------------------------------------------------- #
def gate_a1(lines: list[str]) -> None:
    fit = train_fit()
    lines += ["### GATE A1 — pooled and per-ticker train fit", ""]
    lines.append(_repr_check("pooled a", fit["pooled"]["a"], A1_POOLED_A))
    lines.append(_repr_check("pooled b", fit["pooled"]["b"], A1_POOLED_B))

    with open(TASK9_DIR / "forecaster_trainfit.json", encoding="utf-8") as fh:
        frozen = json.load(fh)

    # Cross-check of the per-ticker table against the frozen Task 9 artifact.
    rows, worst = [], 0.0
    for tk in UNIVERSE:
        ref = frozen["per_ticker"][tk]
        got_a, got_b = float(fit["per_ticker"][tk]["a"]), float(fit["per_ticker"][tk]["b"])
        ok_a = repr(got_a) == repr(float(ref["a"]))
        ok_b = repr(got_b) == repr(float(ref["b"]))
        assert ok_a and ok_b, (
            f"GATE A1: per-ticker fit for {tk} does not reproduce "
            f"forecaster_trainfit.json: got a={got_a!r}, b={got_b!r}; "
            f"frozen a={float(ref['a'])!r}, b={float(ref['b'])!r} -- STOP"
        )
        assert int(fit["per_ticker"][tk]["n"]) == int(ref["n"]), (
            f"GATE A1: {tk} fit row count {fit['per_ticker'][tk]['n']} != "
            f"frozen {ref['n']} -- STOP"
        )
        worst = max(worst, abs(got_a - float(ref["a"])), abs(got_b - float(ref["b"])))
        rows.append(
            {"ticker": tk, "a": repr(got_a), "b": repr(got_b),
             "n": int(ref["n"]), "repr_match": "yes"}
        )
    lines += [
        "",
        "Per-ticker cross-check against `outputs/task9/forecaster_trainfit.json` "
        "(repr-exact, all eight tickers):",
        "",
        _md_table(pd.DataFrame(rows)),
        "",
        f"Max elementwise |diff| vs the frozen artifact: {worst:.1e} "
        "",
        "**GATE A1 PASS** — pooled and per-ticker train-fit coefficients "
        "reproduce the registered Task 9 anchors repr-exact in this session.",
        "",
    ]
    _PASS.append("A1")

    # Recorded, not gated: the H1 premise reads off this table.
    bs = [float(fit["per_ticker"][tk]["b"]) for tk in UNIVERSE]
    lines += [
        f"Recorded (not a gate): per-ticker slope range b ∈ [{min(bs):.4f}, "
        f"{max(bs):.4f}] — the shrinkage premise of hypothesis H1 "
        "(b ∈ [0.26, 0.57]).",
        "",
    ]


# --------------------------------------------------------------------------- #
# GATE A2
# --------------------------------------------------------------------------- #
def gate_a2(lines: list[str]) -> None:
    t = pd.Timestamp(A2_DATE)
    S_cl = sigma(t, "classical")
    R = corr_matrix(t)
    lines += [
        f"### GATE A2 — covariance anchors at t = {A2_DATE}",
        "",
        f"Correlation window = {CORR_WINDOW} trading days (Task 9 module "
        "constant).",
        "",
    ]
    lines.append(
        _repr_check("trace(Sigma_classical)", np.trace(S_cl.to_numpy()),
                    A2_TRACE_SIGMA_CLASSICAL)
    )
    lines.append(
        _repr_check("Sigma_classical[SPY, AGG]", S_cl.at["SPY", "AGG"],
                    A2_SIGMA_CLASSICAL_SPY_AGG)
    )
    lines.append(_repr_check("R[SPY, QQQ]", R.at["SPY", "QQQ"], A2_R_SPY_QQQ))
    lines += [
        "",
        "**GATE A2 PASS** — all three Task 9 Stage B covariance anchors "
        "reproduce repr-exact.",
        "",
    ]
    _PASS.append("A2")


# --------------------------------------------------------------------------- #
# GATE A3
# --------------------------------------------------------------------------- #
def gate_a3(lines: list[str]) -> None:
    t = pd.Timestamp(A3_DATE)
    w = get_allocator("rb_advisor", "balanced").weights(t)
    lines += [
        f"### GATE A3 — rb_advisor balanced weights at t = {A3_DATE}",
        "",
        "Strictest anchor in Stage A: this vector is the output of the "
        "registered L-BFGS-B solve, so repr-exactness here asserts an identical "
        "solver trajectory across sessions, not merely an equivalent optimum.",
        "",
    ]
    for tk in UNIVERSE:
        lines.append(_repr_check(f"w[{tk}]", w[tk], A3_RB_ADVISOR_BALANCED[tk]))
    gap = abs(float(w.sum()) - 1.0)
    lines += [
        "",
        f"|sum(w) − 1| = {gap:.3e}; all weights ≥ 0: {bool((w >= 0).all())}.",
        "",
        "**GATE A3 PASS** — all eight weights reproduce repr-exact.",
        "",
    ]
    _PASS.append("A3")


# --------------------------------------------------------------------------- #
# GATE A4
# --------------------------------------------------------------------------- #
def gate_a4(lines: list[str]) -> pd.DatetimeIndex:
    grid = rebalance_grid()
    n = len(grid)
    assert n == A4_GRID_COUNT, (
        f"GATE A4: rebalance_grid() returned {n} dates, registered "
        f"{A4_GRID_COUNT} -- STOP"
    )
    assert grid[0] == pd.Timestamp(A4_GRID_FIRST), (
        f"GATE A4: first grid date {grid[0].date()}, registered {A4_GRID_FIRST} -- STOP"
    )
    assert grid[-1] == pd.Timestamp(A4_GRID_LAST), (
        f"GATE A4: last grid date {grid[-1].date()}, registered {A4_GRID_LAST} -- STOP"
    )
    missing = [s for s in SMOKE_DATES if pd.Timestamp(s) not in grid]
    assert not missing, f"GATE A4: smoke dates missing from grid: {missing} -- STOP"

    f4 = grid[
        (grid >= pd.Timestamp(A4_F4_WINDOW[0])) & (grid <= pd.Timestamp(A4_F4_WINDOW[1]))
    ]
    feb, mar, apr = [
        grid[(grid.year == 2020) & (grid.month == m)] for m in (2, 3, 4)
    ]
    for label, sel in (("Feb", feb), ("Mar", mar), ("Apr", apr)):
        assert len(sel) == 1, (
            f"GATE A4: {label} 2020 has {len(sel)} grid dates, expected exactly 1 -- STOP"
        )

    lines_local = [
        "### GATE A4 — rebalance grid",
        "",
        f"- `rebalance_grid()` length = **{n}** (registered {A4_GRID_COUNT}).",
        f"- first = **{grid[0].date()}** (registered {A4_GRID_FIRST}).",
        f"- last = **{grid[-1].date()}** (registered {A4_GRID_LAST}).",
        f"- contains all four Task 9 smoke dates {SMOKE_DATES}: yes.",
        "",
        "Recorded verbatim — grid dates 2019-12 through 2020-06 (the exhibit F4 "
        "x-axis, seven dates):",
        "",
        *[f"  {i + 1}. `{d.date()}`" for i, d in enumerate(f4)],
        "",
        "Recorded verbatim — the named H1-Q3 grid dates:",
        "",
        f"- Feb 2020 grid date = **`{feb[0].date()}`** (expected 2020-02-03).",
        f"- Mar 2020 grid date = **`{mar[0].date()}`** (expected 2020-03-02).",
        f"- Apr 2020 grid date = **`{apr[0].date()}`** (expected 2020-04-01).",
        "",
        "**GATE A4 PASS** — grid length, endpoints and smoke-date membership all "
        "match the registered values; the F4 window and the H1-Q3 dates are "
        "named above and are now fixed for Stage C.",
        "",
    ]
    lines.extend(lines_local)
    _PASS.append("A4")
    return grid


# --------------------------------------------------------------------------- #
# Report helpers
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.to_numpy()]
    return "\n".join([header, sep, *rows])

lines: list[str] = []


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines.extend([
        "# Task 11 Stage A report — determinism tie-in",
        "",
        "A fresh session reproduces the Task 9 anchors to full "
        "precision **before any Task 11 engine code exists**. Acceptance is "
        "repr-exactness: a mismatch agreeing to ≤ 1e-6 is still a "
        "STOP, because the object under test is cross-session determinism, not "
        "numerical adequacy. `src/task9/` is consumed as imported, unmodified.",
        "",
        "## Gates",
        "",
    ])
    try:
        gate_a1(lines)
        gate_a2(lines)
        gate_a3(lines)
        gate_a4(lines)
    except AssertionError as e:
        lines.extend([
            "",
            f"**GATE FAILURE — STOP**: {e}",
            "",
            f"Gates passed before the failure: {_PASS or 'none'}. Diagnosis "
            "recorded; nothing patched; no Task 11 engine code is written on a "
            "failed Stage A.",
        ])
        report = "\n".join(lines)
        (OUT_DIR / "stage_a_report.md").write_text(report, encoding="utf-8")
        print(report)
        sys.exit(1)

    lines.extend([
        "## Registered anchor values",
        "",
        "```",
        f"A1  pooled a = {A1_POOLED_A}",
        f"A1  pooled b = {A1_POOLED_B}",
        f"A2  trace(Sigma_classical) @ {A2_DATE} = {A2_TRACE_SIGMA_CLASSICAL}",
        f"A2  Sigma_classical[SPY, AGG] @ {A2_DATE} = {A2_SIGMA_CLASSICAL_SPY_AGG}",
        f"A2  R[SPY, QQQ] @ {A2_DATE}             = {A2_R_SPY_QQQ}",
        *[f"A3  rb_advisor balanced {tk} @ {A3_DATE} = {v}"
          for tk, v in A3_RB_ADVISOR_BALANCED.items()],
        f"A4  grid: {A4_GRID_COUNT} dates, {A4_GRID_FIRST} .. {A4_GRID_LAST}",
        "```",
        "",
        "## Deliverables",
        "",
        "- `src/task11/__init__.py`, `src/task11/stage_a.py`",
        "- `outputs/task11/stage_a_report.md`",
        "",
        "**STOP — Stage A boundary.** Gates A1–A4 all PASS. "
    ])
    report = "\n".join(lines)
    (OUT_DIR / "stage_a_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
