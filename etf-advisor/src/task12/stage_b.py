"""Task 12 Stage B driver: the 12-cell main grid and the two tie-outs.

Gates B1 (full-coalition tie-out), B2 (empty-coalition tie-out -- the
ablation-twin identity), B3 (efficiency). Deliverable:
``outputs/task12/attributions_main.csv``, 12 x 64 = 768 rows at full repr
precision, written only after every gate passes.
"""

from __future__ import annotations

import re
import sys

import numpy as np

from .engine import Enumeration, enumerate_exact
from .fixtures import (
    OUT_DIR,
    RECORD_TASK9_SMOKE,
    RECORD_TASK9_STAGE_B,
    RECORD_TASK9_STAGE_C,
    RECORD_TASK10_SMOKE,
    TICKERS,
    TOL_EFFICIENCY,
    background,
    cells,
    explicand,
    weights_fn,
)
from .reporting import (
    PRODUCTION_PATH_MODULES,
    STAGE_A_ONLY_FIXTURES,
    env_header,
    md_table,
    stage_a_only_scope,
    write_report,
)


# --------------------------------------------------------------------------- #
# Files of record
# --------------------------------------------------------------------------- #
def _roundtrip(cells_: list[str]) -> bool:
    """True when every stored cell satisfies ``repr(float(cell)) == cell``."""
    return all(repr(float(c)) == c for c in cells_)


def read_task10_smoke() -> tuple[dict[tuple[str, str], list[str]], bool]:
    """The 12 weight vectors of ``outputs/task10/smoke_recommendations.csv``."""
    rows = RECORD_TASK10_SMOKE.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    want = [f"w_{tk}" for tk in TICKERS]
    assert header[2:10] == want, (
        f"GATE B1: {RECORD_TASK10_SMOKE.name} weight columns {header[2:10]} != "
        f"{want} -- STOP"
    )
    out: dict[tuple[str, str], list[str]] = {}
    stored: list[str] = []
    for line in rows[1:]:
        c = line.split(",")
        out[(c[0], c[1])] = c[2:10]
        stored += c[2:10]
    assert len(out) == 12, f"GATE B1: {len(out)} rows, expected 12 -- STOP"
    return out, _roundtrip(stored)


def read_task9_classical() -> tuple[dict[tuple[str, str], list[str]], bool]:
    """The 12 ``rb_classical`` rows of ``outputs/task9/smoke_weights.csv``."""
    rows = RECORD_TASK9_SMOKE.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    assert tuple(header[3:]) == TICKERS, (
        f"GATE B2: {RECORD_TASK9_SMOKE.name} ticker columns {tuple(header[3:])} "
        f"!= {TICKERS} -- STOP"
    )
    out: dict[tuple[str, str], list[str]] = {}
    stored: list[str] = []
    for line in rows[1:]:
        c = line.split(",")
        if c[1] != "rb_classical":
            continue
        out[(c[0], c[2])] = c[3:]
        stored += c[3:]
    assert len(out) == 12, (
        f"GATE B2: {len(out)} rb_classical rows, expected 12 -- STOP"
    )
    return out, _roundtrip(stored)


# --------------------------------------------------------------------------- #
# The main grid
# --------------------------------------------------------------------------- #
def run_grid() -> dict[tuple[str, str], Enumeration]:
    out: dict[tuple[str, str], Enumeration] = {}
    for date, profile in cells():
        out[(date, profile)] = enumerate_exact(
            weights_fn(profile, date), explicand(date), background(date)
        )
    return out


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def _tie_out(
    lines: list[str],
    gate: str,
    grid: dict[tuple[str, str], Enumeration],
    ref: dict[tuple[str, str], list[str]],
    roundtrip: bool,
    getter,
    intro: str,
) -> None:
    branch = (
        "full round-trip precision (every cell satisfies "
        "`repr(float(cell)) == cell`), so the **repr-exact branch** applies"
        if roundtrip
        else "less than full round-trip precision -- the registered standard is "
        "repr-exact, so this is a **halt**"
    )
    assert roundtrip, (
        f"{gate}: the file of record does not carry full round-trip precision; "
        "the registered repr-exact tie-out cannot be performed -- STOP"
    )
    rows, bad = [], []
    for cell in cells():
        stored = ref[cell]
        got = getter(grid[cell])
        mism = [
            (tk, s, repr(float(g)))
            for tk, s, g in zip(TICKERS, stored, got)
            if repr(float(g)) != s
        ]
        rows.append([cell[0], cell[1],
                     "8/8 match" if not mism else f"**{len(mism)} MISMATCH**"])
        bad += [(cell, *m) for m in mism]
    lines += [
        (f"**{gate} PASS** -- " if not bad else f"**{gate} FAIL** -- ") + intro
        + f" Stored precision inspected first: the file carries {branch}. All "
        "96 components compared:",
        "",
        md_table(["date", "profile", "components vs record"], rows),
    ]
    assert not bad, f"{gate}: repr mismatches {bad} -- STOP"


def gate_b1(lines: list[str], grid: dict[tuple[str, str], Enumeration]) -> None:
    ref, roundtrip = read_task10_smoke()
    _tie_out(
        lines, "GATE B1", grid, ref, roundtrip, lambda e: e.v_full,
        "v(N) -- the full coalition, every feature taken from the explicand -- "
        f"at every cell against the 12 rows of `{RECORD_TASK10_SMOKE.name}`, the "
        "frozen rb_advisor recommendation.",
    )


def gate_b2(lines: list[str], grid: dict[tuple[str, str], Enumeration]) -> None:
    ref, roundtrip = read_task9_classical()
    _tie_out(
        lines, "GATE B2", grid, ref, roundtrip, lambda e: e.v_empty,
        "v(empty) -- the empty coalition, every feature taken from the "
        "background -- at every cell against the 12 `rb_classical` rows of "
        f"`{RECORD_TASK9_SMOKE.name}`. This certifies that the registered "
        "background is EXACTLY the classical twin's input and hence that the "
        "attributions decompose w_advisor - w_classical.",
    )


def gate_b3(lines: list[str], grid: dict[tuple[str, str], Enumeration]) -> None:
    rows, worst, worst_at = [], 0.0, None
    for cell in cells():
        enum = grid[cell]
        resid = enum.efficiency_residuals()
        cell_worst = float(resid.max())
        j = int(np.argmax(resid))
        if cell_worst > worst:
            worst, worst_at = cell_worst, (cell, TICKERS[j])
        rows.append([cell[0], cell[1], f"{cell_worst:.3e}", TICKERS[j],
                     "PASS" if cell_worst < TOL_EFFICIENCY else "**FAIL**"])
    ok = worst < TOL_EFFICIENCY
    lines += [
        ("**GATE B3 PASS** -- " if ok else "**GATE B3 FAIL** -- ")
        + "efficiency identity `|sum_i phi[i, j] - (v_j(N) - v_j(empty))|` at "
        f"every cell and every output, tolerance {TOL_EFFICIENCY:g}. This is "
        "linear algebra over the SAME cached coalition values appearing on both "
        "sides -- engine arithmetic, not a solve-vs-solve comparison:",
        "",
        md_table(["date", "profile", "worst residual", "at output", "verdict"], rows),
        "",
        f"- worst residual over all 12 x 8 = 96 (cell, output) checks: "
        f"`{worst!r}` (= {worst:.3e}) at {worst_at}.",
    ]
    assert ok, (
        f"GATE B3: worst efficiency residual {worst!r} >= {TOL_EFFICIENCY:g} at "
        f"{worst_at} -- STOP"
    )


# --------------------------------------------------------------------------- #
# Deliverable
# --------------------------------------------------------------------------- #
def write_attributions(grid: dict[tuple[str, str], Enumeration]) -> int:
    """Long-format attributions at full repr precision, from the cached values."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ["date,profile,feature,output,phi"]
    for date, profile in cells():
        phi = grid[(date, profile)].phi
        for i, feature in enumerate(TICKERS):
            for j, output in enumerate(TICKERS):
                out.append(
                    f"{date},{profile},{feature},{output},{float(phi[i, j])!r}"
                )
    (OUT_DIR / "attributions_main.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(out) - 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    lines = env_header(
        "Task 12 Stage B report -- main grid and tie-outs",
    )
    lines += [
        "12 registered cells (4 smoke dates x 3 profiles), one exact "
        "interventional Shapley enumeration each: 2^8 = 256 coalitions per cell, "
        "3072 evaluations of `allocate` in total, each a distinct L-BFGS-B solve. "
        "Explicand d_forecast(t) = `forecast_vol(t)`; background d_classical(t) = "
        "`vol_trail_20(t)`, the ablation twin's D. Profile is a conditioning "
        "variable, never a feature.",
        "",
        "All reference values are read from the on-disk files of record at "
        f"execution time, never retyped: `{RECORD_TASK10_SMOKE.name}`, "
        f"`{RECORD_TASK9_SMOKE.name}`, `{RECORD_TASK9_STAGE_B.name}`, "
        f"`{RECORD_TASK9_STAGE_C.name}`.",
        "",
        "## Gates",
        "",
    ]
    try:
        grid = run_grid()
        gate_b1(lines, grid)
        lines.append("")
        gate_b2(lines, grid)
        lines.append("")
        gate_b3(lines, grid)
        n_rows = write_attributions(grid)
    except AssertionError as exc:
        lines += [
            "",
            f"**GATE FAILURE -- STOP**: {exc}",
            "",
            "Diagnosis recorded; nothing patched; awaiting instruction. "
            "attributions_main.csv is NOT written on a failed stage.",
        ]
        print(write_report("stage_b_report.md", lines))
        sys.exit(1)

    lines += [
        "",
        "## Tolerance doctrine as applied",
        "",
        md_table(
            ["gate", "comparison", "standard"],
            [
                ["B1", "identical-trajectory reproduction of the frozen "
                       "recommendation", "repr-exact"],
                ["B2", "identical-trajectory reproduction of the classical twin",
                 "repr-exact"],
                ["B3", "efficiency identity over the same cached values",
                 f"{TOL_EFFICIENCY:g}"],
            ],
        ),
        "",
        "Each of the 3072 coalition evaluations is a distinct L-BFGS-B solve, "
        "but no two solves are ever compared for agreement inside the "
        "attribution arithmetic -- coalition values are data. The only "
        "solve-vs-solve agreement checks in this stage are B1 and B2, both "
        "identical-trajectory reproductions of a frozen record and therefore "
        "repr-exact. No unplanned solve-vs-solve comparison arose; nothing was "
        "gated at 1e-6 on that account.",
        "",
        "## Deliverables (Stage B)",
        "",
        f"- `outputs/task12/attributions_main.csv` ({n_rows} rows, long format "
        "`date, profile, feature, output, phi`, full `repr()` precision)",
        "- `outputs/task12/stage_b_report.md`",
        "",
        "**STOP -- Stage B boundary.** Gates B1-B3 reported above. ",
    ]
    print(write_report("stage_b_report.md", lines))


if __name__ == "__main__":
    main()
