"""Task 12 Stage C driver: stability probes, verdict, exhibits.

Gates C1 (completeness), C2 (machinery under perturbation), C3 (verdict
emission -- procedural). The stability outcome is a finding, not a gate:
rho_max > 10 does not fail this stage; an incomplete table, an unregistered
probe, or a vocabulary deviation does.
"""

from __future__ import annotations

import sys

import numpy as np

from .engine import Enumeration, enumerate_exact
from .exhibits import x1_attribution_heatmap, x2_stability
from .fixtures import (
    ANCHOR_CELL,
    CONTROL_PAIR,
    EPS,
    MONITORED_PAIR,
    OUT_DIR,
    RHO_DENOM_FLOOR,
    RHO_THRESHOLD,
    TICKERS,
    TOL_EFFICIENCY,
    TOL_UNPLANNED_SOLVE_PAIR,
    background,
    cells,
    explicand,
    ticker_index,
    weights_fn,
)
from .metrics import Stability, stability
from .probes import ALL_PROBES, CONTROL_PROBES, MONITORED_PROBES, perturb
from .reporting import env_header, md_table, write_report


PAIR_LABEL = {
    MONITORED_PAIR: "-".join(MONITORED_PAIR),
    CONTROL_PAIR: "-".join(CONTROL_PAIR),
}


# --------------------------------------------------------------------------- #
# Prior-stage outcomes, read from the reports of record
# --------------------------------------------------------------------------- #
def prior_gate_outcomes() -> dict[str, bool]:
    """Gate A3 and Gate B3 verdicts, read from this task's own stage reports.

    The verdict names three components; two of them were established
    in earlier stages. Their status is read at execution time rather than
    assumed, so the sentence cannot claim something no report carries.
    """
    out = {}
    for gate, name in (("A3", "stage_a_report.md"), ("B3", "stage_b_report.md")):
        path = OUT_DIR / name
        assert path.exists(), (
            f"GATE C3: {name} is absent; the verdict cannot be emitted "
            f"without the Gate {gate} outcome -- STOP"
        )
        text = path.read_text(encoding="utf-8")
        passed = f"**GATE {gate} PASS**" in text
        failed = f"**GATE {gate} FAIL**" in text
        assert passed != failed, (
            f"GATE C3: {name} carries neither a single unambiguous "
            f"'GATE {gate} PASS' nor 'GATE {gate} FAIL' marker -- STOP"
        )
        out[gate] = passed
    return out


# --------------------------------------------------------------------------- #
# The probe grid
# --------------------------------------------------------------------------- #
def run_cell(date: str, profile: str) -> tuple[Enumeration, dict[str, Enumeration]]:
    """Base enumeration plus one enumeration per distinct registered probe.

    p1 belongs to both probe lists; it is enumerated once here and its cached
    values feed both pairs' metrics.
    """
    f = weights_fn(profile, date)
    x = explicand(date)
    b = background(date)          # never perturbed: a constant of the method
    base = enumerate_exact(f, x, b)
    probed = {p.name: enumerate_exact(f, perturb(x, p, EPS), b) for p in ALL_PROBES}
    return base, probed


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def gate_c1(
    lines: list[str],
    table: list[tuple[str, str, str, str, Stability]],
) -> None:
    expected = len(cells()) * (len(MONITORED_PROBES) + len(CONTROL_PROBES))
    names_seen = {(c[0], c[1], c[2], c[3]) for c in table}
    registered = {
        (date, profile, probe.name, PAIR_LABEL[pair])
        for date, profile in cells()
        for pair, probes in (
            (MONITORED_PAIR, MONITORED_PROBES),
            (CONTROL_PAIR, CONTROL_PROBES),
        )
        for probe in probes
    }
    missing = sorted(registered - names_seen)
    extra = sorted(names_seen - registered)
    ok = len(table) == expected and not missing and not extra

    lines += [
        ("**GATE C1 PASS** -- " if ok else "**GATE C1 FAIL** -- ")
        + f"completeness: {len(table)} of {expected} registered "
        "(cell, probe, pair) combinations computed -- 12 cells x 4 probes for "
        "the monitored pair "
        f"{PAIR_LABEL[MONITORED_PAIR]} and 12 x 4 for the control pair "
        f"{PAIR_LABEL[CONTROL_PAIR]}. No cell or probe was skipped or excluded; "
        "no unregistered probe appears.",
        "",
        md_table(
            ["check", "result"],
            [
                ["rows computed", f"{len(table)} (expected {expected})"],
                ["missing combinations", f"{len(missing)}" if not missing else f"**{missing}**"],
                ["unregistered combinations", f"{len(extra)}" if not extra else f"**{extra}**"],
                ["monitored probes", ", ".join(p.name for p in MONITORED_PROBES)],
                ["control probes", ", ".join(p.name for p in CONTROL_PROBES)],
                ["eps", f"{EPS:g} (multiplicative, explicand only)"],
                ["rho denominator floor", f"{RHO_DENOM_FLOOR:g}"],
            ],
        ),
    ]
    assert ok, (
        f"GATE C1: incomplete probe table -- {len(table)}/{expected} rows, "
        f"missing {missing}, unregistered {extra} -- STOP"
    )


def gate_c2(
    lines: list[str],
    residuals: list[tuple[str, str, str, float]],
    empty_gaps: list[tuple[str, str, str, float]],
) -> None:
    worst = max(residuals, key=lambda r: r[3])
    ok = worst[3] < TOL_EFFICIENCY
    worst_gap = max(empty_gaps, key=lambda r: r[3])

    lines += [
        ("**GATE C2 PASS** -- " if ok else "**GATE C2 FAIL** -- ")
        + "the efficiency identity holds inside EVERY probe enumeration: for "
        "each of the 12 x 7 = 84 distinct perturbed enumerations, "
        "`|sum_i phi'[i, j] - (v'_j(N) - v'_j(empty))| < "
        f"{TOL_EFFICIENCY:g}` at every output j. This is engine arithmetic over "
        "each probe's own cached values on both sides of its own identity.",
        "",
        md_table(
            ["quantity", "value"],
            [
                ["enumerations checked", f"{len(residuals)}"],
                ["worst residual", f"`{worst[3]!r}` (= {worst[3]:.3e})"],
                ["at", f"{worst[0]} / {worst[1]} / probe {worst[2]}"],
                ["tolerance", f"{TOL_EFFICIENCY:g}"],
            ],
        ),
        "",
        "**Right-hand side, stated precisely.** The background is never "
        "perturbed -- it is a constant of the method -- so each probe's own "
        "empty coalition evaluates `allocate` at the identical background "
        "vector, and 'perturbed v(N) minus unperturbed v(empty)' is exactly the "
        "probe's own identity as gated above.",
        "",
        "**A solve-vs-solve comparison arose and is reported.** "
        "Confirming that reading required comparing each probe enumeration's "
        "v'(empty) against the base enumeration's v(empty) -- two distinct "
        "L-BFGS-B solves. Per the tolerance doctrine such a comparison is gated "
        f"at {TOL_UNPLANNED_SOLVE_PAIR:g}, never tighter, and its appearance is "
        "itself reported. Worst observed gap "
        f"`{worst_gap[3]!r}` (= {worst_gap[3]:.3e}) at {worst_gap[0]} / "
        f"{worst_gap[1]} / probe {worst_gap[2]}. This comparison is a recorded "
        "observation about the machinery; it is NOT the C2 gate, which is the "
        "engine-arithmetic identity above.",
    ]
    assert ok, (
        f"GATE C2: worst probe efficiency residual {worst[3]!r} >= "
        f"{TOL_EFFICIENCY:g} at {worst[:3]} -- STOP"
    )
    assert worst_gap[3] < TOL_UNPLANNED_SOLVE_PAIR, (
        f"GATE C2 (reported solve-vs-solve comparison): worst v'(empty) vs "
        f"v(empty) gap {worst_gap[3]!r} >= {TOL_UNPLANNED_SOLVE_PAIR:g} at "
        f"{worst_gap[:3]} -- STOP"
    )


def verdict_sentence(
    rho_max: float, efficiency_ok: bool, closed_form_ok: bool, stability_ok: bool
) -> str:
    """Registered vocabulary, quoted verbatim with brackets filled."""
    if efficiency_ok and closed_form_ok and stability_ok:
        return (
            "Attribution faithfulness is supported: the exact-Shapley "
            "attributions satisfy the efficiency identity to engine precision "
            "at every cell, the linear forecaster's attributions equal their "
            "closed form, and SPY-QQQ attribution stability holds under every "
            f"registered perturbation probe (rho_max = {rho_max:.3g}, "
            "registered threshold 10)."
        )

    failing = []
    if not efficiency_ok:
        failing.append("efficiency")
    if not closed_form_ok:
        failing.append("closed form")
    if not stability_ok:
        failing.append("stability")
    components = " and ".join(failing) if len(failing) <= 2 else (
        ", ".join(failing[:-1]) + " and " + failing[-1]
    )
    if not stability_ok and len(failing) == 1:
        quantity = "rho_max"
        value = f"{rho_max:.3g}"
        against = "threshold 10"
    else:
        quantity = "the reported worst residual"
        value = "see the failing gate block above"
        against = "identity"
    held = "held as registered" if len(failing) == 1 else "also failed"
    return (
        f"Attribution faithfulness is not supported on {components}: "
        f"{quantity} = {value} against the registered {against}; the remaining "
        f"components {held}, and the result is reported as measured."
    )


def gate_c3(
    lines: list[str],
    table: list[tuple[str, str, str, str, Stability]],
    prior: dict[str, bool],
    c2_ok: bool,
) -> tuple[float, str]:
    monitored = [
        (d, p, pr, st) for d, p, pr, pair, st in table
        if pair == PAIR_LABEL[MONITORED_PAIR]
    ]
    assert len(monitored) == 48, (
        f"GATE C3: {len(monitored)} monitored-pair combinations, expected 48 -- STOP"
    )
    worst = max(monitored, key=lambda r: r[3].rho)
    rho_max = worst[3].rho

    efficiency_ok = prior["B3"] and c2_ok
    closed_form_ok = prior["A3"]
    stability_ok = rho_max <= RHO_THRESHOLD
    sentence = verdict_sentence(rho_max, efficiency_ok, closed_form_ok, stability_ok)

    lines += [
        "**GATE C3 PASS (procedural)** -- rho_max computed over the 48 "
        "monitored-pair (cell, probe) combinations and the verdict emitted in "
        "the registered Sec. 1.5 vocabulary with brackets filled. The stability "
        "OUTCOME is a finding, not a gate: this gate checks the table, the probe "
        "set and the vocabulary, not the number.",
        "",
        md_table(
            ["quantity", "value"],
            [
                ["rho_max (monitored pair, 48 combinations)",
                 f"`{rho_max!r}` (= {rho_max:.3g} to 3 s.f.)"],
                ["attained at", f"{worst[0]} / {worst[1]} / probe {worst[2]}"],
                ["registered threshold", f"{RHO_THRESHOLD:g}"],
                ["stability component", "holds" if stability_ok else "**does not hold**"],
                ["efficiency component (Gate B3 + Gate C2)",
                 "holds" if efficiency_ok else "**does not hold**"],
                ["closed-form component (Gate A3)",
                 "holds" if closed_form_ok else "**does not hold**"],
            ],
        ),
        "",
        "## Verdict",
        "",
        f"> {sentence}",
        "",
    ]
    return rho_max, sentence


# --------------------------------------------------------------------------- #
# Deliverable
# --------------------------------------------------------------------------- #
def write_probe_csv(table: list[tuple[str, str, str, str, Stability]]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ["date,profile,probe,pair,dphi_pair_max,dw_max,rho"]
    for date, profile, probe, pair, st in table:
        out.append(
            f"{date},{profile},{probe},{pair},"
            f"{st.dphi_pair_max!r},{st.dw_max!r},{st.rho!r}"
        )
    (OUT_DIR / "stability_probes.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(out) - 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    lines = env_header(
        "Task 12 Stage C report -- stability probes, verdict, exhibits",
    )
    lines += [
        f"Perturbation: eps = {EPS:g}, multiplicative, on the EXPLICAND entry "
        "only; the background is a constant of the method and is never "
        "perturbed. Each cell runs one base enumeration plus one enumeration per "
        "distinct registered probe (p1 belongs to both probe lists and is "
        "enumerated once), i.e. 12 x 8 x 256 = 24576 evaluations of `allocate`.",
        "",
        "Scope boundary (binding): this stage evaluates the faithfulness of the "
        "*attributions*. Narration faithfulness is Task 13's evaluation and "
        "appears nowhere in this report, its exhibits or its verdict.",
        "",
        "## Gates",
        "",
    ]
    try:
        prior = prior_gate_outcomes()

        table: list[tuple[str, str, str, str, Stability]] = []
        residuals: list[tuple[str, str, str, float]] = []
        empty_gaps: list[tuple[str, str, str, float]] = []
        monitored_points: dict[tuple[str, str], list[tuple[str, float]]] = {}
        control_points: dict[tuple[str, str], list[tuple[str, float]]] = {}
        anchor_phi = None

        m_idx = (ticker_index(MONITORED_PAIR[0]), ticker_index(MONITORED_PAIR[1]))
        c_idx = (ticker_index(CONTROL_PAIR[0]), ticker_index(CONTROL_PAIR[1]))

        for date, profile in cells():
            base, probed = run_cell(date, profile)
            if (date, profile) == ANCHOR_CELL:
                anchor_phi = base.phi
            monitored_points[(date, profile)] = []
            control_points[(date, profile)] = []

            for name, enum in probed.items():
                residuals.append((date, profile, name, enum.efficiency_residual()))
                empty_gaps.append(
                    (date, profile, name,
                     float(np.abs(enum.v_empty - base.v_empty).max()))
                )

            for pair, idx, probes, sink in (
                (MONITORED_PAIR, m_idx, MONITORED_PROBES, monitored_points),
                (CONTROL_PAIR, c_idx, CONTROL_PROBES, control_points),
            ):
                for probe in probes:
                    enum = probed[probe.name]
                    st = stability(base.phi, enum.phi, base.v_full, enum.v_full, idx)
                    table.append((date, profile, probe.name, PAIR_LABEL[pair], st))
                    sink[(date, profile)].append((probe.name, st.rho))

        assert anchor_phi is not None, (
            f"GATE C3: the anchor cell {ANCHOR_CELL} is not in the registered "
            "cell list -- STOP"
        )

        gate_c1(lines, table)
        lines.append("")
        gate_c2(lines, residuals, empty_gaps)
        c2_ok = max(r[3] for r in residuals) < TOL_EFFICIENCY
        lines.append("")
        rho_max, sentence = gate_c3(lines, table, prior, c2_ok)
        n_rows = write_probe_csv(table)

        x1 = x1_attribution_heatmap(anchor_phi, ANCHOR_CELL[0], ANCHOR_CELL[1])
        x2 = x2_stability(
            list(cells()),
            monitored_points,
            control_points,
            RHO_THRESHOLD,
            f"monitored pair {PAIR_LABEL[MONITORED_PAIR]} (gated)",
            f"control pair {PAIR_LABEL[CONTROL_PAIR]} (descriptive)",
        )
    except AssertionError as exc:
        lines += [
            "",
            f"**GATE FAILURE -- STOP**: {exc}",
            "",
            "Diagnosis recorded. "
            "stability_probes.csv and the exhibits are NOT written on a failed "
            "stage.",
        ]
        print(write_report("stage_c_report.md", lines))
        sys.exit(1)

    # ----------------------------------------------------------------------- #
    # Soft observations (recorded, not gated)
    # ----------------------------------------------------------------------- #
    monitored = [r for r in table if r[3] == PAIR_LABEL[MONITORED_PAIR]]
    control = [r for r in table if r[3] == PAIR_LABEL[CONTROL_PAIR]]
    worst = max(monitored, key=lambda r: r[4].rho)
    dw_all = [r[4].dw_max for r in table]
    m_rho = [r[4].rho for r in monitored]
    c_rho = [r[4].rho for r in control]

    lines += [
        "",
        "## Exhibits",
        "",
        f"- **X1** `{x1.rsplit(chr(92), 1)[-1]}` -- the 8x8 attribution matrix "
        f"at the anchor cell {ANCHOR_CELL[0]} / {ANCHOR_CELL[1]}, diverging "
        "colormap symmetric about zero, every cell also carrying its value as "
        "text, with the column sums annotated and labelled "
        "`w_advisor - w_classical`.",
        f"- **X2** `{x2.rsplit(chr(92), 1)[-1]}` -- rho for ALL 12 cells x 4 "
        "monitored probes (48 points) with the control-pair distribution "
        "alongside and the threshold line at "
        f"{RHO_THRESHOLD:g}. All cells shown regardless of outcome. Log y axis "
        "with `NullFormatter` on the y minor ticks (the DC-1 lesson).",
        "",
        "## Tolerance doctrine as applied",
        "",
        md_table(
            ["gate", "comparison", "standard"],
            [
                ["C1", "completeness of the registered table", "exact set equality"],
                ["C2", "efficiency identity inside each probe enumeration",
                 f"{TOL_EFFICIENCY:g}"],
                ["C2 (reported)", "v'(empty) vs v(empty) -- two distinct solves",
                 f"{TOL_UNPLANNED_SOLVE_PAIR:g}"],
                ["C3", "vocabulary and probe-set conformance", "procedural"],
                ["--", "dphi, dw, rho", "measurements, not gates"],
            ],
        ),
        "",
        "## Deliverables (Stage C)",
        "",
        f"- `outputs/task12/stability_probes.csv` ({n_rows} rows, full `repr()` "
        "precision)",
        "- `outputs/task12/X1_attribution_heatmap_anchor.png`",
        "- `outputs/task12/X2_stability.png`",
        "- `outputs/task12/stage_c_report.md`",
        "",
        "**STOP -- Stage C boundary.** Gates C1-C3 reported above. "
    ]
    print(write_report("stage_c_report.md", lines))


if __name__ == "__main__":
    main()
