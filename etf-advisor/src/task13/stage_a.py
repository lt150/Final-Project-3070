"""Task 13 Stage A -- pin capture, determinism measurement, grounding tie-outs.

Gates:

- **A1** pin captured complete (every field present); invariants I1-I4
  implemented and exercised on the probe calls; I2 held.
- **A2** grounding tie-outs (a)-(c) all pass.  Any failure halts the task --
  this is machinery.
- **A3** the determinism check EXECUTED and the standard in force recorded.
  The OUTCOME is a measurement, not a pass/fail criterion; the gate is that it
  ran and is reported.

"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from . import (
    PROFILE_ORDER,
    ANCHOR_CELL,
    EFFICIENCY_TOL,
    FORBIDDEN_IMPORTS,
    I4_PROMPT_BUDGET,
    K_REPEATS,
    NARRATION_DIR,
    OUT_DIR,
    PROBE_PROMPT,
    ascii_escape,
    env_header,
    md_table,
    package_imports,
    write_report,
)
from . import client as C
from . import grounding as G


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    failures: list[str] = []

    # ---------------------------------------------------------------- pin --- #
    pin = C.capture_pin()
    absent = sorted(k for k, v in pin.items() if v == "ABSENT")

    lines += env_header("Task 13 Stage A -- pin capture, grounding tie-outs, "
                        "determinism measurement", pin)
    lines += [
        "## Module and boundary",
        "",
        "Build the narration layer: local-LLM narration (Ollama, `qwen3.5:9b`) of "
        "a frozen Recommendation and its attributions, plus a pre-registered "
        "evaluation of the narration's faithfulness TO those frozen artifacts.",
        "",
        "> The Task 12 verdict is frozen ground truth. If narration fails "
        "faithfulness, that is a Task 13 finding about the narration layer and "
        "re-litigates nothing upstream. attributions_main.csv, the "
        "Recommendation interface, and all Task 9-12 outputs are read-only.",
        "",
        "## Pin capture (Stage A step 1) -- imported from the live host",
        "",
    ]
    pin_rows = [[k, f"`{v}`"] for k, v in pin.items()]
    lines += [md_table(["field", "value as reported"], pin_rows), ""]
    lines += [
        f"Fields reported ABSENT: {absent if absent else 'none'}.",
    ]

    # ------------------------------------------------------ import posture --- #
    imports = package_imports()
    offenders = [r for r in imports if r[2] in FORBIDDEN_IMPORTS]
    lines += [
        "## Import posture",
        "",
        "No new Python installs are authorized in this task; `shap` and `torch` "
        "are imported nowhere in task13 code; all HTTP to Ollama is stdlib "
        "`urllib.request` (no `ollama` package, no `requests`).",
        "",
        md_table(["file", "statement", "target"],
                 [[a, f"`{b}`", f"`{c}`"] for a, b, c in imports]),
        "",
        f"Forbidden targets {list(FORBIDDEN_IMPORTS)}: "
        f"{'**' + str(len(offenders)) + ' PRESENT**' if offenders else '0 present'}.",
        "",
    ]
    if offenders:
        failures.append(f"forbidden imports present: {offenders}")

    # --------------------------------------------------------- grounding --- #
    g = G.load()
    lines += [
        "## Grounding tie-outs (Stage A step 2)",
        "",
        "### Schema binding -- READ from the frozen headers, never retyped",
        "",
        md_table(["surface", "resolved path / read value"],
                 [[k, f"`{v}`"] for k, v in g.provenance.items()]
                 + [["smoke_weights.csv resolution",
                     f"`{g.smoke_weights_path.as_posix()}` (registered rule: "
                     "task10 first, then task9)"],
                    ["canonical ticker order",
                     "`" + ", ".join(g.tickers) + "`"],
                    ["cell dates",
                     "`" + ", ".join(sorted({c.date for c in g.cells})) + "`"],
                    ["profile order", "`" + ", ".join(PROFILE_ORDER) + "`"],
                    ["attribution orientation",
                     "long form `date, profile, feature, output, phi`; "
                     "`feature` = input ticker, `output` = output ticker"]]),
        "",
        "Every grounding surface was supplied by a frozen CSV. `recommend()` was "
        "NOT called: `forecast_vols` are present on "
        "`smoke_recommendations.csv` as the `d_*` columns, so the "
        "fallback (calling frozen `recommend()` read-only with a repr-exact "
        "cross-check) was not needed and was not exercised.",
        "",
        "### (a) Loader reproduces the frozen CSV values repr-exactly",
        "",
    ]
    repr_rows = G.tieout_repr_exact(g)
    lines += [md_table(
        ["surface", "check", "values checked", "mismatches", "first mismatch"],
        [[a, f"`{b}`", c, d, e] for a, b, c, d, e in repr_rows]), ""]
    repr_bad = sum(r[3] for r in repr_rows)
    if repr_bad:
        failures.append(f"tie-out (a): {repr_bad} repr mismatches")

    comp_rows = G.tieout_completeness(g)
    lines += ["### (b) Cell completeness", "",
              md_table(["check", "expected", "got", "verdict"], comp_rows), ""]
    comp_bad = [r for r in comp_rows if r[3] != "PASS"]
    if comp_bad:
        failures.append(f"tie-out (b): {len(comp_bad)} completeness failures")

    eff_rows, eff_worst = G.tieout_efficiency(g, EFFICIENCY_TOL)
    lines += [
        "### (c) Efficiency cross-check",
        "",
        "Per (cell, output): `sum_features phi == w_advisor - w_classical`, "
        f"tolerance {EFFICIENCY_TOL:.0e} (engine arithmetic). This reproduces the "
        "certified Task 12 identity purely to prove the loaders read the "
        "artifacts correctly; it re-litigates nothing upstream.",
        "",
        md_table(["date", "profile", "max abs gap", "verdict"], eff_rows),
        "",
        f"Worst gap over all 12 cells: `{eff_worst:.6e}` "
        f"(tolerance `{EFFICIENCY_TOL:.0e}`).",
        "",
    ]
    if eff_worst > EFFICIENCY_TOL:
        failures.append(f"tie-out (c): worst gap {eff_worst:.3e} > {EFFICIENCY_TOL:.0e}")

    # ------------------------------------------------------- determinism --- #
    lines += [
        "## Determinism measurement (Stage A step 3)",
        "",
        f"Anchor cell `{ANCHOR_CELL[0]} / {ANCHOR_CELL[1]}`; registered probe "
        "prompt (fixed, deliberately NOT the production template):",
        "",
        f"> {PROBE_PROMPT}",
        "",
        f"One discarded warm-up call, then k = {K_REPEATS} warm generations under "
        "the frozen options block.",
        "",
    ]
    warm = C.generate(PROBE_PROMPT, purpose="stage_a:warmup_discarded")
    reps = [C.generate(PROBE_PROMPT, purpose=f"stage_a:probe_rep{i + 1}")
            for i in range(K_REPEATS)]
    calls = [warm, *reps]

    texts = [r.response for r in reps]
    identical = all(t == texts[0] for t in texts)
    first_div = "-"
    if not identical:
        for i, t in enumerate(texts[1:], start=2):
            if t != texts[0]:
                off = next((j for j, (x, y) in enumerate(zip(texts[0], t)) if x != y),
                           min(len(texts[0]), len(t)))
                first_div = f"rep1 vs rep{i} at char offset {off}"
                break

    lines += [
        md_table(
            ["seq", "purpose", "prompt_eval_count", f"I4 (<= {I4_PROMPT_BUDGET})",
             "eval_count", "done_reason", "load_duration_s", "wall_s", "eval tok/s"],
            [[r.seq, r.purpose, r.i4_prompt_eval_count,
              "PASS" if r.i4_ok else "**BREACH**", r.eval_count,
              f"`{r.i3_done_reason}`", f"{r.load_duration_ns / 1e9:.2f}",
              f"{r.wall_seconds:.2f}", f"{r.eval_tokens_per_second:.2f}"]
             for r in calls]),
        "",
        md_table(["invariant", "exercised on", "outcome"],
                 [["I1 model == pinned tag", f"{len(calls)} calls",
                   "held on all" if all(r.i1_model_ok for r in calls) else "**BREACH**"],
                  ["I2 no thinking content", f"{len(calls)} calls",
                   "held on all" if all(r.i2_no_thinking_ok for r in calls)
                   else "**BREACH**"],
                  ["I3 done == true", f"{len(calls)} calls",
                   "held on all" if all(r.i3_done for r in calls) else "**BREACH**"],
                  ["I4 prompt_eval_count <= 7680", f"{len(calls)} calls",
                   "held on all" if all(r.i4_ok for r in calls) else "**BREACH**"]]),
        "",
        f"I2 evidence strings, per call: "
        f"{[r.i2_thinking_evidence for r in calls]}.",
        "",
        "### Byte-identical comparison across the k = 3 repeats",
        "",
        md_table(["repeat", "response chars", "sha256", "== rep1"],
                 [[i + 1, len(t), __import__("hashlib").sha256(t.encode()).hexdigest()[:16],
                   "yes" if t == texts[0] else "**no**"]
                  for i, t in enumerate(texts)]),
        "",
        f"**Byte-identical across k = {K_REPEATS}: "
        f"{'YES' if identical else 'NO'}.** First divergence: {first_div}.",
        "",
    ]

    standard = (
        "Determinism standard IN FORCE: **repeat-exact** -- the k = 3 repeats of "
        "the registered probe were byte-identical, so the registered k = 3 "
        "byte-identical standard is carried into Stage C unmodified."
        if identical else
        "Determinism standard IN FORCE: **pre-authorized fallback (brief Sec. "
        "0.2) is ARMED** -- the probe repeats were not byte-identical. Under the "
        "fallback the FIRST generation per cell is frozen as the evaluated "
        "transcript, the non-determinism is recorded symmetrically as a finding, "
        "and the faithfulness verdict attaches to the frozen transcripts. "
        "Activation requires no amendment; Stage C measures and records the "
        "per-cell pattern."
    )
    tps = [r.eval_tokens_per_second for r in reps if r.eval_count > 0]
    mean_tps = sum(tps) / len(tps) if tps else float("nan")
    est_s = (36 * 512 / mean_tps) if mean_tps == mean_tps and mean_tps > 0 else float("nan")
    lines += [
        standard, "",
        "Reproducibility standard, stated in the registered words: "
        "**repr-exact where achievable; transcript-freeze with disclosure where "
        "not.** Narration is the first component in the stack without "
        "repr-exact guarantees.",
        "",
    ]

    # probe transcripts + call log land on disk as evidence
    for i, t in enumerate(texts, start=1):
        (NARRATION_DIR / f"probe_rep{i}.txt").write_text(t, encoding="utf-8")
    (NARRATION_DIR / "probe_warmup_discarded.txt").write_text(
        warm.response, encoding="utf-8")
    with (OUT_DIR / "stage_a_call_log.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=C.LOG_COLUMNS)
        w.writeheader()
        for r in calls:
            w.writerow(r.log_row())
    with (OUT_DIR / "determinism_log.csv").open("w", encoding="utf-8", newline="") as fh:
        w2 = csv.writer(fh)
        w2.writerow(["stage", "cell", "repeat", "response_chars", "sha256",
                     "identical_to_rep1"])
        for i, t in enumerate(texts, start=1):
            w2.writerow(["A_probe", f"{ANCHOR_CELL[0]}_{ANCHOR_CELL[1]}", i, len(t),
                         __import__("hashlib").sha256(t.encode()).hexdigest(),
                         t == texts[0]])

    esc, n_non_ascii = ascii_escape(texts[0])
    lines += [
        "### Probe transcript, repeat 1",
        "",
        "```",
        esc,
        "```",
        "",
    ]

    # ------------------------------------------------------------- gates --- #
    a1 = (not absent) and all(r.i1_model_ok and r.i2_no_thinking_ok and r.i3_done
                              for r in calls) and not offenders
    a2 = not failures
    a3 = True   # the check ran and the standard in force is recorded above
    lines += [
        "## Gates",
        "",
        md_table(["gate", "criterion", "evidence", "verdict"], [
            ["A1", "pin captured complete; I1-I4 implemented and exercised on the "
                   "probe calls; I2 held",
             f"{len(pin)} pin fields, {len(absent)} ABSENT; {len(calls)} probe "
             f"calls; I2 held on all",
             "PASS" if a1 else "**FAIL**"],
            ["A2", "grounding tie-outs (a)-(c) all pass",
             f"(a) {repr_bad} mismatches over "
             f"{sum(r[2] for r in repr_rows)} values; (b) {len(comp_bad)} "
             f"failures over {len(comp_rows)} checks; (c) worst gap "
             f"{eff_worst:.3e} <= {EFFICIENCY_TOL:.0e}",
             "PASS" if a2 else "**FAIL**"],
            ["A3", "determinism check EXECUTED and standard in force recorded",
             f"warm-up + k={K_REPEATS} probe generations executed; "
             f"byte-identical = {identical}; recorded standard",
             "PASS" if a3 else "**FAIL**"],
        ]),
        "",
    ]
    if failures:
        lines += ["**HALT.** Gate A2 covers machinery, so any failure halts the "
                  "task:", ""] + [f"- {f}" for f in failures] + [""]

    lines += [
        "## Deliverables written by this stage",
        "",
        "- `outputs/task13/stage_a_report.md` (this file)",
        "- `outputs/task13/stage_a_call_log.csv` (per-call log, 4 calls)",
        "- `outputs/task13/determinism_log.csv` (probe repeats)",
        "- `outputs/task13/narrations/probe_rep{1,2,3}.txt`, "
        "`probe_warmup_discarded.txt`",
        "",
        ("STAGE A COMPLETE -- gates A1/A2/A3 PASS. "
         if (a1 and a2 and a3) else
         "STAGE A HALTED -- see the gate table."),
        "",
    ]

    report = write_report("stage_a_report.md", lines)
    print(report[-2600:])
    return 0 if (a1 and a2 and a3) else 1


if __name__ == "__main__":
    sys.exit(main())
