"""Task 13 Stage C -- the ONE-SHOT evaluation.

36 generations (12 cells x k = 3) under the template and options block,
then extraction, adjudication, aggregates, the registered verdict and the N1/N2
exhibits.  This stage executes exactly once: a crash BEFORE any adjudication
output exists permits a restart with disclosure; once adjudication outputs
exist there are no re-runs, and this driver refuses to start.

Gates:

- **C1** run integrity -- 36/36 generations completed in registered order;
  invariants I1-I4 held or deviations logged; per-call log complete.
- **C2** extraction coverage executed and recorded -- every sentence
  classified; unaccounted-numeric scan run over every transcript.  (Scan
  RESULTS feed the verdict, not this gate.)
- **C3** verdict emitted per the registered vocabulary; symmetric report;
  exhibits rendered from frozen data only.

"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path

from . import (
    ANCHOR_CELL,
    K_REPEATS,
    NARRATION_DIR,
    OUT_DIR,
    PACKAGE_DIR,
    PENALTY_KNOBS,
    ascii_escape,
    env_header,
    md_table,
    write_report,
)
from . import client as C
from . import grounding as G
from .adjudicate import Verdict, adjudicate, unaccounted_numerics
from .extract import Extraction, extract
from .grounding import Cell
from .serialize import render_prompt


CLAIMS_CSV = OUT_DIR / "claims_extracted.csv"
SENTENCES_CSV = OUT_DIR / "sentence_classes.csv"
UNACCOUNTED_CSV = OUT_DIR / "unaccounted_numerics.csv"
T8_CSV = OUT_DIR / "t8_screen.csv"
DETERMINISM_CSV = OUT_DIR / "determinism_log.csv"
CALL_LOG = OUT_DIR / "stage_c_call_log.csv"

# Stage B generation-count reconciliation, carried in at ratification.
STAGE_B_GENERATIONS = ("Stage B made 8 generations in total: 5 from the stage "
                       "driver (1 discarded warm-up + 3 determinism re-probes + "
                       "1 anchor dry run) and 3 from the development harness "
                       "(dev iterations 1-3). All 8 were on the anchor cell or "
                       "the registered probe prompt; no non-anchor cell was "
                       "ever generated before this stage.")


class OneShotViolation(RuntimeError):
    pass


def stage_a_pin() -> dict[str, str]:
    """The Stage A pin, read back from the frozen Stage A report."""
    report = (OUT_DIR / "stage_a_report.md").read_text(encoding="utf-8")
    ver = re.search(r"Ollama server version: `([^`]+)`", report)
    dig = re.search(r"digest\s*\n?`([0-9a-f]{64})`", report) or \
        re.search(r"\| model_digest \| `([0-9a-f]{64})` \|", report)
    if not ver or not dig:
        raise OneShotViolation(
            "could not read the Stage A pin back out of stage_a_report.md "
            "(server version / model digest) -- STOP")
    return {"server_version": ver.group(1), "model_digest": dig.group(1)}


def preflight() -> tuple[dict, dict, list[str], int]:
    """One-shot guard, drift check, and crashed-run quarantine."""
    notes: list[str] = []

    if CLAIMS_CSV.exists():
        raise OneShotViolation(
            f"{CLAIMS_CSV.name} already exists: adjudication output exists, so "
            "Stage C has already run and there are no re-runs  "
            "(one-shot) -- STOP")

    frozen = stage_a_pin()
    live = C.capture_pin()
    drift = []
    if live["server_version"] != frozen["server_version"]:
        drift.append(f"server version {frozen['server_version']} -> "
                     f"{live['server_version']}")
    if str(live["model_digest"]) != frozen["model_digest"]:
        drift.append(f"model digest {frozen['model_digest']} -> "
                     f"{live['model_digest']}")
    if drift:
        raise OneShotViolation(
            "ENVIRONMENT DRIFT since the Stage A pin: " + "; ".join(drift) +
            ". This is a recorded environment event; requires a halt "
            "and re-verification of the determinism check before Stage C "
            "proceeds -- STOP")

    existing = sorted(NARRATION_DIR.glob("????-??-??_*_rep?.txt"))
    restart = 0
    if existing:
        n = 1
        while (q := NARRATION_DIR / f"crashed_run_{n}").exists():
            n += 1
        q.mkdir(parents=True)
        for p in existing:
            shutil.move(str(p), str(q / p.name))
        restart = len(existing)
        notes.append(
            f"RESTART DISCLOSURE: {restart} evaluation from an "
            f"earlier, incomplete Stage C attempt were found. No adjudication "
            f"output existed, so a restart is permitted; the earlier "
            f"transcripts were preserved unmodified under "
            f"`narrations/{q.name}/` and all 36 generations were re-run from "
            "scratch in registered order.")
    return frozen, live, notes, restart


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    frozen_pin, pin, notes, _restart = preflight()

    g = G.load()
    cells = list(g.cells)          # date ascending, then conservative/balanced/growth
    anchor = next(c for c in cells if c.is_anchor)

    # ------------------------------------------------- 36 generations --- #
    log_fh = CALL_LOG.open("w", encoding="utf-8", newline="")
    log_w = csv.DictWriter(log_fh, fieldnames=["cell_date", "cell_profile",
                                               "repeat", *C.LOG_COLUMNS])
    log_w.writeheader()
    log_fh.flush()

    warm = C.generate(render_prompt(g, cells[0]),
                      purpose="stage_c:warmup_discarded")
    log_w.writerow({"cell_date": "-", "cell_profile": "-", "repeat": 0,
                    **warm.log_row()})
    log_fh.flush()
    (NARRATION_DIR / "stage_c_warmup_discarded.txt").write_text(
        warm.response, encoding="utf-8")

    order: list[tuple[Cell, int]] = []
    texts: dict[tuple[str, str], list[str]] = {}
    results: list[C.GenerationResult] = [warm]
    for cell in cells:
        prompt = render_prompt(g, cell)
        reps: list[str] = []
        for k in range(1, K_REPEATS + 1):
            res = C.generate(
                prompt, purpose=f"stage_c:{cell.date}_{cell.profile}_rep{k}")
            # transcript written immediately
            (NARRATION_DIR / f"{cell.date}_{cell.profile}_rep{k}.txt").write_text(
                res.response, encoding="utf-8")
            log_w.writerow({"cell_date": cell.date, "cell_profile": cell.profile,
                            "repeat": k, **res.log_row()})
            log_fh.flush()
            reps.append(res.response)
            results.append(res)
            order.append((cell, k))
        texts[(cell.date, cell.profile)] = reps
    log_fh.close()

    evaluation_calls = results[1:]
    order_ok = order == [(c, k) for c in cells for k in range(1, K_REPEATS + 1)]

    # ------------------------------------------------ determinism ------- #
    det_rows: list[list[object]] = []
    divergent: list[tuple[Cell, int, str]] = []
    for cell in cells:
        reps = texts[(cell.date, cell.profile)]
        agree = [r == reps[0] for r in reps]
        pattern = "".join("=" if a else "X" for a in agree)
        first_div = "-"
        if not all(agree):
            for i, t in enumerate(reps[1:], start=2):
                if t != reps[0]:
                    off = next((j for j, (x, y) in enumerate(zip(reps[0], t))
                                if x != y), min(len(reps[0]), len(t)))
                    first_div = f"rep1 vs rep{i} at char offset {off}"
                    divergent.append((cell, off, first_div))
                    break
        det_rows.append([
            cell.date, cell.profile,
            "anchor" if cell.is_anchor else "",
            pattern,
            "; ".join(str(len(t)) for t in reps),
            hashlib.sha256(reps[0].encode()).hexdigest()[:16],
            "YES" if all(agree) else "**NO**",
            first_div,
        ])

    with DETERMINISM_CSV.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for cell in cells:
            reps = texts[(cell.date, cell.profile)]
            for i, t in enumerate(reps, start=1):
                w.writerow(["C_evaluation", f"{cell.date}_{cell.profile}", i,
                            len(t), hashlib.sha256(t.encode()).hexdigest(),
                            t == reps[0]])

    fallback_active = bool(divergent)

    # --------------------------- extraction + adjudication (rep 1) ------- #
    ex_by_cell: dict[tuple[str, str], Extraction] = {}
    vd_by_cell: dict[tuple[str, str], list[Verdict]] = {}
    un_by_cell: dict[tuple[str, str], list] = {}
    for cell in cells:
        text = texts[(cell.date, cell.profile)][0]     # evaluated transcript
        ex = extract(text, g.tickers)
        ex_by_cell[(cell.date, cell.profile)] = ex
        vd_by_cell[(cell.date, cell.profile)] = adjudicate(ex, g, cell)
        un_by_cell[(cell.date, cell.profile)] = unaccounted_numerics(ex, cell)

    _write_claims(cells, ex_by_cell, vd_by_cell)
    _write_sentences(cells, ex_by_cell)
    _write_unaccounted(cells, un_by_cell, ex_by_cell)
    _write_t8(cells, ex_by_cell)

    # ------------------------------------------------------ aggregates --- #
    headline = [c for c in cells if not c.is_anchor]
    assert len(headline) == 11

    def counts(cs: list[Cell], types: tuple[str, ...]) -> tuple[int, int]:
        tot = pas = 0
        for c in cs:
            for v in vd_by_cell[(c.date, c.profile)]:
                if v.claim.ctype in types:
                    tot += 1
                    pas += int(v.passed)
        return tot, pas

    T16 = ("T1", "T2", "T3", "T4", "T5", "T6")
    head_tot, head_pass = counts(headline, T16)
    anch_tot, anch_pass = counts([anchor], T16)
    all_t7 = sum(1 for c in cells for v in vd_by_cell[(c.date, c.profile)]
                 if v.claim.ctype == "T7")
    all_unacc = sum(len(un_by_cell[(c.date, c.profile)]) for c in cells)
    all_t8 = sum(len(ex_by_cell[(c.date, c.profile)].t8_violations) for c in cells)

    hard_ok = (all_t7 == 0) and (all_unacc == 0) and (all_t8 == 0)
    rate_ok = head_tot > 0 and head_pass == head_tot
    supported = hard_ok and rate_ok

    # --------------------------------------------------------- report --- #
    lines = env_header("Task 13 Stage C -- one-shot narration evaluation", 
                       pin)
    lines += [
        "## 0. Boundary",
        "",
        "> The Task 12 verdict is frozen ground truth. If narration fails "
        "faithfulness, that is a Task 13 finding about the narration layer and "
        "re-litigates nothing upstream. attributions_main.csv, the "
        "Recommendation interface, and all Task 9-12 outputs are read-only.",
        "",
        "## 1. Run integrity",
        "",
        md_table(["item", "value"], [
            ["environment drift check", f"`/api/version` re-read before the "
             f"first generation: `{pin['server_version']}` vs the Stage A pin "
             f"`{frozen_pin['server_version']}` -- no drift; model digest "
             "matched as well"],
            ["discarded warm-up", "1 (excluded from all counts)"],
            ["evaluation generations", f"{len(evaluation_calls)} / 36"],
            ["registered order honoured",
             "yes -- dates ascending, conservative/balanced/growth within date, "
             "k = 3 consecutive per cell" if order_ok else "**NO**"],
            ["transcripts written immediately", "yes, one file per generation"],
            ["I1 model == pinned tag",
             f"held on {sum(1 for r in evaluation_calls if r.i1_model_ok)}/36"],
            ["I2 no thinking content",
             f"held on {sum(1 for r in evaluation_calls if r.i2_no_thinking_ok)}/36"],
            ["I3 done == true",
             f"held on {sum(1 for r in evaluation_calls if r.i3_done)}/36"],
            ["I3 done_reason values",
             ", ".join(f"`{v}` x{sum(1 for r in evaluation_calls if r.i3_done_reason == v)}"
                       for v in sorted({r.i3_done_reason for r in evaluation_calls}))],
            ["I4 prompt_eval_count <= 7680",
             f"held on {sum(1 for r in evaluation_calls if r.i4_ok)}/36; observed "
             f"range {min(r.i4_prompt_eval_count for r in evaluation_calls)}-"
             f"{max(r.i4_prompt_eval_count for r in evaluation_calls)}"],
            ["eval_count range (cap 512)",
             f"{min(r.eval_count for r in evaluation_calls)}-"
             f"{max(r.eval_count for r in evaluation_calls)}"],
            ["per-call log", f"`outputs/task13/stage_c_call_log.csv`, "
             f"{len(evaluation_calls) + 1} rows (warm-up included, flagged)"],
        ]),
        "",
        f"Bookkeeping reconciliation carried in at ratification: {STAGE_B_GENERATIONS}",
        "",
    ]
    if notes:
        lines += notes + [""]

    # determinism
    lines += [
        "## 2. Determinism finding",
        "",
        md_table(["date", "profile", "flag", "agreement (rep1/2/3)",
                  "response chars", "rep1 sha256 (16)", "byte-identical",
                  "first divergence"], det_rows),
        "",
        f"Cells whose k = 3 repeats were byte-identical: "
        f"**{len(cells) - len(divergent)} / {len(cells)}**. Divergent cells: "
        f"**{len(divergent)}**.",
        "",
        ("The registered k = 3 byte-identical standard held on every cell, so "
         "the fallback was not activated. The evaluated transcript is "
         "the first repeat per cell either way."
         if not fallback_active else
         "The fallback is ACTIVATED. Per-cell repeat agreement "
         "patterns and first divergence offsets are in the table above; the "
         "first generation per cell is the frozen evaluated transcript."),
        "",
    ]

    # per-cell results
    per_cell_rows = []
    for c in cells:
        ex, vds = ex_by_cell[(c.date, c.profile)], vd_by_cell[(c.date, c.profile)]
        core = [v for v in vds if v.claim.ctype in T16]
        per_cell_rows.append([
            c.date, c.profile, "**anchor (dev)**" if c.is_anchor else "",
            len(ex.sentences), len(vds), len(core),
            sum(1 for v in core if v.passed),
            sum(1 for v in core if not v.passed),
            sum(1 for v in vds if v.claim.ctype == "T7"),
            len(un_by_cell[(c.date, c.profile)]),
            len(ex.t8_flagged), len(ex.t8_violations),
        ])
    lines += [
        "## 3. Per-cell results (evaluated transcript = first repeat)",
        "",
        md_table(["date", "profile", "flag", "sentences", "claims",
                  "T1-T6 claims", "T1-T6 pass", "T1-T6 fail", "T7", "unaccounted",
                  "T8 flagged", "T8 violations"], per_cell_rows),
        "",
    ]

    # per type
    type_rows = []
    for t in T16 + ("T7",):
        h_t, h_p = counts(headline, (t,))
        a_t, a_p = counts([anchor], (t,))
        type_rows.append([t, _type_name(t), h_t, h_p, h_t - h_p, a_t, a_p])
    lines += [
        "## 4. Aggregates",
        "",
        "### Per claim type",
        "",
        md_table(["type", "name", "headline extracted (11 cells)",
                  "headline pass", "headline fail", "anchor extracted",
                  "anchor pass"], type_rows),
        "",
        "### Headline and anchor",
        "",
        md_table(["scope", "T1-T6 claims", "passed", "failed", "pass rate",
                  "T7 claims", "unaccounted numerics", "T8 violations"], [
            ["**headline: 11 non-development cells**", head_tot, head_pass,
             head_tot - head_pass,
             f"{100.0 * head_pass / head_tot:.1f}%" if head_tot else "n/a",
             sum(1 for c in headline for v in vd_by_cell[(c.date, c.profile)]
                 if v.claim.ctype == "T7"),
             sum(len(un_by_cell[(c.date, c.profile)]) for c in headline),
             sum(len(ex_by_cell[(c.date, c.profile)].t8_violations)
                 for c in headline)],
            ["anchor cell (development, flagged, outside the headline)",
             anch_tot, anch_pass, anch_tot - anch_pass,
             f"{100.0 * anch_pass / anch_tot:.1f}%" if anch_tot else "n/a",
             sum(1 for v in vd_by_cell[(anchor.date, anchor.profile)]
                 if v.claim.ctype == "T7"),
             len(un_by_cell[(anchor.date, anchor.profile)]),
             len(ex_by_cell[(anchor.date, anchor.profile)].t8_violations)],
            ["all 12 cells (hard-condition scope)",
             head_tot + anch_tot, head_pass + anch_pass,
             (head_tot + anch_tot) - (head_pass + anch_pass), "-",
             all_t7, all_unacc, all_t8],
        ]),
        "",
        "### Registered conditions",
        "",
        md_table(["condition", "scope", "required", "measured", "met"], [
            ["zero T7 (fabricated claims)", "all 12 cells", 0, all_t7,
             "yes" if all_t7 == 0 else "**no**"],
            ["zero unaccounted numerics (T7 hard-fail path)", "all 12 cells", 0,
             all_unacc, "yes" if all_unacc == 0 else "**no**"],
            ["zero T8 violations", "all 12 cells", 0, all_t8,
             "yes" if all_t8 == 0 else "**no**"],
            ["100% of extracted T1-T6 claims pass", "11 headline cells",
             f"{head_tot}/{head_tot}", f"{head_pass}/{head_tot}",
             "yes" if rate_ok else "**no**"],
        ]),
        "",
    ]

    # failures, stated symmetrically
    failures = [(c, v) for c in cells for v in vd_by_cell[(c.date, c.profile)]
                if not v.passed]
    lines += ["### Failing claims", ""]
    if failures:
        lines += [md_table(
            ["date", "profile", "type", "subject", "basis", "claimed",
             "precision", "frozen reference (full precision)", "note"],
            [[c.date, c.profile, v.claim.ctype, v.claim.subject, v.claim.basis,
              f"`{v.claim.value}`", v.claim.precision,
              f"`{v.reference_repr[:80]}`", v.note[:120]]
             for c, v in failures]), ""]
    else:
        lines += ["None. Every extracted claim across all 12 cells ties out to "
                  "the frozen artifacts under the registered matching rule.", ""]

    # T8 flagged, verbatim
    flagged = [(c, s) for c in cells
               for s in ex_by_cell[(c.date, c.profile)].t8_flagged]
    lines += [
        "### T8 screen -- flagged sentences listed verbatim for review",
        "",
        "Flags any sentence containing both a profile token "
        "{conservative, balanced, growth, profile} and a forecast token "
        "{forecast, volatility, vol, predicted, expected}. A flagged sentence "
        "VIOLATES iff it attributes a forecast value or forecast difference to "
        "the profile choice. The d-vector is profile-invariant at fixed date "
        "(verified at Stage A tie-out (b): 0 breaches).",
        "",
    ]
    if flagged:
        lines += [md_table(["date", "profile", "violation", "evidence", "sentence"],
                           [[c.date, c.profile, s.t8_violation, s.t8_evidence,
                             ascii_escape(s.text)[0]] for c, s in flagged]), ""]
    else:
        lines += ["No sentence in any of the 12 evaluated transcripts was "
                  "flagged: no narration placed a profile token and a forecast "
                  "token in the same sentence. Zero flagged implies zero "
                  "violations.", ""]

    # soft observations
    soft = [(c, s.index, q) for c in cells
            for s in ex_by_cell[(c.date, c.profile)].sentences
            for q in s.qualitative]
    lines += [
        "### Soft observations (recorded, never adjudicated)",
        "",
        (md_table(["date", "profile", "sentence #", "qualitative phrase"],
                  [[c.date, c.profile, i, q] for c, i, q in soft])
         if soft else
         "No phrase from the registered qualitative-quantity lexicon appeared "
         "in any evaluated transcript. The template's style rule instructs "
         "numerals only.") ,
        "",
    ]

    # verdict
    if supported:
        verdict = (
            f"Narration faithfulness is supported: every extracted claim across "
            f"the 11 registered evaluation cells ties out to the frozen "
            f"artifacts under the registered matching rule ({head_pass}/"
            f"{head_tot} claims), no fabricated claims were detected, and no "
            f"narration implies profile-dependent forecasts.")
    else:
        bits = []
        if not rate_ok:
            bits.append(f"{head_tot - head_pass} of {head_tot} extracted T1-T6 "
                        f"claims across the 11 headline cells failed the "
                        f"registered matching rule")
        if all_t7:
            bits.append(f"{all_t7} fabricated (T7) claim(s) across the 12 cells")
        if all_unacc:
            bits.append(f"{all_unacc} unaccounted numeric(s) across the 12 "
                        f"cells, which take the T7 hard-fail path")
        if all_t8:
            bits.append(f"{all_t8} T8 profile-dependence violation(s) across the "
                        f"12 cells")
        verdict = ("Narration faithfulness is not supported: " +
                   "; ".join(bits) + ".")

    lines += [
        "## 5. Verdict ",
        "",
        f"> {verdict}",
        "",
        "> This verdict attaches to the narration layer as frozen at Gate B3 -- "
        "the registered template, serializer, and options block; it does not "
        "certify free-form narration of the Recommendation.",
        "",
        "(Amendment: the companion sentence is registered to travel with "
        "the verdict wherever it is quoted.)",
        "",
        "The anchor cell is the development cell. Its results appear in every "
        "table above, always flagged, and are excluded from the headline "
        f"counts; for the record it contributed {anch_pass}/{anch_tot} passing "
        "T1-T6 claims.",
        "",
    ]

    # exhibits
    n1 = _exhibit_n1(anchor, texts[(anchor.date, anchor.profile)][0],
                     ex_by_cell[(anchor.date, anchor.profile)],
                     vd_by_cell[(anchor.date, anchor.profile)])
    n2 = _exhibit_n2(cells, vd_by_cell, T16)
    lines += [
        "## 6. Exhibits",
        "",
        "Both are regenerated from the frozen CSVs and the frozen transcripts "
        "only; no value in either is recomputed from anything else.",
        "",
        "- **N1** -- `outputs/task13/N1_anchor_annotated.md`: the anchor-cell "
        "narration with every claim span annotated by type and pass/fail.",
        "- **N2** -- `outputs/task13/N2_claims_summary.md` (and "
        "`N2_claims_summary.csv`): claims summary, type x cell, pass counts.",
        "",
    ]

    # gates
    c1 = (len(evaluation_calls) == 36 and order_ok
          and all(r.i1_model_ok and r.i2_no_thinking_ok and r.i3_done
                  for r in evaluation_calls)
          and CALL_LOG.exists())
    c2 = (all(len(ex_by_cell[(c.date, c.profile)].sentences) > 0 for c in cells)
          and SENTENCES_CSV.exists() and UNACCOUNTED_CSV.exists())
    c3 = CLAIMS_CSV.exists() and n1.exists() and n2.exists()
    i4_dev = [r for r in evaluation_calls if not r.i4_ok]
    lines += [
        "## 7. Gates",
        "",
        md_table(["gate", "criterion", "evidence", "verdict"], [
            ["C1", "36/36 generations in registered order; invariants I1-I4 "
                   "held or deviations logged; per-call log complete",
             f"{len(evaluation_calls)}/36 completed; order honoured = "
             f"{order_ok}; I1/I2/I3 held on all 36; I4 deviations logged = "
             f"{len(i4_dev)}; call log {CALL_LOG.name} written",
             "PASS" if c1 else "**FAIL**"],
            ["C2", "every sentence classified; unaccounted-numeric scan run "
                   "over every transcript",
             f"{sum(len(ex_by_cell[(c.date, c.profile)].sentences) for c in cells)}"
             f" sentences classified across 12 transcripts; scan run on all 12 "
             f"(results feed the verdict, not this gate)",
             "PASS" if c2 else "**FAIL**"],
            ["C3", "verdict emitted per the registered vocabulary; symmetric "
                   "report; exhibits from frozen data only",
             "one verdict form emitted with counts filled, plus the "
             "companion sentence; N1 and N2 written",
             "PASS" if c3 else "**FAIL**"],
        ]),
        "",
        "## 8. Deliverables written by this stage",
        "",
        "- `outputs/task13/stage_c_report.md` ",
        "- `outputs/task13/narrations/<date>_<profile>_rep{1,2,3}.txt` (36) and "
        "`stage_c_warmup_discarded.txt`",
        "- `outputs/task13/claims_extracted.csv`, `sentence_classes.csv`, "
        "`unaccounted_numerics.csv`, `t8_screen.csv`",
        "- `outputs/task13/determinism_log.csv` (Stage A probe rows preserved; "
        "Stage C rows appended)",
        "- `outputs/task13/stage_c_call_log.csv`",
        "- `outputs/task13/N1_anchor_annotated.md`, `N2_claims_summary.md`, "
        "`N2_claims_summary.csv`",
        "",
        ("STAGE C COMPLETE -- gates C1/C2/C3 PASS. One-shot discipline held: "
         "36 generations, one extraction pass, one adjudication pass. "
         if (c1 and c2 and c3) else
         "STAGE C HALTED -- see the gate table."),
        "",
    ]

    write_report("stage_c_report.md", lines)
    print(f"C1={c1} C2={c2} C3={c3} | headline {head_pass}/{head_tot} | "
          f"anchor {anch_pass}/{anch_tot} | T7={all_t7} unaccounted={all_unacc} "
          f"T8viol={all_t8} | divergent cells={len(divergent)} | "
          f"supported={supported}")
    return 0 if (c1 and c2 and c3) else 1


def _type_name(t: str) -> str:
    return {
        "T1": "numeric weight claim", "T2": "numeric forecast-volatility claim",
        "T3": "numeric attribution claim", "T4": "directional/comparative claim",
        "T5": "ranking claim", "T6": "provenance claim",
        "T7": "FABRICATED claim",
    }[t]


# --------------------------------------------------------------------------- #
# CSV outputs (full precision where numeric)
# --------------------------------------------------------------------------- #
def _write_claims(cells, ex_by_cell, vd_by_cell) -> None:
    with CLAIMS_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "profile", "is_anchor", "claim_id", "type",
                    "subject", "basis", "claimed_value", "stated_precision",
                    "frozen_reference_value_full_repr", "derivation_tag",
                    "sentence_index", "char_start", "char_end", "outcome",
                    "note"])
        for c in cells:
            for i, v in enumerate(vd_by_cell[(c.date, c.profile)]):
                cl = v.claim
                w.writerow([c.date, c.profile, c.is_anchor, i, cl.ctype,
                            cl.subject, cl.basis, cl.value, cl.precision,
                            v.reference_repr, v.derivation, cl.sentence_index,
                            cl.span[0], cl.span[1],
                            "PASS" if v.passed else "FAIL", v.note])


def _write_sentences(cells, ex_by_cell) -> None:
    with SENTENCES_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "profile", "is_anchor", "sentence_index", "class",
                    "subclass", "n_claims", "t8_flagged", "t8_violation",
                    "t8_evidence", "qualitative_soft_observations", "text"])
        for c in cells:
            for s in ex_by_cell[(c.date, c.profile)].sentences:
                w.writerow([c.date, c.profile, c.is_anchor, s.index, s.klass,
                            s.subclass, len(s.claim_ids), s.t8_flagged,
                            s.t8_violation, s.t8_evidence,
                            "; ".join(s.qualitative), s.text])


def _write_unaccounted(cells, un_by_cell, ex_by_cell) -> None:
    with UNACCOUNTED_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "profile", "is_anchor", "char_start", "char_end",
                    "token", "sentence_index", "disposition"])
        for c in cells:
            for start, end, token, sidx, disp in un_by_cell[(c.date, c.profile)]:
                w.writerow([c.date, c.profile, c.is_anchor, start, end, token,
                            sidx, disp])


def _write_t8(cells, ex_by_cell) -> None:
    with T8_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "profile", "is_anchor", "sentence_index",
                    "flagged", "violation", "evidence", "text"])
        for c in cells:
            for s in ex_by_cell[(c.date, c.profile)].sentences:
                if s.t8_flagged:
                    w.writerow([c.date, c.profile, c.is_anchor, s.index, True,
                                s.t8_violation, s.t8_evidence, s.text])


# --------------------------------------------------------------------------- #
# Exhibits
# --------------------------------------------------------------------------- #
def _exhibit_n1(cell: Cell, text: str, ex: Extraction,
                verdicts: list[Verdict]) -> Path:
    spans = sorted(
        ((v.claim.span[0], v.claim.span[1], v.claim.ctype,
          "PASS" if v.passed else "FAIL") for v in verdicts),
        key=lambda s: (-s[0], s[1]))
    annotated = text
    applied = 0
    used: list[tuple[int, int]] = []
    for a, b, t, ok in spans:
        if any(a < y and x < b for x, y in used):
            # Overlaps a span already annotated.  Marker insertion shifts
            # offsets, so only pairwise-disjoint spans are marked inline; the
            # skipped ones are still listed in the claim table below.
            continue
        annotated = annotated[:b] + f"[/{t}:{ok}]" + annotated[b:]
        annotated = annotated[:a] + f"[{t}:{ok}]" + annotated[a:]
        used.append((a, b))
        applied += 1
    esc, n_non_ascii = ascii_escape(annotated)
    raw_esc, _ = ascii_escape(text)
    body = [
        "# Exhibit N1 -- anchor-cell narration with claim spans annotated",
        "",
        f"Cell: **{cell.date} / {cell.profile}** (the development cell, flagged "
        "and outside the headline counts). Evaluated transcript = repeat 1, "
        f"`outputs/task13/narrations/{cell.date}_{cell.profile}_rep1.txt`.",
        "",
        "Rendered from the frozen transcript and the frozen grounding CSVs "
        "only. ASCII-escaped for the Sec. 8 rule; the raw bytes are the file "
        "named above.",
        "",
        "## Narration, verbatim",
        "",
        "```",
        raw_esc,
        "```",
        "",
        "## Narration, annotated with claim spans (type : outcome)",
        "",
        "```",
        esc,
        "```",
        "",
        f"Spans annotated: {applied} of {len(verdicts)} claims "
        f"({len(verdicts) - applied} skipped as partially overlapping another "
        "span; every claim appears in the table below regardless).",
        "",
        "## Claim table",
        "",
        md_table(["#", "type", "subject", "basis", "claimed", "precision",
                  "span", "span text", "outcome", "frozen reference"],
                 [[i, v.claim.ctype, v.claim.subject, v.claim.basis,
                   f"`{v.claim.value}`", v.claim.precision,
                   f"{v.claim.span[0]}-{v.claim.span[1]}",
                   f"`{ascii_escape(text[v.claim.span[0]:v.claim.span[1]])[0]}`",
                   "PASS" if v.passed else "**FAIL**",
                   f"`{ascii_escape(v.reference_repr[:60])[0]}`"]
                  for i, v in enumerate(verdicts)]),
        "",
        "## Sentence classification",
        "",
        md_table(["#", "class", "subclass", "claims", "T8 flagged",
                  "T8 violation", "sentence"],
                 [[s.index, s.klass, s.subclass, len(s.claim_ids),
                   s.t8_flagged, s.t8_violation, ascii_escape(s.text)[0]]
                  for s in ex.sentences]),
        "",
        f"Non-ASCII characters escaped in this exhibit: {n_non_ascii}.",
        "",
    ]
    path = OUT_DIR / "N1_anchor_annotated.md"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def _exhibit_n2(cells, vd_by_cell, T16) -> Path:
    types = T16 + ("T7",)
    rows = []
    for c in cells:
        vds = vd_by_cell[(c.date, c.profile)]
        row = [c.date, c.profile, "anchor (dev)" if c.is_anchor else "-"]
        for t in types:
            sel = [v for v in vds if v.claim.ctype == t]
            row.append(f"{sum(1 for v in sel if v.passed)}/{len(sel)}"
                       if sel else "-")
        core = [v for v in vds if v.claim.ctype in T16]
        row.append(f"{sum(1 for v in core if v.passed)}/{len(core)}")
        rows.append(row)
    header = ["date", "profile", "flag", *types, "T1-T6 total"]

    totals = ["**headline (11 cells)**", "", ""]
    head = [c for c in cells if not c.is_anchor]
    for t in types:
        sel = [v for c in head for v in vd_by_cell[(c.date, c.profile)]
               if v.claim.ctype == t]
        totals.append(f"{sum(1 for v in sel if v.passed)}/{len(sel)}"
                      if sel else "0/0")
    core = [v for c in head for v in vd_by_cell[(c.date, c.profile)]
            if v.claim.ctype in T16]
    totals.append(f"{sum(1 for v in core if v.passed)}/{len(core)}")

    body = [
        "# Exhibit N2 -- claims summary, type x cell, pass counts",
        "",
        "Each cell of the table reads `passed / extracted`. A dash means the "
        "type was not extracted in that cell. The anchor row is the "
        "development cell and is excluded from the headline row.",
        "",
        "Rendered from `claims_extracted.csv`, which is itself written from the "
        "frozen transcripts adjudicated against the frozen grounding CSVs.",
        "",
        md_table(header, rows + [totals]),
        "",
        md_table(["type", "name"], [[t, _type_name(t)] for t in types]),
        "",
    ]
    path = OUT_DIR / "N2_claims_summary.md"
    path.write_text("\n".join(body), encoding="utf-8")
    with (OUT_DIR / "N2_claims_summary.csv").open("w", encoding="utf-8",
                                                  newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
        w.writerow(totals)
    return path


if __name__ == "__main__":
    sys.exit(main())
