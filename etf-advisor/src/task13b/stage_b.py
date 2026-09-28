"""Task 13b Stage B -- one-shot readability computation.

Execute once:

  1. Compute FRE and FKGL per cell on the verbatim file contents; write
     outputs/task13b/readability_results.csv.
  2. Report the full 12-row table (anchor cell flagged), the 11-cell means, and
     the R1/R2/R3 outcomes.
  3. Emit the verdict in the registered vocabulary and the filled Sec. 3 chapter
     sentence.
  4. STOP.
"""

from __future__ import annotations

import csv
import sys
from statistics import fmean

import textstat

from . import (
    CELLS,
    OUT_DIR,
    R1_MEAN_FRE_MIN,
    R2_MEAN_FKGL_MAX,
    R3_CELL_FKGL_CEIL,
    R3_CELL_FRE_FLOOR,
    RESULTS_CSV,
    ROUNDING_DP,
    env_header,
    md_table,
    write_report,
)

REPORT_NAME = "stage_b_report.md"

# CSV columns registered in brief Sec. 3, in order.
CSV_COLUMNS = ["cell_id", "profile", "date", "dev_flag",
               "n_words", "n_sentences", "fre", "fkgl"]


# --------------------------------------------------------------------------- #
# Pure computation (unit-testable on synthetic text; see scratchpad self-test)
# --------------------------------------------------------------------------- #
def compute_cell(text: str) -> dict[str, float]:
    """Full-precision readability quantities for one verbatim transcript.

    ``text`` is passed to textstat exactly as read from disk -- no stripping, no
    normalisation.  n_words / n_sentences are textstat's own tokenisation, 
    i.e. the very counts that feed its FRE/FKGL.
    """
    return {
        "n_words": textstat.lexicon_count(text),
        "n_sentences": textstat.sentence_count(text),
        "fre": textstat.flesch_reading_ease(text),      # full precision
        "fkgl": textstat.flesch_kincaid_grade(text),    # full precision
    }


def _eng_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def naming_clause(r1: bool, r2: bool, r3: bool) -> str:
    """Names which of R1/R2/R3 hold"""
    held = [n for n, v in (("R1", r1), ("R2", r2), ("R3", r3)) if v]
    failed = [n for n, v in (("R1", r1), ("R2", r2), ("R3", r3)) if not v]
    if not failed:
        return ""
    if not held:
        return "none of R1, R2 or R3 is met"
    held_v = "is" if len(held) == 1 else "are"
    failed_v = "is" if len(failed) == 1 else "are"
    return (f"{_eng_join(held)} {held_v} met, "
            f"{_eng_join(failed)} {failed_v} not")


def verdict_word(r1: bool, r2: bool, r3: bool) -> str:
    if r1 and r2 and r3:
        return "supported"
    if not (r1 or r2 or r3):
        return "not supported"
    return "partially supported"


def evaluate(eval_rows: list[dict]) -> dict:
    """R1/R2/R3 on FULL-PRECISION values over the 11 evaluation cells.

    Rounding is a display convention: means are computed at full
    precision and the registered thresholds are evaluated on the full-precision
    quantities, exactly as means are.  The 1-dp values are what gets reported.
    """
    fres = [r["fre"] for r in eval_rows]
    fkgls = [r["fkgl"] for r in eval_rows]
    mean_fre = fmean(fres)
    mean_fkgl = fmean(fkgls)
    min_fre = min(fres)
    max_fkgl = max(fkgls)
    r1 = mean_fre >= R1_MEAN_FRE_MIN
    r2 = mean_fkgl <= R2_MEAN_FKGL_MAX
    r3 = (min_fre >= R3_CELL_FRE_FLOOR) and (max_fkgl <= R3_CELL_FKGL_CEIL)
    word = verdict_word(r1, r2, r3)
    clause = naming_clause(r1, r2, r3)
    return {
        "mean_fre": mean_fre, "mean_fkgl": mean_fkgl,
        "min_fre": min_fre, "max_fkgl": max_fkgl,
        "mean_words": fmean([r["n_words"] for r in eval_rows]),
        "mean_sentences": fmean([r["n_sentences"] for r in eval_rows]),
        "R1": r1, "R2": r2, "R3": r3,
        "verdict": word, "naming_clause": clause,
    }


def chapter_sentence(ev: dict) -> str:
    """The filled chapter sentence (the registered verbatim unit)."""
    x = f"{round(ev['mean_fre'], ROUNDING_DP):.1f}"
    y = f"{round(ev['mean_fkgl'], ROUNDING_DP):.1f}"
    base = (
        f"Readability is {ev['verdict']}: mean Flesch Reading Ease {x} and mean "
        f"Flesch–Kincaid grade {y} across the 11 evaluation cells, against "
        f"targets registered before computation (FRE ≥ 50, FKGL ≤ 12, "
        f"with per-cell floors)"
    )
    if ev["naming_clause"]:
        return base + "; " + ev["naming_clause"] + "."
    return base + "."


# --------------------------------------------------------------------------- #
# One-shot driver
# --------------------------------------------------------------------------- #
def main() -> int:
    # The registered chapter sentence carries en-dash and >=/<= glyphs; make the
    # console print safe on a terminal (the .md/.csv deliverables
    # are always written as UTF-8 regardless).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    report_path = OUT_DIR / REPORT_NAME
    if RESULTS_CSV.exists() or report_path.exists():
        raise RuntimeError(
            "one-shot guard: Stage B outputs already exist "
            f"({RESULTS_CSV.name} / {REPORT_NAME}); Stage B computes exactly "
            "once. A re-run requires a logged defect and a planning-level "
            "ruling, never a look at results -- STOP"
        )

    # ----- compute once, per cell, on verbatim contents -----
    cells_out: list[dict] = []
    for c in CELLS:
        text = c.path.read_text(encoding="utf-8")   # verbatim; no CR/LF (Gate A2)
        m = compute_cell(text)
        cells_out.append({
            "cell": c,
            "n_words": m["n_words"],
            "n_sentences": m["n_sentences"],
            "fre": m["fre"],
            "fkgl": m["fkgl"],
        })

    eval_rows = [r for r in cells_out if not r["cell"].dev_flag]
    ev = evaluate(eval_rows)
    sentence = chapter_sentence(ev)

    # ----- write the exhibit CSV (12 rows + means row over 11 eval cells) -----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in cells_out:
            c = r["cell"]
            w.writerow([
                c.cell_id, c.profile, c.date,
                "anchor_dev" if c.dev_flag else "",
                r["n_words"], r["n_sentences"],
                f"{round(r['fre'], ROUNDING_DP):.1f}",
                f"{round(r['fkgl'], ROUNDING_DP):.1f}",
            ])
        # means row over the 11 evaluation cells (anchor excluded)
        w.writerow([
            "mean_11_evaluation_cells", "", "", "anchor_excluded",
            f"{round(ev['mean_words'], ROUNDING_DP):.1f}",
            f"{round(ev['mean_sentences'], ROUNDING_DP):.1f}",
            f"{round(ev['mean_fre'], ROUNDING_DP):.1f}",
            f"{round(ev['mean_fkgl'], ROUNDING_DP):.1f}",
        ])

    # ----- report -----
    table_rows = []
    for r in cells_out:
        c = r["cell"]
        table_rows.append([
            c.cell_id, c.profile, c.date,
            "**anchor (dev)**" if c.dev_flag else "",
            r["n_words"], r["n_sentences"],
            f"{round(r['fre'], ROUNDING_DP):.1f}",
            f"{round(r['fkgl'], ROUNDING_DP):.1f}",
        ])
    table_rows.append([
        "**mean (11 eval cells)**", "", "", "anchor excluded",
        f"{round(ev['mean_words'], ROUNDING_DP):.1f}",
        f"{round(ev['mean_sentences'], ROUNDING_DP):.1f}",
        f"**{round(ev['mean_fre'], ROUNDING_DP):.1f}**",
        f"**{round(ev['mean_fkgl'], ROUNDING_DP):.1f}**",
    ])

    r_rows = [
        ["R1 (mean ease)", "mean FRE over 11 eval cells >= 50",
         f"{ev['mean_fre']:.4f} (full); {round(ev['mean_fre'], 1):.1f} (1dp)",
         ">= 50", "yes" if ev["R1"] else "NO"],
        ["R2 (mean grade)", "mean FKGL over 11 eval cells <= 12",
         f"{ev['mean_fkgl']:.4f} (full); {round(ev['mean_fkgl'], 1):.1f} (1dp)",
         "<= 12", "yes" if ev["R2"] else "NO"],
        ["R3 (per-cell floors)", "no eval cell FRE < 30 and none FKGL > 16",
         f"min FRE {ev['min_fre']:.4f}; max FKGL {ev['max_fkgl']:.4f} (full)",
         "FRE >= 30, FKGL <= 16", "yes" if ev["R3"] else "NO"],
    ]

    verdict_line = f"Readability is {ev['verdict']}"
    if ev["naming_clause"]:
        verdict_line += f" ({ev['naming_clause']})"
    verdict_line += "."

    lines: list[str] = []
    lines += env_header("Task 13b Stage B -- one-shot readability computation")
    lines += [
        "Compute FRE and FKGL once over the twelve frozen "
        "`_rep1` narration transcripts, emit the exhibit, and report the verdict "
        "against the targets registered before any computation existed.  This is "
        "the readability half of revised RQ3; the "
        "factual-consistency half is carried by the Task 13 verdict.",
        "",
        "Extraction rule: each metric is computed on the transcript "
        "file contents verbatim -- read as UTF-8 text, passed to textstat with no "
        "stripping and no normalisation.  Gate A2 established every file carries "
        "no CR/LF, so this text read equals the raw bytes; n_words / n_sentences "
        "are textstat's own tokenisation (the inputs to its FRE/FKGL).",
        "",
        "Rounding: FRE and FKGL are reported at one decimal place; "
        "the 11-cell means are computed at full precision and rounded last (so a "
        "mean need not equal the mean of the rounded per-cell values shown).  The "
        "registered R1/R2/R3 thresholds are evaluated on the full-precision "
        "quantities, consistent with the same rounding rule.",
        "",
        "## Results -- all 12 cells (anchor development cell flagged and excluded "
        "from the means)",
        "",
        md_table(["cell_id", "profile", "date", "flag", "n_words",
                  "n_sentences", "FRE", "FKGL"], table_rows),
        "",
        f"Exhibit written: `{RESULTS_CSV.as_posix()}` "
        "(12 cell rows + one means row over the 11 evaluation cells).",
        "",
        "## Registered conditions R1/R2/R3 ",
        "",
        md_table(["condition", "registered rule", "measured", "target", "met"],
                 r_rows),
        "",
        "## Verdict ",
        "",
        f"**{verdict_line}**",
        "",
        "## Filled chapter sentence ",
        "",
        "> " + sentence,
        "",
        "## Stop line",
        "",
        "STAGE B COMPLETE ",
    ]

    write_report(REPORT_NAME, lines)

    print(f"Verdict: {verdict_line}")
    print(f"mean FRE (11 eval) = {ev['mean_fre']:.6f} -> {round(ev['mean_fre'],1):.1f}")
    print(f"mean FKGL (11 eval) = {ev['mean_fkgl']:.6f} -> {round(ev['mean_fkgl'],1):.1f}")
    print(f"R1={ev['R1']} R2={ev['R2']} R3={ev['R3']}")
    print(f"min FRE={ev['min_fre']:.4f}  max FKGL={ev['max_fkgl']:.4f}")
    print(f"CSV: {RESULTS_CSV.as_posix()}")
    print(f"Report: {(OUT_DIR / REPORT_NAME).as_posix()}")
    print("\nChapter sentence:\n" + sentence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
