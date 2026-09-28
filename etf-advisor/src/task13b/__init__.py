"""Task 13b -- automated readability of the frozen Task 13 narration transcripts.

Pre-registered mini-task.  Computes two standard readability indices -- 
Flesch Reading Ease (FRE) and Flesch-Kincaid Grade Level
(FKGL), as implemented by ``textstat`` -- over the twelve frozen ``_rep1``
narration transcripts of the Task 13 evaluation grid, once, and emits a verdict
in the registered vocabulary.

Nothing here contacts a model, regenerates a transcript, or edits a frozen file.
This module holds ONLY the registered constants and the report plumbing shared
by the two stage drivers; it computes no evidentiary number.

Extraction rule: the metric is computed on the transcript file
contents verbatim -- the narration text exactly as frozen and as displayed to
the user, no stripping, no normalisation.

Reproducibility conventions inherited from src/task13: ASCII-safe report output,
overwrite guards on failure/halt records, full-precision checksums, staged hard
gates.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Sequence

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
TASK13_OUT = REPO_ROOT / "outputs" / "task13"
NARRATION_DIR = TASK13_OUT / "narrations"
TASK13_STAGE_C_REPORT = TASK13_OUT / "stage_c_report.md"
OUT_DIR = REPO_ROOT / "outputs" / "task13b"
RESULTS_CSV = OUT_DIR / "readability_results.csv"


# --------------------------------------------------------------------------- #
# Registered 12-cell map.  The 12 registered cells = 11 evaluation
# cells + the flagged anchor development cell (2024-12-02 / balanced).  The anchor
# cell is computed and reported FLAGGED, and is excluded from every headline
# aggregate.
#
# ``exp_sha16`` and ``resp_chars`` are transcribed verbatim from the Task 13
# Stage C report (outputs/task13/stage_c_report.md, Section 2 determinism
# table): the "rep1 sha256 (16)" and "response chars" columns.  These are the
# frozen-integrity anchors Gate A2 checks the on-disk files against.
# --------------------------------------------------------------------------- #
class Cell(NamedTuple):
    date: str
    profile: str
    dev_flag: bool          # True == flagged anchor development cell
    exp_sha16: str          # Task 13 Stage C rep1 sha256, first 16 hex
    resp_chars: int         # Task 13 Stage C "response chars" for rep1

    @property
    def cell_id(self) -> str:
        return f"{self.date}_{self.profile}"

    @property
    def path(self) -> Path:
        return NARRATION_DIR / f"{self.date}_{self.profile}_rep1.txt"


CELLS: tuple[Cell, ...] = (
    Cell("2012-02-01", "conservative", False, "1175158ad6222851", 663),
    Cell("2012-02-01", "balanced",     False, "6519acedf8cc3ab9", 660),
    Cell("2012-02-01", "growth",       False, "2f612a038425ca08", 658),
    Cell("2020-03-02", "conservative", False, "9e95240b1b187a51", 663),
    Cell("2020-03-02", "balanced",     False, "718897b4e7e60a5b", 659),
    Cell("2020-03-02", "growth",       False, "a02283a0404ea98d", 656),
    Cell("2022-06-01", "conservative", False, "99f859cf0e6bc4aa", 663),
    Cell("2022-06-01", "balanced",     False, "546ad58b0b9be787", 660),
    Cell("2022-06-01", "growth",       False, "db55cab141c0f778", 659),
    Cell("2024-12-02", "conservative", False, "9d0fb181788c6292", 662),
    Cell("2024-12-02", "balanced",     True,  "5b38652b480760c1", 660),  # anchor (dev)
    Cell("2024-12-02", "growth",       False, "84ac9b2b4a88ec5d", 658),
)
ANCHOR_CELL: tuple[str, str] = ("2024-12-02", "balanced")   # development cell
N_EVALUATION_CELLS = 11
N_TOTAL_CELLS = 12


# --------------------------------------------------------------------------- #
# Registered thresholds and verdict rule.  Registered BEFORE any
# computation exists; justified a priority by the audience (novice retail investors
# reading necessarily technical financial prose) and NOT revisited after results.
# --------------------------------------------------------------------------- #
R1_MEAN_FRE_MIN = 50.0      # mean FRE over the 11 evaluation cells >= 50
R2_MEAN_FKGL_MAX = 12.0     # mean FKGL over the 11 evaluation cells <= 12
R3_CELL_FRE_FLOOR = 30.0    # no evaluation cell with FRE < 30
R3_CELL_FKGL_CEIL = 16.0    # no evaluation cell with FKGL > 16

# Registered rounding: FRE and FKGL reported at one decimal place;
# means computed at FULL precision, rounded LAST.
ROUNDING_DP = 1


# --------------------------------------------------------------------------- #
# Gate A4 synthetic fixtures.  Invented here for the tool
# determinism + direction check; NONE of this text comes from any transcript.
# --------------------------------------------------------------------------- #
FIXTURE_PARAGRAPH = (
    "The cat sat on the mat. A dog ran to the big red log. "
    "We had a lot of fun in the warm sun today."
)
FIXTURE_SIMPLE = "The cat sat on the mat. The dog ran to the log. We had fun today."
FIXTURE_COMPLEX = (
    "The multifaceted epistemological ramifications of institutional "
    "heterogeneity necessitate comprehensive interdisciplinary methodological "
    "reconsideration throughout contemporary organizational infrastructures."
)


# --------------------------------------------------------------------------- #
# Report plumbing
# --------------------------------------------------------------------------- #
def md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "| " + " | ".join(str(h) for h in header) + " |"
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def machinery_versions() -> dict[str, str]:
    """Exact versions of the readability machinery, pinned at Stage A."""
    from importlib.metadata import version as _v

    return {
        "python": sys.version.split()[0],
        "textstat": _v("textstat"),
        "pyphen": _v("pyphen"),
        "nltk": _v("nltk"),
    }


def pinned_stack_versions() -> dict[str, str]:
    """The evidentiary pins carried from Task 10 onward (must stay undisturbed)."""
    import numpy as np
    import pandas as pd
    import pyarrow
    import scipy

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
    }


def env_header(title: str) -> list[str]:
    mv = machinery_versions()
    pin = pinned_stack_versions()
    return [
        f"# {title}",
        "",
        "Environment pin (registered convention): everything evidentiary runs "
        "under the repo venv; the system interpreter is never used.",
        "",
        f"- interpreter: `{sys.executable}`",
        f"- python {mv['python']}, numpy {pin['numpy']}, pandas {pin['pandas']}, "
        f"scipy {pin['scipy']}, pyarrow {pin['pyarrow']} (pinned stack, "
        "undisturbed by this task's install)",
        f"- readability machinery pinned at Stage A: **textstat {mv['textstat']}**, "
        f"**pyphen {mv['pyphen']}** (syllable backend), nltk {mv['nltk']} "
        "",
    ]


# The registered template carries three non-ASCII
# typographic characters that must be minted VERBATIM: the en-dash in
# "Flesch-Kincaid" and the >= / <= relation signs in the targets clause.  These
# are whitelisted; any OTHER non-ASCII in a report (e.g. a stray smart quote)
# still halts, preserving the src/task13 ASCII-hygiene intent.
ALLOWED_NON_ASCII = {
    "–",  # EN DASH  -- "Flesch-Kincaid"
    "≤",  # LESS-THAN OR EQUAL TO
    "≥",  # GREATER-THAN OR EQUAL TO
}


def write_report(name: str, lines: Sequence[str]) -> str:
    """Write the report into OUT_DIR, then enforce the ASCII-safe rule.

    The file lands first so a violation never costs the evidence; an overwrite
    guard refuses to silently replace a halt/failure record.  Non-ASCII output
    halts EXCEPT for the registered typographic whitelist above.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / name
    if target.exists():
        old = target.read_text(encoding="utf-8", errors="replace")
        if "STOP -- DEFECT" in old or "HALTED" in old or "DEFECT RECORD" in old:
            raise RuntimeError(
                f"overwrite guard: {name} holds a halt/defect record and must "
                "not be regenerated in place (registered posture) -- STOP"
            )
    report = "\n".join(lines)
    target.write_text(report, encoding="utf-8")
    bad = sorted({c for c in report if ord(c) > 127 and c not in ALLOWED_NON_ASCII})
    if bad:
        raise RuntimeError(
            f"{name}: unexpected non-ASCII characters in report output "
            f"{[(c, hex(ord(c))) for c in bad]} -- STOP"
        )
    return report
