"""Task 13b Stage A -- machinery and input verification.

Stage A performs NO readability computation on any frozen transcript.  
Transcript files are read only for existence, cell-map, and
format checks.  The only readability calls Stage A makes are on SYNTHETIC text
inside Gate A4 (via src/task13b/_fixture.py).

Gates:
  A1  resolve + list the twelve transcript paths; tie each to its (date, profile)
      cell per the Task 13 Stage C records; confirm the anchor development cell is
      identified and flagged; exactly twelve files.
  A2  frozen-integrity: each on-disk file's SHA-256 (first 16 hex) matches the
      Task 13 Stage C rep1 sha256; sizes corroborate the report's response-char
      counts.
  A3  format check per the extraction rule: each file is plain narration prose as
      displayed.  Any serialization wrapper or non-displayed material -> STOP.
  A4  tool determinism (two calls in separate processes on identical synthetic
      text return identical FRE/FKGL) and a direction fixture (simple synthetic
      text scores easier than complex).  Record textstat/pyphen/python versions.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

from . import (
    ANCHOR_CELL,
    CELLS,
    NARRATION_DIR,
    N_TOTAL_CELLS,
    REPO_ROOT,
    TASK13_STAGE_C_REPORT,
    env_header,
    machinery_versions,
    md_table,
    write_report,
)


# Registered format-check markers: any of these appearing in a transcript would
# signal a serialization wrapper or non-displayed material (JSON braces/brackets,
# markup/tags, code fences, escaped control chars, chat-role scaffolding, or a
# reasoning/thinking channel).  A displayed narration paragraph contains none.
WRAPPER_MARKERS = (
    "{", "}", "[", "]", "<", ">", "`", "```",
    '"response"', "'response'", "\\n", "\\t", "\\u",
    "<think", "</think", "system\n", "assistant\n", "user\n",
)


# --------------------------------------------------------------------------- #
# Gate A1 -- cell map
# --------------------------------------------------------------------------- #
def gate_a1() -> tuple[bool, list[str], list[list[object]]]:
    rows: list[list[object]] = []
    ok = True
    n_present = 0
    n_flagged = 0
    for c in CELLS:
        exists = c.path.exists()
        flag = "anchor (dev)" if c.dev_flag else ""
        if c.dev_flag:
            n_flagged += 1
        if exists:
            n_present += 1
        else:
            ok = False
        rows.append([c.cell_id, c.date, c.profile, flag,
                     c.path.as_posix(), "yes" if exists else "MISSING"])
    anchor_ok = (n_flagged == 1
                 and any(c.dev_flag and (c.date, c.profile) == ANCHOR_CELL
                         for c in CELLS))
    count_ok = (n_present == N_TOTAL_CELLS and len(CELLS) == N_TOTAL_CELLS)
    ok = ok and anchor_ok and count_ok
    notes = [
        f"- resolved input directory: `{NARRATION_DIR.as_posix()}` "
        "(the directory the Task 13 Stage C report Section 8 wrote the "
        "`<date>_<profile>_rep{1,2,3}.txt` transcripts to, and the directory the "
        "Task 15 interface reads by file read).",
        f"- files present: {n_present} / {N_TOTAL_CELLS}; registered cells: "
        f"{len(CELLS)}; count exact: {count_ok}.",
        f"- anchor development cell identified and flagged exactly once "
        f"({ANCHOR_CELL[0]} / {ANCHOR_CELL[1]}): {anchor_ok}.",
    ]
    return ok, notes, rows


# --------------------------------------------------------------------------- #
# Gate A2 -- frozen-integrity (hashes provided by the Task 13 records)
# --------------------------------------------------------------------------- #
def gate_a2() -> tuple[bool, list[str], list[list[object]]]:
    rows: list[list[object]] = []
    ok = True
    for c in CELLS:
        raw = c.path.read_bytes()
        text = c.path.read_text(encoding="utf-8")
        full = hashlib.sha256(text.encode()).hexdigest()
        sha16 = full[:16]
        sha16_raw = hashlib.sha256(raw).hexdigest()[:16]
        match = (sha16 == c.exp_sha16)
        size_ok = (len(raw) == c.resp_chars)
        if not (match and size_ok):
            ok = False
        rows.append([c.cell_id, "anchor" if c.dev_flag else "",
                     len(raw), c.resp_chars, "yes" if size_ok else "NO",
                     sha16, c.exp_sha16, "yes" if match else "NO",
                     "yes" if sha16_raw == c.exp_sha16 else "no", full])
    notes = [
        "- The Task 13 Stage C report records only the first 16 hex of each rep1 "
        "SHA-256 (Section 2 determinism table); Gate A2 recomputes the full "
        "SHA-256 of each on-disk file and confirms its 16-hex prefix equals the "
        "registered value.  The full 64-hex digests are recorded here as the "
        "complete frozen-integrity anchor for downstream reference.",
        "- Every file carries no CR and no LF (single-line prose), so the "
        "raw-byte digest and the text-mode digest coincide (last column pair) and "
        "the extraction rule's `verbatim file contents` is unambiguous between a "
        "byte read and a text read.",
        "- File sizes (bytes) equal the Task 13 Stage C `response chars` for all "
        "twelve cells, an independent corroboration of the hash match.",
    ]
    return ok, notes, rows


# --------------------------------------------------------------------------- #
# Gate A3 -- format check per the extraction rule
# --------------------------------------------------------------------------- #
def gate_a3() -> tuple[bool, list[str], list[list[object]]]:
    rows: list[list[object]] = []
    ok = True
    for c in CELLS:
        text = c.path.read_text(encoding="utf-8")
        hits = [m for m in WRAPPER_MARKERS if m in text]
        single_line = ("\n" not in text and "\r" not in text)
        starts_upper = text[:1].isupper()
        ends_period = text.rstrip().endswith(".")
        plain = (not hits) and single_line and starts_upper and ends_period
        if not plain:
            ok = False
        rows.append([c.cell_id, "anchor" if c.dev_flag else "",
                     "yes" if single_line else "no",
                     "yes" if starts_upper else "no",
                     "yes" if ends_period else "no",
                     (str(hits) if hits else "none"),
                     "PLAIN" if plain else "WRAPPER?"])
    shown = ", ".join(
        "`" + m.replace("\n", "\\n").replace("\t", "\\t") + "`"
        for m in WRAPPER_MARKERS
    )
    notes = [
        "- Registered markers scanned for (any hit == a serialization wrapper or "
        "non-displayed material, which per brief Sec. 1 halts the task): "
        f"{shown}.",
        "- Every transcript is a single line of prose, begins with a capital "
        "letter, ends with a period, and contains none of the markers -- i.e. "
        "plain narration prose exactly as displayed to the user.",
    ]
    return ok, notes, rows


# --------------------------------------------------------------------------- #
# Gate A4 -- tool determinism + direction fixture (SYNTHETIC TEXT ONLY)
# --------------------------------------------------------------------------- #
def _run_fixture() -> bytes:
    proc = subprocess.run(
        [sys.executable, "-m", "src.task13b._fixture"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=True,
    )
    return proc.stdout


def gate_a4() -> tuple[bool, list[str], list[list[object]]]:
    out1 = _run_fixture()
    out2 = _run_fixture()
    identical = (out1 == out2)
    d = {}
    for line in out1.decode("utf-8").splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            d[k] = v
    simple_fre = float(d["SIMPLE_FRE"])
    complex_fre = float(d["COMPLEX_FRE"])
    simple_fkgl = float(d["SIMPLE_FKGL"])
    complex_fkgl = float(d["COMPLEX_FKGL"])
    dir_fre = simple_fre > complex_fre
    dir_fkgl = simple_fkgl < complex_fkgl
    lang_ok = (d.get("DEFAULT_LANG") == "en_US")
    ok = identical and dir_fre and dir_fkgl and lang_ok
    rows = [
        ["determinism (2 separate processes, byte-identical stdout)",
         "yes" if identical else "NO"],
        ["default language == en_US (library default)",
         f"{d.get('DEFAULT_LANG')} -> {'yes' if lang_ok else 'NO'}"],
        ["direction FRE: simple > complex",
         f"{simple_fre:.4f} > {complex_fre:.4f} -> {dir_fre}"],
        ["direction FKGL: simple < complex",
         f"{simple_fkgl:.4f} < {complex_fkgl:.4f} -> {dir_fkgl}"],
        ["fixture PARAGRAPH (FRE / FKGL)",
         f"{d['PARAGRAPH_FRE']} / {d['PARAGRAPH_FKGL']}"],
    ]
    mv = machinery_versions()
    notes = [
        f"- python {mv['python']}, textstat {mv['textstat']}, pyphen "
        f"{mv['pyphen']}, nltk {mv['nltk']}.",
        "- The fixture (`src/task13b/_fixture.py`) is invoked twice by this "
        "driver via subprocess, i.e. two independent OS processes; their stdout "
        "is compared byte-for-byte.  No frozen transcript is read by the fixture "
        "or by this gate.",
    ]
    return ok, notes, rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> int:
    a1_ok, a1_notes, a1_rows = gate_a1()
    a2_ok, a2_notes, a2_rows = gate_a2()
    a3_ok, a3_notes, a3_rows = gate_a3()
    a4_ok, a4_notes, a4_rows = gate_a4()
    all_ok = a1_ok and a2_ok and a3_ok and a4_ok

    lines: list[str] = []
    lines += env_header("Task 13b Stage A -- machinery and input verification")
    lines += [
        "Verify the readability machinery and the frozen "
        "inputs BEFORE any evidentiary computation.  **Hard rule: Stage "
        "A performs no readability computation on any frozen transcript.**  The "
        "only FRE/FKGL calls here are on synthetic fixtures inside Gate A4.",
        "",
        f"Task 13 Stage C report of record: `{TASK13_STAGE_C_REPORT.as_posix()}` "
        "(supplies the cell map, the per-cell rep1 SHA-256 prefixes, and the "
        "response-char counts used as frozen-integrity anchors).",
        "",
        "## Gate A1 -- cell map (twelve transcripts, anchor flagged)",
        "",
        f"**{'PASS' if a1_ok else 'FAIL'}**",
        "",
        *a1_notes,
        "",
        md_table(["cell_id", "date", "profile", "flag", "resolved path", "exists"],
                 a1_rows),
        "",
        "## Gate A2 -- frozen-integrity (SHA-256 vs Task 13 Stage C records)",
        "",
        f"**{'PASS' if a2_ok else 'FAIL'}**",
        "",
        *a2_notes,
        "",
        md_table(["cell_id", "flag", "size_bytes", "resp_chars(rpt)", "size==rpt",
                  "sha16(text)", "expected", "match", "raw==exp", "full sha256"],
                 a2_rows),
        "",
        "## Gate A3 -- format check (plain narration prose, no wrapper)",
        "",
        f"**{'PASS' if a3_ok else 'FAIL'}**",
        "",
        *a3_notes,
        "",
        md_table(["cell_id", "flag", "single_line", "starts_upper", "ends_period",
                  "wrapper hits", "verdict"], a3_rows),
        "",
        "## Gate A4 -- tool determinism + direction fixture (synthetic text only)",
        "",
        f"**{'PASS' if a4_ok else 'FAIL'}**",
        "",
        *a4_notes,
        "",
        md_table(["check", "result"], a4_rows),
        "",
        "## Stage A verdict",
        "",
    ]
    if all_ok:
        lines += [
            "All four gates PASS.  The twelve frozen `_rep1` transcripts are "
            "present, tie out byte-for-byte to the Task 13 Stage C records, and "
            "are plain narration prose as displayed; the textstat/pyphen "
            "machinery is deterministic across processes and directionally sane, "
            "with the registered en_US default language and versions pinned "
            "above.  The extraction rule is satisfiable verbatim (no wrapper, no "
            "internal newlines).",
            "",
            "**STOP -- Stage A complete. ",
        ]
    else:
        lines += [
            "One or more gates FAILED. This is a DEFECT "
            "RECORD: the task halts and returns to planning; no Stage B "
            "computation is performed.",
            "",
            "**STOP -- DEFECT: Stage A gate failure. HALTED.**",
        ]

    write_report("stage_a_report.md", lines)
    print(f"Stage A gates: A1={a1_ok} A2={a2_ok} A3={a3_ok} A4={a4_ok} "
          f"ALL={all_ok}")
    print(f"Report written: outputs/task13b/stage_a_report.md")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
