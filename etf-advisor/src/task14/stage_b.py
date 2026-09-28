"""Task 14 Stage B -- prompt development, both arms, dev personas only.

Gates:
- **B1** dev-only discipline held -- call-log audit shows every Stage B call
  was D1-D3 or the registered probe; zero evaluation-persona calls.
- **B2** both prompts frozen with hashes recorded; both dev logs complete;
  static prompt bound checked for all 14 evaluation personas x both arms
  (rendering only, no generation).

Each run appends one development ITERATION per arm: D1, D2, D3 through the
current template, logged to ``dev_iteration_log_arm_{a,b}.csv``.  The logs are
cumulative across runs, so the report always shows the whole development
history; the report itself reflects the current (final) templates.
"""

from __future__ import annotations

import csv
import json
import re
import sys

from . import (
    I4_PROMPT_BUDGET,
    OUT_DIR,
    PERSONA_PATH,
    PROBE_PROMPT,
    PROMPT_CHAR_BOUND,
    Q_KEYS,
    TRANSCRIPT_DIR,
    env_header,
    generate,
    log_row,
    md_table,
    read_live_pin,
    sha256_file,
    write_report,
    LOG_COLUMNS,
)
from . import elicit_arms as EA


DEV_DIR = TRANSCRIPT_DIR / "dev"
DEV_LOG = {"A": OUT_DIR / "dev_iteration_log_arm_a.csv",
           "B": OUT_DIR / "dev_iteration_log_arm_b.csv"}

DEV_COLUMNS = [
    "iteration", "arm", "persona", "prompt_sha256", "prompt_chars",
    "response_chars", "response", "parse_verdict", "parse_reason",
    "extraction", "extraction_matches_expected", "rubric_trace", "outcome",
    "expected_profile", "outcome_matches_expected",
    "seq", "done_reason", "prompt_eval_count", "eval_count", "wall_seconds",
]

# Every Stage B call must match this: the registered probe, or a dev persona.
# Extended to D4 by amendment T14-A2 ("Stage B+ calls are D1-D4 or the
# registered probe").  The T14-A2 driver emits purposes in this same grammar
# and appends to the same call log, so one audit covers all Stage B work.
PURPOSE_RE = re.compile(r"^stage_b:(warmup_discarded|arm_[ab]:(D[1-4]):iter\d+)$")


def _personas() -> tuple[list[dict], list[dict]]:
    doc = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
    dev = [p for p in doc["personas"] if p["kind"] == "dev"]
    evaluation = [p for p in doc["personas"] if p["kind"] != "dev"]
    return dev, evaluation


def _next_iteration(arm: str) -> int:
    path = DEV_LOG[arm]
    if not path.exists():
        return 1
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return max((int(r["iteration"]) for r in rows), default=0) + 1


def _append_dev_rows(arm: str, rows: list[dict]) -> None:
    path = DEV_LOG[arm]
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DEV_COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_dev_rows(arm: str) -> list[dict]:
    path = DEV_LOG[arm]
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    dev, evaluation = _personas()
    failures: list[str] = []

    # ---- template checks BEFORE any generation ------------------------- #
    issues = EA.template_issues("A") + EA.template_issues("B")

    # ---- pin re-read (no generation) ----------------------------------- #
    pin = read_live_pin()

    calls = []
    if not issues:
        calls.append(generate(PROBE_PROMPT, purpose="stage_b:warmup_discarded"))

        for arm in ("A", "B"):
            n = _next_iteration(arm)
            psha = EA.prompt_sha(arm)
            rows: list[dict] = []
            for p in dev:
                pid = p["id"]
                purpose = f"stage_b:arm_{arm.lower()}:{pid}:iter{n}"
                prompt = EA.render(arm, p["answers"])
                res = generate(prompt, purpose=purpose)
                calls.append(res)
                out = EA.INTERPRET[arm](pid, res.response)
                EA.write_pair(DEV_DIR, f"arm_{arm.lower()}_{pid}_iter{n}",
                              prompt, res.response)
                ex_match = "-"
                if arm == "A" and out.extraction is not None:
                    ex_match = str(out.extraction == p["expected_extraction"])
                rows.append({
                    "iteration": n, "arm": arm, "persona": pid,
                    "prompt_sha256": psha, "prompt_chars": res.prompt_chars,
                    "response_chars": len(res.response),
                    "response": res.response,
                    "parse_verdict": "MALFORMED" if out.malformed else "VALID",
                    "parse_reason": out.parse_reason,
                    "extraction": ("/".join(out.extraction[q] for q in Q_KEYS)
                                   if out.extraction else "-"),
                    "extraction_matches_expected": ex_match,
                    "rubric_trace": out.rubric_trace,
                    "outcome": out.outcome,
                    # D4 (amendment T14-A2) expects `clarify`, which is an
                    # outcome but not a profile, so the expectation is read
                    # from expected_outcome with expected_profile as fallback.
                    "expected_profile": (p.get("expected_outcome")
                                         or p["expected_profile"]),
                    "outcome_matches_expected":
                        str(out.outcome == (p.get("expected_outcome")
                                            or p["expected_profile"])),
                    "seq": res.seq, "done_reason": res.i3_done_reason,
                    "prompt_eval_count": res.i4_prompt_eval_count,
                    "eval_count": res.eval_count,
                    "wall_seconds": f"{res.wall_seconds:.3f}",
                })
            _append_dev_rows(arm, rows)

    # ---- Gate B1: contamination audit ---------------------------------- #
    bad_purposes = [r.purpose for r in calls if not PURPOSE_RE.match(r.purpose)]
    personas_called = sorted({m.group(2) for r in calls
                              if (m := PURPOSE_RE.match(r.purpose)) and m.group(2)})
    eval_ids = {p["id"] for p in evaluation}
    called_eval = sorted(set(personas_called) & eval_ids)

    # Stronger than a purpose-string audit: scan every prompt actually sent to
    # the model, on disk, for any evaluation-persona answer text.
    eval_answers = [(p["id"], q, p["answers"][q]) for p in evaluation for q in Q_KEYS]
    sent_prompts = sorted(TRANSCRIPT_DIR.rglob("*_prompt.txt"))
    contamination: list[str] = []
    for path in sent_prompts:
        text = path.read_text(encoding="utf-8")
        for pid, q, ans in eval_answers:
            if ans in text:
                contamination.append(f"{path.name}: contains {pid} {q}")

    if bad_purposes:
        failures.append(f"Gate B1: {len(bad_purposes)} call(s) with an "
                        f"unregistered purpose: {bad_purposes[:3]}")
    if called_eval:
        failures.append(f"Gate B1: evaluation personas called: {called_eval}")
    if contamination:
        failures.append(f"Gate B1: {len(contamination)} evaluation-persona "
                        f"answer(s) found in sent prompts")

    # ---- Gate B2: static prompt bound, rendering only ------------------ #
    bound_rows = []
    worst = 0
    for arm in ("A", "B"):
        lengths = []
        for p in evaluation:
            rendered = EA.render(arm, p["answers"])      # rendering only
            lengths.append((p["id"], len(rendered)))
        mx = max(lengths, key=lambda t: t[1])
        worst = max(worst, mx[1])
        bound_rows.append([
            f"Arm {arm}", len(lengths), min(t[1] for t in lengths), mx[1],
            f"{mx[0]}", f"<= {PROMPT_CHAR_BOUND}",
            "PASS" if mx[1] <= PROMPT_CHAR_BOUND else "**BREACH**"])
    if worst > PROMPT_CHAR_BOUND:
        failures.append(f"Gate B2: rendered prompt {worst} > {PROMPT_CHAR_BOUND}")
    if issues:
        failures.append(f"template checks: {len(issues)} issue(s)")

    prompt_shas = {arm: EA.prompt_sha(arm) for arm in ("A", "B")}
    dev_rows = {arm: _read_dev_rows(arm) for arm in ("A", "B")}
    iters = {arm: sorted({int(r["iteration"]) for r in dev_rows[arm]})
             for arm in ("A", "B")}
    logs_complete = all(len(dev_rows[arm]) == len(iters[arm]) * len(dev)
                        for arm in ("A", "B")) and all(iters.values())
    if not logs_complete:
        failures.append("Gate B2: dev logs incomplete "
                        f"(A {len(dev_rows['A'])} rows over {len(iters['A'])} "
                        f"iterations; B {len(dev_rows['B'])} rows over "
                        f"{len(iters['B'])} iterations)")

    # ================================================================== #
    lines = env_header("Task 14 Stage B -- prompt development, both arms, "
                       "dev personas only", pin)
    lines += [
        "Determinism standard: **repeat-exact** (measured at Stage A). "
        "No options change is available at this stage and the "
        "options variant is byte-identical to Stage A's, so no re-probe is "
        "registered and none was run.",
        "",
        "## 0. Boundary",
        "",
        "> The Task 12 and Task 13 verdicts are frozen ground truth, and all "
        "Task 9-13 outputs are read-only. The elicitor creates no new profile, "
        "ladder, or budget freedom: its output is exactly one registered "
        "profile string from `PROFILES` (src.task9.allocators) or a clarify "
        "signal. `recommend()` validation is the backstop, never the primary "
        "check. A weak elicitor is a Task 14 finding about the elicitation "
        "layer and re-litigates nothing upstream.",
        "",
        "## 1. Prompt design and the registered asymmetry",
        "",
        "| requirement | Arm A | Arm B |",
        "|---|---|---|",
        "| five registered questions verbatim | yes | yes |",
        "| persona's answers substituted | yes | yes |",
        "| instructs the registered output format | five lines `Qk: <value>` | "
        "exactly one word |",
        "| instructs `unclear` where the text does not support a value | yes | "
        "n/a (Arm B has no extraction) |",
        "| profile semantics / mapping guidance | **none** -- the rubric "
        "decides | **permitted and used** -- horizon guidance, "
        "capacity-over-willingness, when to clarify |",
        "",
        "The architectural difference under test is WHERE the decision lives, "
        "not how well either prompt is written. Arm B is "
        "therefore given the profile semantics its architecture needs, "
        "including that `growth` is the schema maximum and out-of-schema "
        "requests are not honoured -- a property Arm A gets structurally for "
        "free, because its rubric cannot emit anything outside `PROFILES`. ",
        "",
        "Mechanical template checks, run before any generation: each registered "
        "question must appear verbatim in both templates; all five placeholders "
        "must be present and fully consumed at render time; Arm A's template "
        "must contain no profile name (`balanced` exempt -- it is a registered "
        "Q3 domain value) and no `clarify`; Arm B's template must offer all "
        "four registered vocabulary words.",
        "",
        f"**Template issues: {len(issues)}.**"
        + ("" if not issues else " " + "; ".join(issues)),
        "",
    ]

    # ---- dev history ---------------------------------------------------- #
    lines += ["## 2. Development iterations (D1-D3 only)", ""]
    for arm in ("A", "B"):
        rows = dev_rows[arm]
        lines += [
            f"### Arm {arm} -- {len(iters[arm])} iteration(s), {len(rows)} "
            "logged calls",
            "",
            md_table(
                ["iter", "persona", "prompt sha256 (16)", "parse",
                 "extraction", "extraction == expected", "rubric trace",
                 "outcome", "expected", "match"],
                [[r["iteration"], r["persona"], f"`{r['prompt_sha256'][:16]}`",
                  r["parse_verdict"], f"`{r['extraction']}`",
                  r["extraction_matches_expected"], f"`{r['rubric_trace']}`",
                  f"**{r['outcome']}**", r["expected_profile"],
                  "PASS" if r["outcome_matches_expected"] == "True"
                  else "**MISS**"]
                 for r in rows]),
            "",
        ]
        last = [r for r in rows if int(r["iteration"]) == max(iters[arm])] \
            if iters[arm] else []
        hits = sum(1 for r in last if r["outcome_matches_expected"] == "True")
        lines += [f"Arm {arm}, final iteration: **{hits}/{len(last)}** dev "
                  "personas reproduce their registered expected profile.", ""]

    lines += [
        "Dev personas are excluded from every evaluation count (register "
        "Sec. 4). Their only role is prompt development; the numbers above "
        "certify nothing about either arm's evaluation performance.",
        "",
        "## 3. Prompt freeze",
        "",
        md_table(["artifact", "SHA-256", "chars"], [
            ["`src/task14/prompt_arm_a.txt`", f"`{prompt_shas['A']}`",
             len(EA.load_template("A"))],
            ["`src/task14/prompt_arm_b.txt`", f"`{prompt_shas['B']}`",
             len(EA.load_template("B"))],
        ]),
        "",
        "Both prompts are FROZEN at these hashes. Any later edit is an "
        "amendment (T14-Ax) and requires halting first.",
        "",
        "## 4. Gate B1 -- dev-only discipline",
        "",
        md_table(["check", "result"], [
            ["Stage B calls this run", len(calls)],
            ["calls matching the registered purpose grammar",
             f"{len(calls) - len(bad_purposes)}/{len(calls)}"],
            ["distinct personas called", ", ".join(personas_called) or "(none)"],
            ["evaluation personas called",
             ", ".join(called_eval) if called_eval else "**0**"],
            ["prompt files on disk scanned", len(sent_prompts)],
            ["evaluation-persona answer strings searched for",
             len(eval_answers)],
            ["occurrences found in any sent prompt",
             f"**{len(contamination)}**"],
        ]),
        "",
        "The purpose-string audit is backed by a content scan: every prompt "
        "actually sent to the model is on disk, and all "
        f"{len(eval_answers)} evaluation-persona answer strings (14 personas x "
        "5 questions) are searched for in every one of them. Zero occurrences "
        "is the evidence that no evaluation persona has reached the model.",
        "",
        "## 5. Gate B2 -- static prompt bound, rendering only",
        "",
        f"All 14 evaluation personas rendered through BOTH templates -- "
        "rendering only, no generation -- and measured against the registered "
        f"bound of {PROMPT_CHAR_BOUND} characters. Rendering an evaluation "
        "persona is not a call and does not touch the model.",
        "",
        md_table(["arm", "personas rendered", "min chars", "max chars",
                  "longest persona", "bound", "verdict"], bound_rows),
        "",
        f"Worst rendered prompt over both arms: **{worst}** characters "
        f"(bound {PROMPT_CHAR_BOUND}). For reference, invariant I4 caps "
        f"`prompt_eval_count` at {I4_PROMPT_BUDGET} tokens; the dev calls "
        "logged above record the measured token counts.",
        "",
    ]

    b1 = not (bad_purposes or called_eval or contamination)
    b2 = not (issues or worst > PROMPT_CHAR_BOUND or not logs_complete)
    lines += [
        "## 6. Gates",
        "",
        md_table(["gate", "criterion", "evidence", "verdict"], [
            ["B1", "dev-only discipline held -- every Stage B call was D1-D3 or "
                   "the registered probe; zero evaluation-persona calls",
             f"{len(calls)} calls this run, {len(bad_purposes)} off-grammar; "
             f"personas called {personas_called or '(none)'}; content scan "
             f"{len(contamination)} hits over {len(sent_prompts)} sent prompts",
             "PASS" if b1 else "**FAIL**"],
            ["B2", "both prompts frozen with hashes recorded; both dev logs "
                   "complete; static prompt bound checked for all 14 "
                   "evaluation personas x both arms (rendering only)",
             f"A `{prompt_shas['A'][:16]}`, B `{prompt_shas['B'][:16]}`; dev "
             f"logs A {len(dev_rows['A'])} rows / B {len(dev_rows['B'])} rows; "
             f"28 renders, worst {worst} <= {PROMPT_CHAR_BOUND}",
             "PASS" if b2 else "**FAIL**"],
        ]),
        "",
    ]
    if failures:
        lines += ["**Gate failure.** No Stage C work is authorized:", ""]
        lines += [f"- {f}" for f in failures] + [""]

    if calls:
        call_log = OUT_DIR / "stage_b_call_log.csv"
        fresh = not call_log.exists()
        with call_log.open("a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            if fresh:
                w.writeheader()
            for r in calls:
                w.writerow(log_row(r))

    lines += [
        "## 7. Deliverables written by this stage",
        "",
        "- `src/task14/prompt_arm_a.txt`, `prompt_arm_b.txt` (frozen), "
        "`elicit_arms.py`, `stage_b.py`",
        "- `outputs/task14/stage_b_report.md` (this file)",
        "- `outputs/task14/dev_iteration_log_arm_a.csv` "
        f"({len(dev_rows['A'])} rows), `dev_iteration_log_arm_b.csv` "
        f"({len(dev_rows['B'])} rows)",
        "- `outputs/task14/stage_b_call_log.csv` (cumulative per-call log)",
        "- `outputs/task14/transcripts/dev/` (prompt + response per dev call)",
        "",
        "## 8. Stop line",
        "",
        ("STAGE B COMPLETE -- gates B1/B2 PASS. Both prompts frozen. No "
         "evaluation persona has been sent to the model."
         if (b1 and b2) else
         "STAGE B HALTED -- see the gate table. No Stage C work is authorized."),
        "",
    ]

    report = write_report("stage_b_report.md", lines)
    print(report[-2500:])
    return 0 if (b1 and b2) else 1


if __name__ == "__main__":
    sys.exit(main())
