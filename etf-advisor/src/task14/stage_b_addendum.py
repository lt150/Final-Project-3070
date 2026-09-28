"""Task 14 Stage B ADDENDUM -- amendment T14-A2: dev persona D4.

D4 is the abstention probe.  D1-D3 are clear-cut by construction, so Stage B
never observed either arm emitting ``unclear`` (Arm A) or ``clarify`` (Arm B),
even though five evaluation personas turn on exactly that behaviour.  T14-A2
adds D4 to the development sandbox to exercise the path before Stage C.

Registered expectations (amendment T14-A2): outcome = clarify for BOTH arms;
Arm A must extract ``unclear`` at Q1, Q2 and Q3 (routing clarify(Q1)), with Q4
and Q5 unconstrained at dev.

Procedure: run D4 through both arms under the FROZEN prompts, k = 1, logged.
If both arms meet the registered expectations the prompts STAND at their frozen
hashes and the result is recorded as "D4 exercised, prompts unchanged".  If
either arm fails, a bounded prompt-iteration window opens against D1-D4 only --
this driver reports that and stops; it never edits a prompt itself.

Gate B1's discipline extends: Stage B+ calls are D1-D4 or the registered probe,
audited here over the CUMULATIVE Stage B call log, and the content scan re-runs
over every prompt ever sent, before Stage C.
"""

from __future__ import annotations

import csv
import json
import sys

from . import (
    OUT_DIR,
    PERSONA_PATH,
    PROBE_PROMPT,
    PROMPT_CHAR_BOUND,
    Q_KEYS,
    TRANSCRIPT_DIR,
    UNCLEAR,
    ascii_escape,
    env_header,
    generate,
    log_row,
    md_table,
    normalize_ws,
    read_live_pin,
    sha256_file,
    write_report,
    LOG_COLUMNS,
)
from . import elicit_arms as EA
from . import stage_a as SA
from . import stage_b as SB


# Amendment text transcribed for verbatim cross-checking.
T14A2_NEEDLES = [
    ("T14-A2 registration",
     "dev persona D4 is added to the development sandbox"),
    ("T14-A2 expectation, both arms",
     "Registered expectations: outcome = clarify for BOTH arms."),
    ("T14-A2 Arm A unclear set",
     "Arm A: Q1, Q2, Q3 must extract `unclear` (routing clarify(Q1)); Q4 and "
     "Q5 are unconstrained at dev"),
    ("T14-A2 pass wording",
     'the prompts STAND at their frozen hashes and the result is recorded as '
     '"D4 exercised, prompts unchanged"'),
    ("T14-A2 B1 extension",
     "Stage B+ calls are D1-D4 or the registered probe, and the content scan "
     "re-runs over all sent prompts before Stage C"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SB.DEV_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    doc = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
    personas = doc["personas"]
    d4 = next(p for p in personas if p["id"] == "D4")
    others = [p for p in personas if p["id"] != "D4"]
    evaluation = [p for p in personas if p["kind"] != "dev"]

    persona_sha = sha256_file(PERSONA_PATH)

    issues = EA.template_issues("A") + EA.template_issues("B")
    if issues:
        failures.append(f"template checks: {len(issues)} issue(s)")

    frozen_before = {arm: EA.prompt_sha(arm) for arm in ("A", "B")}

    # ---- run D4 through both arms, k = 1, under the FROZEN prompts ------ #
    pin = read_live_pin()
    calls = []
    results = {}
    if not failures:
        calls.append(generate(PROBE_PROMPT, purpose="stage_b:warmup_discarded"))
        for arm in ("A", "B"):
            n = SB._next_iteration(arm)
            prompt = EA.render(arm, d4["answers"])
            purpose = f"stage_b:arm_{arm.lower()}:D4:iter{n}"
            res = generate(prompt, purpose=purpose)
            calls.append(res)
            out = EA.INTERPRET[arm]("D4", res.response)
            EA.write_pair(SB.DEV_DIR, f"arm_{arm.lower()}_D4_iter{n}",
                          prompt, res.response)
            results[arm] = (n, res, out)
            SB._append_dev_rows(arm, [{
                "iteration": n, "arm": arm, "persona": "D4",
                "prompt_sha256": frozen_before[arm],
                "prompt_chars": res.prompt_chars,
                "response_chars": len(res.response), "response": res.response,
                "parse_verdict": "MALFORMED" if out.malformed else "VALID",
                "parse_reason": out.parse_reason,
                "extraction": ("/".join(out.extraction[q] for q in Q_KEYS)
                               if out.extraction else "-"),
                "extraction_matches_expected": "-",
                "rubric_trace": out.rubric_trace, "outcome": out.outcome,
                "expected_profile": d4["expected_outcome"],
                "outcome_matches_expected":
                    str(out.outcome == d4["expected_outcome"]),
                "seq": res.seq, "done_reason": res.i3_done_reason,
                "prompt_eval_count": res.i4_prompt_eval_count,
                "eval_count": res.eval_count,
                "wall_seconds": f"{res.wall_seconds:.3f}",
            }])

        with (OUT_DIR / "stage_b_call_log.csv").open(
                "a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            for r in calls:
                w.writerow(log_row(r))

    # ---- evaluate the registered D4 expectations ------------------------ #
    check_rows: list[list[object]] = []
    arm_ok = {}
    if results:
        _n, _res, outA = results["A"]
        _n, _res, outB = results["B"]
        a_unclear = {q: (outA.extraction or {}).get(q) for q in d4["expected_unclear"]}
        a_all_unclear = all(v == UNCLEAR for v in a_unclear.values())
        a_clarify_q = outA.clarify_question
        check_rows = [
            ["Arm A", "parse VALID", "VALID",
             "MALFORMED" if outA.malformed else "VALID",
             "PASS" if not outA.malformed else "**FAIL**"],
            ["Arm A", "Q1, Q2, Q3 extract `unclear`",
             "unclear x3",
             ", ".join(f"{q}={v}" for q, v in a_unclear.items()),
             "PASS" if a_all_unclear else "**FAIL**"],
            ["Arm A", "routing clarify(Q1)", "clarify(Q1)",
             f"clarify(Q{a_clarify_q})" if a_clarify_q else str(outA.outcome),
             "PASS" if a_clarify_q == d4["expected_clarify_question"]
             else "**FAIL**"],
            ["Arm A", "Q4, Q5 unconstrained at dev", "any",
             ", ".join(f"{q}={(outA.extraction or {}).get(q)}"
                       for q in d4["unconstrained"]), "n/a"],
            ["Arm A", "outcome", d4["expected_outcome"], outA.outcome,
             "PASS" if outA.outcome == d4["expected_outcome"] else "**FAIL**"],
            ["Arm B", "parse VALID", "VALID",
             "MALFORMED" if outB.malformed else "VALID",
             "PASS" if not outB.malformed else "**FAIL**"],
            ["Arm B", "outcome", d4["expected_outcome"], outB.outcome,
             "PASS" if outB.outcome == d4["expected_outcome"] else "**FAIL**"],
        ]
        arm_ok["A"] = (not outA.malformed and a_all_unclear
                       and a_clarify_q == d4["expected_clarify_question"]
                       and outA.outcome == d4["expected_outcome"])
        arm_ok["B"] = (not outB.malformed
                       and outB.outcome == d4["expected_outcome"])

    both_met = bool(arm_ok) and all(arm_ok.values())

    # ---- Gate B1 extension: cumulative audit ---------------------------- #
    with (OUT_DIR / "stage_b_call_log.csv").open(encoding="utf-8", newline="") as fh:
        all_calls = list(csv.DictReader(fh))
    purposes = [r["purpose"] for r in all_calls]
    bad_purposes = [p for p in purposes if not SB.PURPOSE_RE.match(p)]
    called = sorted({m.group(2) for p in purposes
                     if (m := SB.PURPOSE_RE.match(p)) and m.group(2)})
    eval_ids = {p["id"] for p in evaluation}
    called_eval = sorted(set(called) & eval_ids)

    eval_answers = [(p["id"], q, p["answers"][q]) for p in evaluation for q in Q_KEYS]
    sent_prompts = sorted(TRANSCRIPT_DIR.rglob("*_prompt.txt"))
    contamination = [f"{path.name}: contains {pid} {q}"
                     for path in sent_prompts
                     for pid, q, ans in eval_answers
                     if ans in path.read_text(encoding="utf-8")]

    if bad_purposes:
        failures.append(f"Gate B1: {len(bad_purposes)} off-grammar purpose(s)")
    if called_eval:
        failures.append(f"Gate B1: evaluation personas called: {called_eval}")
    if contamination:
        failures.append(f"Gate B1: {len(contamination)} evaluation-persona "
                        "answer(s) found in sent prompts")
    if results and not both_met:
        failures.append("T14-A2: an arm did not meet the registered D4 "
                        "expectations; the bounded iteration window opens")

    frozen_after = {arm: EA.prompt_sha(arm) for arm in ("A", "B")}
    prompts_unchanged = frozen_before == frozen_after

    # ---- B2 re-check: bound still holds over the 14 evaluation personas -- #
    worst = max(len(EA.render(arm, p["answers"]))
                for arm in ("A", "B") for p in evaluation)

    # ================================================================== #
    lines = env_header("Task 14 Stage B ADDENDUM -- amendment T14-A2, dev "
                       "persona D4 (abstention probe)", pin)
    lines += [
        "Determinism standard IN FORCE: **repeat-exact** (measured at Stage A; "
        "options variant unchanged, so no re-probe is registered).",
        "",
        "## 1. Why D4 exists",
        "",
        "D1-D3 are clear-cut by construction, so Stage B never observed either "
        "arm emitting `unclear` (Arm A) or `clarify` (Arm B) -- the machinery "
        "for it was fixture-proven at Stage A (PF4), but neither arm's "
        "propensity to abstain had been exercised against the model, even "
        "though five evaluation personas turn on exactly that and two accept "
        "`clarify` only. Amendment T14-A2 adds D4 to the development sandbox "
        "to exercise the path before Stage C, under the frozen prompts.",
        "",
        "### D4, as stored",
        "",
        md_table(["question", "answer"],
                 [[q, ascii_escape(d4["answers"][q])[0]] for q in Q_KEYS]),
        "",
        f"Registered expectations: {d4['note']}",
        "",
        "## 2. D4 through both arms, k = 1, frozen prompts",
        "",
    ]

    if results:
        lines += [
            md_table(["arm", "iteration", "prompt sha256 (16)", "raw response",
                      "parse", "extraction", "outcome"],
                     [[arm, results[arm][0], f"`{frozen_before[arm][:16]}`",
                       f"`{ascii_escape(results[arm][1].response)[0].replace(chr(10), chr(92) + 'n')}`",
                       "MALFORMED" if results[arm][2].malformed else "VALID",
                       f"`{'/'.join(results[arm][2].extraction[q] for q in Q_KEYS)}`"
                       if results[arm][2].extraction else "`-`",
                       f"**{results[arm][2].outcome}**"]
                      for arm in ("A", "B")]),
            "",
            "### Registered expectation checks",
            "",
            md_table(["arm", "registered expectation", "expected", "observed",
                      "verdict"], check_rows),
            "",
        ]
        if both_met:
            lines += [
                "**Both arms meet the registered expectations.** Per T14-A2 the "
                "prompts STAND at their frozen hashes; the result of record is:",
                "",
                "> D4 exercised, prompts unchanged.",
                "",
                md_table(["prompt", "SHA-256 before", "SHA-256 after",
                          "unchanged"],
                         [[f"`src/task14/prompt_arm_{a.lower()}.txt`",
                           f"`{frozen_before[a]}`", f"`{frozen_after[a]}`",
                           "yes" if frozen_before[a] == frozen_after[a]
                           else "**NO**"] for a in ("A", "B")]),
                "",
                "No bounded iteration window opened; no prompt byte changed; "
                "the Stage B freeze hashes ratified at Gate B2 remain in force "
                "for Stage C.",
                "",
            ]
        else:
            lines += [
                "**At least one arm did not meet the registered expectations.** "
                "Per T14-A2 a bounded prompt-iteration window against D1-D4 "
                "ONLY now opens. This driver does not edit prompts; it stops "
                "here so the window is opened at planning level with the "
                "evidence above in hand.",
                "",
            ]
    else:
        lines += ["**D4 was NOT run.** A pre-flight check failed; see Sec. 4.",
                  ""]

    lines += [
        "## 3. Gate B1 discipline, extended to D1-D4 (cumulative audit)",
        "",
        md_table(["check", "result"], [
            ["Stage B calls logged in total (all runs)", len(all_calls)],
            ["calls matching the registered purpose grammar",
             f"{len(all_calls) - len(bad_purposes)}/{len(all_calls)}"],
            ["distinct personas called", ", ".join(called) or "(none)"],
            ["evaluation personas called",
             ", ".join(called_eval) if called_eval else "**0**"],
            ["prompt files on disk scanned", len(sent_prompts)],
            ["evaluation-persona answer strings searched for", len(eval_answers)],
            ["occurrences found in any sent prompt", f"**{len(contamination)}**"],
            ["worst rendered evaluation prompt (rendering only)",
             f"{worst} <= {PROMPT_CHAR_BOUND}"],
        ]),
        "",
        "The audit is cumulative over `outputs/task14/stage_b_call_log.csv`, so "
        "it covers every Stage B call ever made, not just this run. The content "
        "scan re-runs over every prompt ever sent, before Stage C.",
        "",
    ]

    lines += [
        "## 4. Deliverables written by this stage",
        "",
        "- `outputs/task14/stage_b_addendum.md` (this file)",
        "- `outputs/task14/dev_iteration_log_arm_{a,b}.csv` (D4 rows appended)",
        "- `outputs/task14/stage_b_call_log.csv` (cumulative; D4 calls appended)",
        "- `outputs/task14/transcripts/dev/arm_{a,b}_D4_iter*.txt`",
        "",
        "## Stop line",
        "",
        ("STAGE B ADDENDUM COMPLETE -- T14-A2 concluded: D4 exercised, prompts "
         "unchanged. Gate B1 discipline holds over the cumulative call log. No "
         "evaluation persona has been sent to the model."
         if not failures else
         "STAGE B ADDENDUM INCOMPLETE -- see above."),
        "",
    ]

    report = write_report("stage_b_addendum.md", lines)
    print(report[-2600:])
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
