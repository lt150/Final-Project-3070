"""Task 14 -- the T14-A2 bounded prompt-iteration record.

Opened because Arm A did not meet the registered D4 expectations under the
Gate B2 frozen prompt: it extracted `long` at Q1 from an answer that names two
irreconcilable horizons, routing clarify(Q2) instead of the registered
clarify(Q1).  Arm B met its expectations and its prompt is NOT touched.

Bound, declared before the window opens and reported either way: content is
restricted to D1-D4 (amendment T14-A2), and this driver runs at most
``MAX_ATTEMPTS`` revision attempts.  Each run is one attempt: it re-runs the
full dev set D1-D4 through BOTH arms under the current prompts, so a fix for D4
cannot silently cost D1-D3, and the passing arm is re-measured under an
unchanged prompt as a determinism control rather than being assumed.

The driver never edits a prompt.
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
    read_live_pin,
    sha256_file,
    write_report,
    LOG_COLUMNS,
)
from . import elicit_arms as EA
from . import stage_b as SB


MAX_ATTEMPTS = 2
ARMS = ("A", "B")


def _expectations_met(pid: str, persona: dict, out) -> tuple[bool, str]:
    """Registered dev expectation for one persona/arm result."""
    if out.malformed:
        return False, "MALFORMED"
    if pid == "D4":
        if out.arm == "A":
            ex = out.extraction or {}
            unclear_ok = all(ex.get(q) == UNCLEAR
                             for q in persona["expected_unclear"])
            route_ok = out.clarify_question == persona["expected_clarify_question"]
            ok = unclear_ok and route_ok and out.outcome == persona["expected_outcome"]
            return ok, (f"Q1-Q3 unclear={unclear_ok}; "
                        f"clarify(Q{out.clarify_question}); {out.outcome}")
        return out.outcome == persona["expected_outcome"], out.outcome
    expected = persona.get("expected_outcome") or persona["expected_profile"]
    detail = out.outcome
    if out.arm == "A" and persona.get("expected_extraction"):
        detail += (" / extraction "
                   + ("exact" if out.extraction == persona["expected_extraction"]
                      else "DIVERGENT"))
    return out.outcome == expected, detail


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SB.DEV_DIR.mkdir(parents=True, exist_ok=True)

    doc = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
    dev = [p for p in doc["personas"] if p["kind"] == "dev"]
    evaluation = [p for p in doc["personas"] if p["kind"] != "dev"]
    by_id = {p["id"]: p for p in dev}

    issues = EA.template_issues("A") + EA.template_issues("B")
    shas = {arm: EA.prompt_sha(arm) for arm in ARMS}
    stage_b_text = (OUT_DIR / "stage_b_report.md").read_text(encoding="utf-8")
    pin = read_live_pin()

    # An "attempt" is a REVISION of Arm A actually exercised inside this
    # window, counted as a distinct Arm A prompt hash in the dev log that is
    # not the Gate B2 hash.  Iterations 1 (original Stage B dev run) and 2 (the
    # D4 run that OPENED the window) both used the Gate B2 prompt and are
    # therefore not attempts.  Counted after the run, from the log itself.
    def _revisions_exercised() -> int:
        return len({r["prompt_sha256"] for r in SB._read_dev_rows("A")
                    if r["arm"] == "A" and r["prompt_sha256"] not in stage_b_text})
    calls = []
    rows: list[list[object]] = []
    met: dict[tuple[str, str], bool] = {}

    if not issues:
        calls.append(generate(PROBE_PROMPT, purpose="stage_b:warmup_discarded"))
        for arm in ARMS:
            n = SB._next_iteration(arm)
            log_rows = []
            for p in dev:
                pid = p["id"]
                prompt = EA.render(arm, p["answers"])
                res = generate(
                    prompt, purpose=f"stage_b:arm_{arm.lower()}:{pid}:iter{n}")
                calls.append(res)
                out = EA.INTERPRET[arm](pid, res.response)
                EA.write_pair(SB.DEV_DIR, f"arm_{arm.lower()}_{pid}_iter{n}",
                              prompt, res.response)
                ok, detail = _expectations_met(pid, p, out)
                met[(arm, pid)] = ok
                ex_match = "-"
                if arm == "A" and out.extraction and p.get("expected_extraction"):
                    ex_match = str(out.extraction == p["expected_extraction"])
                rows.append([
                    n, arm, pid, f"`{shas[arm][:16]}`",
                    "MALFORMED" if out.malformed else "VALID",
                    f"`{'/'.join(out.extraction[q] for q in Q_KEYS)}`"
                    if out.extraction else "`-`",
                    f"**{out.outcome}**", detail,
                    "PASS" if ok else "**FAIL**"])
                log_rows.append({
                    "iteration": n, "arm": arm, "persona": pid,
                    "prompt_sha256": shas[arm], "prompt_chars": res.prompt_chars,
                    "response_chars": len(res.response),
                    "response": res.response,
                    "parse_verdict": "MALFORMED" if out.malformed else "VALID",
                    "parse_reason": out.parse_reason,
                    "extraction": ("/".join(out.extraction[q] for q in Q_KEYS)
                                   if out.extraction else "-"),
                    "extraction_matches_expected": ex_match,
                    "rubric_trace": out.rubric_trace, "outcome": out.outcome,
                    "expected_profile": (p.get("expected_outcome")
                                         or p["expected_profile"]),
                    "outcome_matches_expected": str(ok),
                    "seq": res.seq, "done_reason": res.i3_done_reason,
                    "prompt_eval_count": res.i4_prompt_eval_count,
                    "eval_count": res.eval_count,
                    "wall_seconds": f"{res.wall_seconds:.3f}",
                })
            SB._append_dev_rows(arm, log_rows)

        with (OUT_DIR / "stage_b_call_log.csv").open(
                "a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            for r in calls:
                w.writerow(log_row(r))

    all_met = bool(met) and all(met.values())
    attempt_no = _revisions_exercised()

    # ---- Gate B1, cumulative ------------------------------------------- #
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
    worst = max(len(EA.render(arm, p["answers"]))
                for arm in ARMS for p in evaluation)

    failures: list[str] = []
    if issues:
        failures.append(f"template checks: {len(issues)} issue(s)")
    if bad_purposes:
        failures.append(f"Gate B1: {len(bad_purposes)} off-grammar purpose(s)")
    if called_eval:
        failures.append(f"Gate B1: evaluation personas called: {called_eval}")
    if contamination:
        failures.append(f"Gate B1: {len(contamination)} evaluation-persona "
                        "answer(s) in sent prompts")
    if worst > PROMPT_CHAR_BOUND:
        failures.append(f"Gate B2: rendered prompt {worst} > {PROMPT_CHAR_BOUND}")
    if not all_met:
        failures.append("T14-A2 iteration: the dev set D1-D4 is not fully met "
                        "under the current prompts")
    if attempt_no > MAX_ATTEMPTS:
        failures.append(f"T14-A2 iteration: declared bound of {MAX_ATTEMPTS} "
                        f"revision attempts exceeded ({attempt_no} exercised)")

    # ================================================================== #
    lines = env_header("Task 14 -- T14-A2 bounded prompt-iteration record",
                       pin)
    lines += [
        "## 0. Why the window opened",
        "",
        "Under the Gate B2 frozen prompts, D4 was run through both arms "
        "(`outputs/task14/stage_b_addendum.md` record). "
        "Arm B met the registered expectations. Arm A did not: from "
        "an answer naming two irreconcilable horizons it extracted `long` at "
        "Q1 rather than `unclear`, so it routed clarify(Q2) instead of the "
        "registered clarify(Q1). Its outcome was still `clarify`, so the "
        "failure was in WHICH question it flagged, not in whether it "
        "abstained. Amendment T14-A2 opens a bounded prompt-iteration window "
        "against D1-D4 only.",
        "",
        "## 1. Bound and scope, declared",
        "",
        md_table(["item", "value"], [
            ["content available to the window", "dev personas D1-D4 only "
             "(amendment T14-A2)"],
            ["declared attempt bound", f"{MAX_ATTEMPTS} revision attempts"],
            ["arms revised", "Arm A only -- Arm B met its expectations and its "
                             "prompt is untouched"],
            ["evaluation personas involved", "none, at any point"],
            ["dev set re-run each attempt", "D1-D4 through BOTH arms"],
        ]),
        "",
        "Both arms are re-run every attempt for two reasons: a fix aimed at D4 "
        "must not silently cost D1-D3, and the unrevised arm is re-measured "
        "rather than assumed, which doubles as a determinism control on an "
        "unchanged prompt.",
        "",
        "## 2. The Arm A revision",
        "",
        "The revision is a general extraction rule and names nothing specific "
        "to D4. The prompt previously listed `self-contradictory` inside a "
        "catch-all clause about answers that support NO label; the model read "
        "an answer supporting TWO labels as a case for picking the better one. "
        "The revision separates those cases and states the multi-label rule "
        "explicitly:",
        "",
        "> If the answer supports more than one of that question's labels -- "
        "for example it gives two different answers, or changes its mind -- "
        "then it supports none of them: output: unclear. Do not pick whichever "
        "seems more likely, and do not reconcile the two.",
        "",
        "This is the definition of `unclear` (*a first-class "
        "output whenever the text does not support a registered value*) made "
        "operative for the two-label case. It carries no profile semantics and "
        "no mapping guidance, so Arm A's registered constraint still holds. "
        "Note that `balanced` on Q3 is a single label meaning both, so an "
        "answer like D2's \"A bit of both\" supports exactly one label and is "
        "not caught by this rule -- D2's result below is the check on that.",
        "",
        "## 3. Attempt results -- D1-D4 through both arms",
        "",
    ]
    if rows:
        lines += [
            md_table(["iter", "arm", "persona", "prompt sha256 (16)", "parse",
                      "extraction", "outcome", "detail", "verdict"], rows),
            "",
            f"Dev set met on both arms: **{all_met}** "
            f"({sum(1 for v in met.values() if v)}/{len(met)} persona-arm "
            "cells).",
            "",
        ]
    else:
        lines += ["No attempt was run: the pre-flight template checks failed.",
                  ""]

    lines += [
        "## 4. Prompt freeze after the window",
        "",
        md_table(["prompt", "current hash",
                  "hash present in the ratified `stage_b_report.md`",
                  "revised in this window"], [
            [f"`src/task14/prompt_arm_{a.lower()}.txt`", f"`{shas[a]}`",
             "yes -- unchanged since Gate B2" if shas[a] in stage_b_text
             else "no -- this is a new hash",
             "no -- byte-unchanged" if shas[a] in stage_b_text
             else "yes -- multi-label rule added"]
            for a in ARMS]),
        "",
        "Whether each prompt moved is established by looking its CURRENT hash "
        "up in the ratified Gate B2 report rather than by retyping the old "
        "value: Arm B's hash still resolves there, Arm A's does not. Arm A is "
        "re-frozen at the hash above and is the prompt Stage C will use. Any "
        "later edit to either prompt is a fresh amendment.",
        "",
        "## 5. Gate B1 / B2 discipline, cumulative",
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
    ]

    lines += [
        "## 6. Deliverables written by this stage",
        "",
        "- `outputs/task14/stage_b_t14a2_iteration.md` (this file)",
        "- `outputs/task14/dev_iteration_log_arm_{a,b}.csv` (attempt rows "
        "appended)",
        "- `outputs/task14/stage_b_call_log.csv` (cumulative)",
        "- `outputs/task14/transcripts/dev/` (prompt + response per call)",
        "- `src/task14/prompt_arm_a.txt` re-frozen; `prompt_arm_b.txt` "
        "untouched",
        "",
        "## 7. Stop line",
        "",
        ("T14-A2 CONCLUDED -- the bounded window closed within its declared "
         "bound. Arm A re-frozen at a new hash, Arm B byte-unchanged, D1-D4 "
         "met on both arms. No evaluation persona has been sent to the model. "
         if not failures else
         "T14-A2 ITERATION INCOMPLETE -- see the check table."),
        "",
    ]

    report = write_report("stage_b_t14a2_iteration.md", lines)
    print(report[-2600:])
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
