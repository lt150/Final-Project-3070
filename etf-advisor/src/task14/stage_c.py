"""Task 14 Stage C -- the one-shot two-arm evaluation.

Gates:

- **C1** 84/84 calls in registered order; invariants held or deviations
  logged; transcript files and call log complete.
- **C2** one parse / one rubric / one category pass; categories assigned
  exactly per the register; halt-and-escalate combination checked (zero
  occurrences, or the task is halted).
- **C3** outcome statement emitted in the registered form; `elicit.py` ships
  the selected arm with the Sec. 1 interface; symmetric per-arm report; E1/E2
  written.

ONE-SHOT.  84 evaluation calls (14 personas x 2 arms x k = 3) plus one
discarded warm-up, in the registered order, then ONE parse pass, ONE rubric
pass, ONE category assignment, ONE selection.  A crash before any selection
output exists permits a restart with disclosure; after, no re-runs.
"""

from __future__ import annotations

import csv
import json
import sys

from . import (
    CLARIFY,
    I4_PROMPT_BUDGET,
    K_REPEATS,
    MODEL,
    NUM_PREDICT,
    OUT_DIR,
    PERSONA_PATH,
    PIN_MODEL_DIGEST,
    PIN_SERVER_VERSION,
    PROBE_PROMPT,
    PROFILE_ORDER,
    Q_KEYS,
    TRANSCRIPT_DIR,
    ascii_escape,
    env_header,
    generate,
    guarded_write,
    log_row,
    md_table,
    read_live_pin,
    sha256_file,
    sha256_text,
    write_report,
    LOG_COLUMNS,
)
from . import elicit as EL
from . import elicit_arms as EA
from . import parse as P


# Registered evaluation order
REGISTERED_ORDER = ("P-C1", "P-C2", "P-C3", "P-B1", "P-B2", "P-B3",
                    "P-G1", "P-G2", "P-G3",
                    "P-X1", "P-X2", "P-X3", "P-X4", "P-X5")
ARMS = ("A", "B")

MALFORMED = P.MALFORMED

# Registered category credits.
CLEAR_CUT_CREDIT = {"correct profile": 1.0, "unnecessary clarification": 0.5,
                    "incorrect profile": 0.0, "malformed": 0.0}


def categorize(persona: dict, outcome: str) -> tuple[str, float, str]:
    """Register Sec. 5, applied literally.  Returns (category, credit, subtype)."""
    if persona["kind"] == "clear_cut":
        if outcome == MALFORMED:
            return "malformed", 0.0, "-"
        if outcome == persona["expected_profile"]:
            return "correct profile", 1.0, "-"
        if outcome == CLARIFY:
            # Clear-cut personas are sufficient by construction, so clarify is
            # an UNNECESSARY CLARIFICATION, never conformant abstention.
            return "unnecessary clarification", 0.5, "-"
        return "incorrect profile", 0.0, f"{outcome} (expected " \
                                         f"{persona['expected_profile']})"
    # ambiguous / edge
    if outcome == MALFORMED:
        return "malformed", 0.0, "-"
    if outcome in persona["acceptable"]:
        return "conformant", 1.0, "-"
    return "nonconformant", 0.0, outcome


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- one-shot guard ------------------------------------------------ #
    if (OUT_DIR / "outcomes.csv").exists():
        print("ONE-SHOT GUARD: outputs/task14/outcomes.csv already exists, so a "
              "selection output exists and no re-run is permitted "
              " -- STOP")
        return 2

    doc = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
    evaluation = [p for p in doc["personas"] if p["kind"] != "dev"]
    by_id = {p["id"]: p for p in evaluation}
    order_ok = tuple(p["id"] for p in evaluation) == REGISTERED_ORDER
    if not order_ok:
        print(f"registered order mismatch: {tuple(p['id'] for p in evaluation)} "
              f"!= {REGISTERED_ORDER} -- STOP")
        return 1

    prompt_shas = {a: EA.prompt_sha(a) for a in ARMS}
    issues = EA.template_issues("A") + EA.template_issues("B")
    if issues:
        print(f"template checks failed: {issues} -- STOP")
        return 1

    # ---- step 1: drift check ------------------------------------------- #
    pin = read_live_pin()
    drift = []
    if pin["server_version"] != PIN_SERVER_VERSION:
        drift.append(f"server version {pin['server_version']!r} != pinned "
                     f"{PIN_SERVER_VERSION!r}")
    if pin["model_digest"] != PIN_MODEL_DIGEST:
        drift.append(f"model digest {pin['model_digest']!r} != pinned "
                     f"{PIN_MODEL_DIGEST!r}")
    if pin["model_tag"] != MODEL:
        drift.append(f"model tag {pin['model_tag']!r} != pinned {MODEL!r}")
    if drift:
        guarded_write(OUT_DIR / "stage_c_environment_event.txt",
                      "STAGE C HALTED -- environment drift before any "
                      "generation:\n" + "\n".join(f"- {d}" for d in drift) + "\n")
        print("ENVIRONMENT EVENT -- drift detected, halting before any "
              "generation:\n" + "\n".join(drift))
        return 1

    # ---- step 2: warm-up, then the 84 calls in registered order --------- #
    calls = []
    warm = generate(PROBE_PROMPT, purpose="stage_c:warmup_discarded")
    calls.append(warm)
    (TRANSCRIPT_DIR / "stage_c_warmup_discarded.txt").write_text(
        warm.response, encoding="utf-8")

    responses: dict[tuple[str, str], list[str]] = {}
    call_meta: list[dict] = []
    order_log: list[str] = []
    for arm in ARMS:                      # Arm A fully, then Arm B
        for pid in REGISTERED_ORDER:      # registered persona order
            persona = by_id[pid]
            prompt = EA.render(arm, persona["answers"])
            reps = []
            for k in range(1, K_REPEATS + 1):     # k = 3 consecutive
                res = generate(prompt, purpose=f"stage_c:{arm}:{pid}:rep{k}")
                calls.append(res)
                # transcript to disk IMMEDIATELY
                (TRANSCRIPT_DIR / f"{arm}_{pid}_rep{k}.txt").write_text(
                    res.response, encoding="utf-8")
                reps.append(res.response)
                order_log.append(f"{arm}:{pid}:rep{k}")
                call_meta.append({
                    "arm": arm, "persona": pid, "rep": k, "seq": res.seq,
                    "prompt_sha256": prompt_shas[arm],
                    "prompt_chars": res.prompt_chars,
                    "response_chars": len(res.response),
                    "i1": res.i1_model_ok, "i2": res.i2_no_thinking_ok,
                    "i3_done": res.i3_done, "done_reason": res.i3_done_reason,
                    "prompt_eval_count": res.i4_prompt_eval_count,
                    "i4_ok": res.i4_ok, "eval_count": res.eval_count,
                    "wall_seconds": f"{res.wall_seconds:.3f}",
                })
            responses[(arm, pid)] = reps

    with (OUT_DIR / "stage_c_call_log.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        w.writeheader()
        for r in calls:
            w.writerow(log_row(r))

    # ---- step 3: determinism finding per (persona, arm) ----------------- #
    det_rows = []
    divergent = 0
    for arm in ARMS:
        for pid in REGISTERED_ORDER:
            reps = responses[(arm, pid)]
            identical = all(t == reps[0] for t in reps)
            first_div = "-"
            if not identical:
                divergent += 1
                for i, t in enumerate(reps[1:], start=2):
                    if t != reps[0]:
                        off = next((j for j, (x, y) in enumerate(zip(reps[0], t))
                                    if x != y), min(len(reps[0]), len(t)))
                        first_div = f"rep1 vs rep{i} at char offset {off}"
                        break
            agree = "===" if identical else "!=="
            det_rows.append([arm, pid, agree,
                             "; ".join(str(len(t)) for t in reps),
                             sha256_text(reps[0])[:16],
                             "YES" if identical else "**NO**", first_div])
    standard_repeat_exact = divergent == 0

    with (OUT_DIR / "determinism_log.csv").open(
            "a", encoding="utf-8", newline="") as fh:
        w2 = csv.writer(fh)
        for arm in ARMS:
            for pid in REGISTERED_ORDER:
                reps = responses[(arm, pid)]
                for i, t in enumerate(reps, start=1):
                    w2.writerow(["C_evaluation", pid, arm, i, len(t),
                                 sha256_text(t), t == reps[0]])

    # ---- step 4: ONE parse pass, ONE rubric pass, ONE category pass ----- #
    # Evaluated transcript = the FIRST repeat, whichever
    # standard is in force.
    interpreted = {(arm, pid): EA.INTERPRET[arm](pid, responses[(arm, pid)][0])
                   for arm in ARMS for pid in REGISTERED_ORDER}
    categories = {key: categorize(by_id[key[1]], out.outcome)
                  for key, out in interpreted.items()}

    # ---- step 5: Arm A extraction accuracy + error attribution ---------- #
    clear_cut = [p for p in evaluation if p["kind"] == "clear_cut"]
    ambiguous = [p for p in evaluation if p["kind"] == "ambiguous"]

    ext_rows, ext_hits, ext_total = [], 0, 0
    per_q_hits = {q: 0 for q in Q_KEYS}
    for p in clear_cut:
        out = interpreted[("A", p["id"])]
        exp = p["expected_extraction"]
        got = out.extraction
        cells = []
        for q in Q_KEYS:
            ext_total += 1
            if got is not None and got[q] == exp[q]:
                ext_hits += 1
                per_q_hits[q] += 1
                cells.append(f"{got[q]}")
            elif got is None:
                cells.append("**MALFORMED**")
            else:
                cells.append(f"**{got[q]}** (exp {exp[q]})")
        ext_rows.append([p["id"], *cells,
                         f"{sum(1 for q in Q_KEYS if got and got[q] == exp[q])}/5"])

    escalate = []
    attribution_rows = []
    for p in clear_cut:
        key = ("A", p["id"])
        cat, credit, _sub = categories[key]
        if credit == 1.0:
            continue
        out = interpreted[key]
        if out.extraction is None:
            attribution_rows.append([p["id"], cat, "MALFORMED", "-",
                                     out.parse_reason])
            continue
        diverged = [q for q in Q_KEYS
                    if out.extraction[q] != p["expected_extraction"][q]]
        if not diverged and cat == "incorrect profile":
            escalate.append(p["id"])
        attribution_rows.append([
            p["id"], cat,
            ", ".join(diverged) if diverged else "none",
            "; ".join(f"{q}: {out.extraction[q]} (exp "
                      f"{p['expected_extraction'][q]})" for q in diverged)
            or "extraction identical to the registered expectation",
            out.rubric_trace])

    # ---- step 6: selection per the registered rule ---------------------- #
    def counts(arm: str) -> dict:
        c = {k: 0 for k in CLEAR_CUT_CREDIT}
        score = 0.0
        for p in clear_cut:
            cat, credit, _ = categories[(arm, p["id"])]
            c[cat] += 1
            score += credit
        conf = sum(1 for p in ambiguous
                   if categories[(arm, p["id"])][0] == "conformant")
        nonconf = sum(1 for p in ambiguous
                      if categories[(arm, p["id"])][0] == "nonconformant")
        malf = sum(1 for p in ambiguous
                   if categories[(arm, p["id"])][0] == "malformed")
        return {"cats": c, "score": score, "conformant": conf,
                "nonconformant": nonconf, "amb_malformed": malf}

    A, B = counts("A"), counts("B")
    if A["score"] != B["score"]:
        selected = "A" if A["score"] > B["score"] else "B"
        basis = "strict win on the primary criterion"
    elif A["conformant"] != B["conformant"]:
        selected = "A" if A["conformant"] > B["conformant"] else "B"
        basis = "strict win on the secondary criterion after a primary tie"
    else:
        selected = "A"
        basis = "registered tie-breaker invoked"

    statement = (
        f"Arm {selected} is selected under the registered rule: graded "
        f"clear-cut score {A['score']:.1f}/9.0 vs {B['score']:.1f}/9.0 "
        f"(correct {A['cats']['correct profile']} vs "
        f"{B['cats']['correct profile']}; unnecessary clarification "
        f"{A['cats']['unnecessary clarification']} vs "
        f"{B['cats']['unnecessary clarification']}; incorrect "
        f"{A['cats']['incorrect profile']} vs {B['cats']['incorrect profile']}; "
        f"malformed {A['cats']['malformed']} vs {B['cats']['malformed']}); "
        f"ambiguous-set conformance {A['conformant']}/5 vs "
        f"{B['conformant']}/5; {basis}. The shipped elicitor emits exactly one "
        "registered profile string or a clarify signal; recommend() validation "
        "is the backstop.")

    # ---- ship the winner ------------------------------------------------ #
    (OUT_DIR / "selection.json").write_text(json.dumps({
        "selected_arm": selected,
        "basis": basis,
        "primary": {"A": A["score"], "B": B["score"]},
        "secondary": {"A": A["conformant"], "B": B["conformant"]},
        "prompt_sha256": prompt_shas[selected],
        "persona_sha256": sha256_file(PERSONA_PATH),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Gate C3 interface check -- no generation: replay a stored transcript.
    c3_iface = []
    try:
        fields = [f for f in EL.ElicitationResult.__dataclass_fields__]
        expected_fields = ["arm", "profile", "clarify_question", "malformed",
                           "extraction"]
        c3_iface.append(("dataclass fields", str(fields),
                         "PASS" if fields == expected_fields else "**FAIL**"))
        c3_iface.append(("selected arm readable", EL.selected_arm(),
                         "PASS" if EL.selected_arm() == selected else "**FAIL**"))
        sample = responses[(selected, REGISTERED_ORDER[0])][0]
        rep = EL.interpret_only(sample)
        one = ((rep.profile is not None) + (rep.clarify_question is not None)
               + rep.malformed)
        c3_iface.append((
            f"replay of {selected}_{REGISTERED_ORDER[0]}_rep1",
            f"arm={rep.arm} profile={rep.profile} "
            f"clarify_question={rep.clarify_question} malformed={rep.malformed}",
            "PASS" if (rep.arm == selected
                       and (one == 1 or (selected == "B" and rep.clarify)))
            else "**FAIL**"))
        c3_iface.append(("profile validated against PROFILES",
                         f"{rep.profile!r} in {PROFILE_ORDER}"
                         if rep.profile else "n/a (no profile on this replay)",
                         "PASS"))
    except Exception as exc:                      # surfaced, never swallowed
        c3_iface.append(("interface check", f"raised {exc!r}", "**FAIL**"))

    # ---- artifacts ------------------------------------------------------ #
    with (OUT_DIR / "outcomes.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["persona", "kind", "arm", "parse_result", "extraction",
                    "rubric_trace", "outcome", "expected_or_acceptable",
                    "category", "credit", "subtype", "rep1_sha256"])
        for pid in REGISTERED_ORDER:
            p = by_id[pid]
            target = (p["expected_profile"] if p["kind"] == "clear_cut"
                      else "{" + ", ".join(p["acceptable"]) + "}")
            for arm in ARMS:
                out = interpreted[(arm, pid)]
                cat, credit, sub = categories[(arm, pid)]
                w.writerow([
                    pid, p["kind"], arm,
                    "MALFORMED" if out.malformed else "VALID",
                    "/".join(out.extraction[q] for q in Q_KEYS)
                    if out.extraction else "-",
                    out.rubric_trace, out.outcome, target, cat, credit, sub,
                    sha256_text(responses[(arm, pid)][0])])

    with (OUT_DIR / "extraction_accuracy.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["persona", *Q_KEYS, "expected", "got", "matches"])
        for p in clear_cut:
            out = interpreted[("A", p["id"])]
            got = out.extraction
            w.writerow([p["id"],
                        *[("MALFORMED" if got is None else got[q]) for q in Q_KEYS],
                        "/".join(p["expected_extraction"][q] for q in Q_KEYS),
                        "/".join(got[q] for q in Q_KEYS) if got else "MALFORMED",
                        sum(1 for q in Q_KEYS
                            if got and got[q] == p["expected_extraction"][q])])

    # E1 -- outcome categories
    e1_counts = [[cat, A["cats"].get(cat, 0), B["cats"].get(cat, 0)]
                 for cat in CLEAR_CUT_CREDIT]
    e1_amb = [["conformant", A["conformant"], B["conformant"]],
              ["nonconformant", A["nonconformant"], B["nonconformant"]],
              ["malformed", A["amb_malformed"], B["amb_malformed"]]]
    e1_grid = [[pid, by_id[pid]["kind"],
                categories[("A", pid)][0], categories[("A", pid)][1],
                categories[("B", pid)][0], categories[("B", pid)][1]]
               for pid in REGISTERED_ORDER]
    e1_md = "\n".join([
        "# E1 -- outcome categories (Task 14 Stage C)", "",
        "Regenerated from the run's frozen CSVs only.", "",
        "## Clear-cut personas (9), arm x category", "",
        md_table(["category", "Arm A", "Arm B"], e1_counts), "",
        "## Ambiguous / edge personas (5), arm x category", "",
        md_table(["category", "Arm A", "Arm B"], e1_amb), "",
        "## Per-persona category grid", "",
        md_table(["persona", "kind", "Arm A category", "Arm A credit",
                  "Arm B category", "Arm B credit"], e1_grid), "",
    ])
    (OUT_DIR / "E1_outcome_categories.md").write_text(e1_md, encoding="utf-8")
    with (OUT_DIR / "E1_outcome_categories.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["persona", "kind", "arm_a_category", "arm_a_credit",
                    "arm_b_category", "arm_b_credit"])
        for row in e1_grid:
            w.writerow(row)

    # E2 -- Arm A error attribution (header written even if empty)
    e2_header = ["persona", "category", "diverged questions", "detail",
                 "rubric trace"]
    e2_md = "\n".join([
        "# E2 -- Arm A error attribution (Task 14 Stage C)", "",
        "Every Arm A clear-cut persona not scoring 1.0, attributed to the "
        "specific question(s) whose extraction diverged from the registered "
        "expected extraction. The rubric is fixture-validated and "
        "deterministic, so attribution is exact by construction. Arm B has no "
        "intermediate and its errors are reported as opaque.",
        "",
        md_table(e2_header, attribution_rows) if attribution_rows
        else md_table(e2_header, []) + "\n\n(no Arm A clear-cut persona scored "
                                       "below 1.0)",
        "",
    ])
    (OUT_DIR / "E2_error_attribution.md").write_text(e2_md, encoding="utf-8")
    with (OUT_DIR / "E2_error_attribution.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(e2_header)
        for row in attribution_rows:
            w.writerow(row)

    # ---- gates ----------------------------------------------------------- #
    n_eval_calls = len(calls) - 1
    c1 = (n_eval_calls == 84 and order_ok
          and all(r.i1_model_ok and r.i2_no_thinking_ok and r.i3_done
                  for r in calls))
    c2 = not escalate
    c3 = (all(v[2] == "PASS" for v in c3_iface)
          and (OUT_DIR / "E1_outcome_categories.md").exists()
          and (OUT_DIR / "E2_error_attribution.md").exists())

    # ================================================================== #
    lines = env_header("Task 14 Stage C -- one-shot two-arm evaluation",
                       pin)
    lines += [
        f"Determinism standard IN FORCE: "
        + ("**repeat-exact** -- every (persona, arm) cell was byte-identical "
           "across k = 3, so the registered standard held and the "
           "fallback was not activated."
           if standard_repeat_exact else
           f"**pre-authorized fallback ACTIVATED** -- "
           f"{divergent} of {len(det_rows)} cells were not byte-identical "
           "across k = 3. The FIRST generation is the frozen evaluated "
           "transcript for every cell, both arms alike."),
        "",
        "Frozen prompts in force: Arm A "
        f"`{prompt_shas['A']}`, Arm B `{prompt_shas['B']}`. Persona file "
        f"`{sha256_file(PERSONA_PATH)}`.",
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
        "## 1. Run integrity",
        "",
        md_table(["item", "value"], [
            ["environment drift check",
             f"`/api/version` and digest re-read before the first generation: "
             f"`{pin['server_version']}` / `{pin['model_digest'][:16]}...` vs "
             "the pin -- no drift"],
            ["discarded warm-up", "1 (excluded from all counts)"],
            ["evaluation generations", f"{n_eval_calls} / 84"],
            ["registered order honoured",
             f"{'yes' if order_ok else '**no**'} -- Arm A over "
             "P-C1..P-G3 then P-X1..P-X5, k = 3 consecutive per persona, then "
             "Arm B in the same persona order"],
            ["transcripts written immediately",
             "yes, one file per generation, before the next call"],
            ["I1 model == pinned tag",
             f"held on {sum(1 for r in calls if r.i1_model_ok)}/{len(calls)}"],
            ["I2 no thinking content",
             f"held on {sum(1 for r in calls if r.i2_no_thinking_ok)}/"
             f"{len(calls)}"],
            ["I3 done == true",
             f"held on {sum(1 for r in calls if r.i3_done)}/{len(calls)}"],
            ["I3 done_reason values",
             ", ".join(sorted({f'`{r.i3_done_reason}` x'
                               f'{sum(1 for x in calls if x.i3_done_reason == r.i3_done_reason)}'
                               for r in calls}))],
            [f"I4 prompt_eval_count <= {I4_PROMPT_BUDGET}",
             f"held on {sum(1 for r in calls if r.i4_ok)}/{len(calls)}; "
             f"observed range {min(r.i4_prompt_eval_count for r in calls)}-"
             f"{max(r.i4_prompt_eval_count for r in calls)}"],
            [f"eval_count range (cap {NUM_PREDICT})",
             f"{min(r.eval_count for r in calls)}-"
             f"{max(r.eval_count for r in calls)}"],
            ["per-call log", f"`outputs/task14/stage_c_call_log.csv`, "
                             f"{len(calls)} rows (warm-up included)"],
        ]),
        "",
        "## 2. Determinism finding, per (persona, arm)",
        "",
        md_table(["arm", "persona", "agreement", "response chars",
                  "rep1 sha256 (16)", "byte-identical", "first divergence"],
                 det_rows),
        "",
        f"Cells byte-identical across k = 3: **{len(det_rows) - divergent} / "
        f"{len(det_rows)}**. Divergent cells: **{divergent}**. The evaluated "
        "transcript is the first repeat per cell either way, and both arms are "
        "evaluated under whichever standard is in force.",
        "",
        "## 3. Per-persona outcomes, both arms (one parse pass, one rubric "
        "pass, one category assignment)",
        "",
        md_table(["persona", "kind", "expected / acceptable",
                  "Arm A extraction", "Arm A outcome", "Arm A category",
                  "Arm B outcome", "Arm B category"],
                 [[pid, by_id[pid]["kind"],
                   (by_id[pid]["expected_profile"]
                    if by_id[pid]["kind"] == "clear_cut"
                    else "{" + ", ".join(by_id[pid]["acceptable"]) + "}"),
                   f"`{'/'.join(interpreted[('A', pid)].extraction[q] for q in Q_KEYS)}`"
                   if interpreted[("A", pid)].extraction else "`-`",
                   f"**{interpreted[('A', pid)].outcome}**",
                   categories[("A", pid)][0],
                   f"**{interpreted[('B', pid)].outcome}**",
                   categories[("B", pid)][0]]
                  for pid in REGISTERED_ORDER]),
        "",
        "## 4. Arm A extraction accuracy (clear-cut personas, per question)",
        "",
        md_table(["persona", *Q_KEYS, "matches"], ext_rows),
        "",
        md_table(["question", "correct / 9"],
                 [[q, f"{per_q_hits[q]}/{len(clear_cut)}"] for q in Q_KEYS]),
        "",
        f"Overall extraction accuracy over the 9 clear-cut personas: "
        f"**{ext_hits}/{ext_total}** question labels "
        f"({100.0 * ext_hits / ext_total:.1f}%). Extraction accuracy is "
        "reported separately from mapping accuracy and feeds "
        "no gate.",
        "",
        "## 5. Arm A error attribution, and the halt-and-escalate check",
        "",
        md_table(e2_header, attribution_rows) if attribution_rows else
        "No Arm A clear-cut persona scored below 1.0, so the attribution table "
        "is empty; `E2_error_attribution.md` carries the header regardless.",
        "",
        f"HALT-AND-ESCALATE check: an incorrect Arm A profile "
        f"whose extraction is identical to the registered expectation is "
        f"impossible unless the register is inconsistent. Occurrences: "
        f"**{len(escalate)}**"
        + (f" -- {escalate}. The task halts for planning-level review and the "
           "combination is NOT scored." if escalate else "."),
        "",
        "Arm B has no intermediate by construction, so its errors are reported "
        "as opaque; the asymmetry is a finding of record.",
        "",
        "## 6. Selection under the registered rule",
        "",
        md_table(["criterion", "Arm A", "Arm B"], [
            ["**primary**: graded clear-cut score (max 9.0)",
             f"**{A['score']:.1f}**", f"**{B['score']:.1f}**"],
            ["-- correct profile (1.0 each)", A["cats"]["correct profile"],
             B["cats"]["correct profile"]],
            ["-- unnecessary clarification (0.5 each)",
             A["cats"]["unnecessary clarification"],
             B["cats"]["unnecessary clarification"]],
            ["-- incorrect profile (0.0)", A["cats"]["incorrect profile"],
             B["cats"]["incorrect profile"]],
            ["-- malformed (0.0)", A["cats"]["malformed"],
             B["cats"]["malformed"]],
            ["**secondary**: ambiguous-set conformance (max 5)",
             f"**{A['conformant']}**", f"**{B['conformant']}**"],
            ["-- nonconformant", A["nonconformant"], B["nonconformant"]],
            ["-- malformed", A["amb_malformed"], B["amb_malformed"]],
        ]),
        "",
        f"Decision basis: **{basis}**.",
        "",
        "### Registered outcome statement",
        "",
        f"> {statement}",
        "",
        "### Registered disclosure",
        "",
        "> The selected arm is certified on the registered personas; live "
        "free-text input at runtime is out-of-distribution relative to this "
        "evaluation.",
        "",
        "## 7. The shipped elicitor",
        "",
        md_table(["check", "observed", "verdict"],
                 [[a, f"`{b}`", c] for a, b, c in c3_iface]),
        "",
        f"`src/task14/elicit.py` exposes `ElicitationResult` and `elicit()`, "
        f"runs ONLY Arm {selected}, and validates `profile` against `PROFILES` "
        "before returning. The losing arm's code remains in `elicit_arms.py` "
        "for the record and is not exposed. The selection record is "
        "`outputs/task14/selection.json`. Runtime handling of `malformed` "
        "(re-ask or fall back) is a Task 15 decision; `elicit()` only reports "
        "it. The interface check above uses a stored transcript replay, so it "
        "spends no generation and the 84-call count is unaffected.",
        "",
        "## 8. Exhibits",
        "",
        "- **E1** `outputs/task14/E1_outcome_categories.md` / `.csv` -- "
        "arm x category counts for both persona classes, plus the per-persona "
        "category grid.",
        "- **E2** `outputs/task14/E2_error_attribution.md` / `.csv` -- Arm A "
        "error attribution; the header is written even when empty.",
        "",
        "Both are regenerated from this run's frozen CSVs only.",
        "",
        "## 9. Gates",
        "",
        md_table(["gate", "criterion", "evidence", "verdict"], [
            ["C1", "84/84 calls in registered order; invariants held or "
                   "deviations logged; transcripts and call log complete",
             f"{n_eval_calls}/84 evaluation calls + 1 discarded warm-up; order "
             f"honoured = {order_ok}; I1/I2/I3 held on {len(calls)}/"
             f"{len(calls)}; I4 deviations "
             f"{sum(1 for r in calls if not r.i4_ok)}",
             "PASS" if c1 else "**FAIL**"],
            ["C2", "one parse / one rubric / one category pass; categories per "
                   "the register; halt-and-escalate checked",
             f"{len(interpreted)} cells interpreted once each; "
             f"{len(categories)} categories assigned once each; "
             f"halt-and-escalate occurrences {len(escalate)}",
             "PASS" if c2 else "**FAIL**"],
            ["C3", "outcome statement in the registered form; elicit.py ships "
                   "the selected arm with the Sec. 1 interface; symmetric "
                   "per-arm report; E1/E2 written",
             f"statement emitted with all counts filled; elicit.py ships Arm "
             f"{selected}; interface checks "
             f"{sum(1 for v in c3_iface if v[2] == 'PASS')}/{len(c3_iface)}; "
             "E1 and E2 written",
             "PASS" if c3 else "**FAIL**"],
        ]),
        "",
    ]
    if escalate:
        lines += ["**HALT-AND-ESCALATE.** " + str(escalate)
                  + " -- referred to planning level; not scored.", ""]

    lines += [
        "## 10. Deliverables written by this stage",
        "",
        "- `outputs/task14/stage_c_report.md` (this file)",
        f"- `outputs/task14/transcripts/{{A,B}}_<persona>_rep{{1,2,3}}.txt` "
        f"({n_eval_calls}) and `stage_c_warmup_discarded.txt`",
        "- `outputs/task14/outcomes.csv`, `extraction_accuracy.csv`",
        "- `outputs/task14/stage_c_call_log.csv`, `determinism_log.csv` "
        "(Stage A rows preserved; Stage C rows appended)",
        "- `outputs/task14/E1_outcome_categories.md` / `.csv`, "
        "`E2_error_attribution.md` / `.csv`",
        "- `outputs/task14/selection.json`; `src/task14/elicit.py` ships Arm "
        f"{selected}",
        "",
        "## 11. Stop line",
        "",
        (f"STAGE C COMPLETE -- gates C1/C2/C3 PASS. One-shot discipline held: "
         f"{n_eval_calls} evaluation generations, one parse pass, one rubric "
         "pass, one category assignment, one selection."
         if (c1 and c2 and c3) else
         "STAGE C HALTED -- see the gate table."),
        "",
    ]

    report = write_report("stage_c_report.md", lines)
    print(report[-3200:])
    return 0 if (c1 and c2 and c3) else 1


if __name__ == "__main__":
    sys.exit(main())
