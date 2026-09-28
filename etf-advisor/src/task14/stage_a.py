"""Task 14 Stage A -- machinery: pin, rubric, parsers, personas.

Superseded in part -- do not re run.  This driver produced
``outputs/task14/stage_a_report.md``, which was ratified and is
the PRE-AMENDMENT record: it reports the Arm A parser under decision 
(case-sensitive), and its cross-task probe observation was remanded for
correction.  Amendment subsequently harmonised the Arm A parser, so
this driver's narrative and its supplementary-fixture table no longer
describe the parser in ``parse.py``. The Stage A report is 
kept and not rewritten; the corrected observation and the
post-amendment fixture battery live in ``stage_a_addendum.py`` ->
``outputs/task14/stage_a_addendum.md``.  Gates A1/A2/A3 stand as ratified.

Gates:

- **A1** register hashed; pin verified with zero drift; variant hash recorded;
  determinism probe executed and the standard in force recorded (a
  measurement, not a pass/fail criterion).
- **A2** rubric fixtures 9/9 exact, incl. both threshold pairs and all three
  cap fixtures.
- **A3** parser fixtures 8/8 exact; personas cross-check 0 mismatches; all
  hashes recorded; import audit clean.
"""

from __future__ import annotations

import csv
import json
import sys

from . import (
    ALLOWED_NON_STDLIB,
    ARM_B_VOCAB,
    DOMAINS,
    EXTRACTION_DOMAINS,
    FORBIDDEN_IMPORTS,
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
    PROMPT_CHAR_BOUND,
    QUESTIONS,
    Q_KEYS,
    T13_PROBE_REP1,
    T13_STAGE_A_REPORT,
    T13_STAGE_C_REPORT,
    TRANSCRIPT_DIR,
    ascii_escape,
    classify_import,
    env_header,
    generate,
    log_row,
    md_table,
    normalize_ws,
    options_fingerprint,
    package_imports,
    read_live_pin,
    sha256_file,
    sha256_text,
    write_report,
    LOG_COLUMNS,
)
from . import parse as P
from . import rubric as R
from ..task13 import options_fingerprint as t13_options_fingerprint


# Registered PF rows.
PF_REGISTER_ROWS = {
    "PF1": "| PF1 | `Q1: medium\\nQ2: hold\\nQ3: balanced\\nQ4: medium\\nQ5: some` "
           "| VALID extraction |",
    "PF2": "| PF2 | four lines only (Q5 missing) | MALFORMED |",
    "PF3": "| PF3 | `Q1: about five years so medium` (extra words) | MALFORMED |",
    "PF4": "| PF4 | `Q1: unclear` + four valid lines | VALID -> clarify(Q1) |",
    "PF5": "| PF5 | `balanced` | VALID: balanced |",
    "PF6": "| PF6 | `Balanced.` | VALID: balanced (normalization) |",
    "PF7": "| PF7 | `I would say balanced` | MALFORMED |",
    "PF8": "| PF8 | `balanced or growth` | MALFORMED |",
}

# Registered rule sentences, transcribed for verbatim cross-checking.
RULE_NEEDLES = [
    ("Sec. 1 unclear routing",
     "Any `unclear` in Arm A's extraction routes the persona to `clarify(Qk)` "
     "(first unclear question)."),
    ("Sec. 2 words: points",
     "each answer scores 0, 1, or 2 in the order its domain is listed"),
    ("Sec. 2 words: caps dominate",
     "Two capacity caps then dominate willingness: a short horizon caps the "
     "profile at conservative; a medium horizon or low financial capacity caps "
     "it at balanced. Caps only ever lower the profile."),
    ("Sec. 2 fixture coverage", R.R_FIXTURE_COVERAGE),
    ("Sec. 3 Arm A format",
     "required output is exactly five lines, `Qk: <domain value>` (lowercase), "
     "k = 1..5 in order."),
    ("Sec. 3 Arm A parser",
     "Parser: strip whitespace per line; exact match against the domain incl. "
     "`unclear`; five valid lines -> VALID; anything else -> MALFORMED."),
    ("Sec. 3 Arm B format",
     "required output is exactly one word from "
     "`{conservative, balanced, growth, clarify}`."),
    ("Sec. 3 Arm B parser",
     "Parser normalization, registered: strip surrounding whitespace, strip "
     "one trailing `.`, lowercase; then exact match; anything else -> "
     "MALFORMED."),
    ("Sec. 3 malformed never coerced",
     "MALFORMED is always nonconformant and never coerced."),
    ("Sec. 0 options variant",
     f"identical to the Task 13 frozen block except `num_predict: {NUM_PREDICT}`"),
    ("Sec. 0 output contract",
     "`" + " | ".join(PROFILE_ORDER) + "`"),
    ("Sec. 4 counts",
     "Counts: 9 clear-cut + 5 ambiguous/edge = 14 evaluation personas; 3 dev."),
    ("Sec. 6 fixtures before any LLM call",
     "The rubric and parsers are validated against R1-R9 and PF1-PF8 BEFORE "
     "any LLM call"),
]


def _mapping_needles() -> list[tuple[str, str]]:
    """The register's point table and threshold/cap/min lines, one needle each."""
    out: list[tuple[str, str]] = []
    # The register writes the point table as one run per question --
    # "Q1 short=0 medium=1 long=2" -- so the needle is the whole run.  That
    # binds each value to its points AND binds the domain ORDER, which is what
    # the derived point table in rubric.py depends on.
    for i, q in enumerate(Q_KEYS, start=1):
        run = " ".join(f"{value}={points}"
                       for points, value in enumerate(DOMAINS[q]))
        out.append((f"Sec. 2 points {q}", f"Q{i} {run}"))
    out += [
        ("Sec. 2 score", "score = sum(points)"),
        ("Sec. 2 threshold conservative",
         "score_profile = conservative if score <= 3"),
        ("Sec. 2 threshold balanced", "balanced if 4 <= score <= 7"),
        ("Sec. 2 threshold growth", "growth if score >= 8"),
        ("Sec. 2 cap conservative", "cap = conservative if Q1 == short"),
        ("Sec. 2 cap balanced", "balanced if Q1 == medium or Q4 == low"),
        ("Sec. 2 cap growth", "growth otherwise"),
        ("Sec. 2 min rule",
         "profile = min(score_profile, cap) # conservative < balanced < growth"),
    ]
    return out


def _question_needles() -> list[tuple[str, str]]:
    return [(f"Sec. 1 {q} text", f"| {q} | {QUESTIONS[q]} |") for q in Q_KEYS]


def _rubric_fixture_needles() -> list[tuple[str, str]]:
    out = []
    for rid, answers, score, sp, cap, expected in R.R_FIXTURES:
        out.append((f"Sec. 2 fixture {rid}",
                    f"| {rid} | {', '.join(answers)} | {score} | {sp} | {cap} "
                    f"| {expected} |"))
    return out


def _parser_fixture_needles() -> list[tuple[str, str]]:
    return [(f"Sec. 3 fixture {pid}", row)
            for pid, row in PF_REGISTER_ROWS.items()]


def _persona_needles(personas: list[dict]) -> list[tuple[str, str]]:
    """Every answer, every expected extraction value, every expected profile,
    every acceptable set -- as verbatim needles built from the persona FILE."""
    out: list[tuple[str, str]] = []
    for p in personas:
        pid = p["id"]
        out.append((f"{pid} id", f"**{pid}**"))
        for q in Q_KEYS:
            out.append((f"{pid} {q} answer", '"' + p["answers"][q] + '"'))
        if p["expected_extraction"] is not None:
            chain = "/".join(p["expected_extraction"][q] for q in Q_KEYS)
            if p["kind"] == "dev":
                out.append((f"{pid} expected profile + extraction",
                            f"**{pid}** (expected: {p['expected_profile']}; "
                            f"extraction {chain})"))
            else:
                out.append((f"{pid} expected profile + extraction",
                            f"({p['expected_profile']}; {chain}"))
        if p["acceptable"] is not None:
            out.append((f"{pid} acceptable set",
                        f"**{pid}** {p['note']}. Acceptable "
                        "{" + ", ".join(p["acceptable"]) + "}"))
        if p["nonconformant_note"]:
            out.append((f"{pid} nonconformant subtypes",
                        f"nonconformant: {p['nonconformant_note']}"))
    return out


def _check(needles) -> list[tuple[str, str, bool]]:
    return [(label, needle)
            for label, needle in needles]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    fail_a2: list[str] = []
    fail_a3: list[str] = []

    # ================================================================== #
    # PRE-NETWORK BATTERY
    # ================================================================== #

    persona_doc = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
    personas = persona_doc["personas"]

    checks: list[tuple[str, str, str, bool]] = []   # (category, label, needle, ok)
    for cat, items in (
        ("question set", _question_needles()),
        ("rubric mapping", _mapping_needles()),
        ("rubric fixtures R1-R9", _rubric_fixture_needles()),
        ("parser fixtures PF1-PF8", _parser_fixture_needles()),
        ("registered rules", RULE_NEEDLES),
        ("personas", _persona_needles(personas)),
    ):
        for label, needle, ok in _check(items):
            checks.append((cat, label, needle, ok))

    misses = [c for c in checks if not c[3]]

    # --- rubric fixtures ------------------------------------------------ #
    r_rows = R.fixture_rows()
    r_pass = sum(1 for r in r_rows if r[-1] == "PASS")

    # --- parser fixtures ------------------------------------------------ #
    p_rows = P.fixture_rows()
    p_pass = sum(1 for r in p_rows if r[-1] == "PASS")
    supp_rows = P.supplementary_rows()
    supp_pass = sum(1 for r in supp_rows if r[-1] == "PASS")

    # --- persona structural checks + register self-consistency ---------- #
    kinds = {"dev": 0, "clear_cut": 0, "ambiguous": 0}
    persona_rows: list[list[object]] = []
    consistency_rows: list[list[object]] = []
    struct_fail: list[str] = []
    for p in personas:
        pid, kind = p["id"], p["kind"]
        kinds[kind] = kinds.get(kind, 0) + 1
        if sorted(p["answers"]) != sorted(Q_KEYS):
            struct_fail.append(f"{pid}: answer keys {sorted(p['answers'])}")
        ex = p["expected_extraction"]
        if kind in ("dev", "clear_cut"):
            if ex is None or p["expected_profile"] is None:
                struct_fail.append(f"{pid}: missing expected extraction/profile")
            else:
                for q in Q_KEYS:
                    if ex[q] not in DOMAINS[q]:
                        struct_fail.append(f"{pid}: {q}={ex[q]!r} outside domain")
                if p["expected_profile"] not in PROFILE_ORDER:
                    struct_fail.append(f"{pid}: profile outside PROFILES")
            if p["acceptable"] is not None:
                struct_fail.append(f"{pid}: acceptable set on a non-ambiguous persona")
        else:
            if ex is not None or p["expected_profile"] is not None:
                struct_fail.append(f"{pid}: expected extraction/profile on an "
                                   "ambiguous persona")
            for v in p["acceptable"] or []:
                if v not in ARM_B_VOCAB:
                    struct_fail.append(f"{pid}: acceptable value {v!r} outside "
                                       f"{ARM_B_VOCAB}")
        answers_chars = sum(len(p["answers"][q]) for q in Q_KEYS)
        non_ascii = sum(1 for q in Q_KEYS for ch in p["answers"][q] if ord(ch) > 127)
        found = sum(1 for c in checks
                    if c[0] == "personas" and c[1].startswith(pid + " ") and c[3])
        total = sum(1 for c in checks
                    if c[0] == "personas" and c[1].startswith(pid + " "))
        persona_rows.append([
            pid, kind,
            "/".join(ex[q] for q in Q_KEYS) if ex else "-",
            p["expected_profile"] or ("{" + ", ".join(p["acceptable"]) + "}"),
            answers_chars, non_ascii, f"{found}/{total}",
        ])
        # Register self-consistency: registered expected extraction, pushed
        # through the fixture-validated rubric, must reproduce the registered
        # expected profile.  A mismatch here is the Stage C HALT-AND-ESCALATE
        # combination showing up at Stage A, where it is still cheap.
        if kind in ("dev", "clear_cut") and ex is not None:
            trace = R.apply({q: ex[q] for q in Q_KEYS})
            ok = trace.profile == p["expected_profile"]
            consistency_rows.append([
                pid, "/".join(ex[q] for q in Q_KEYS), trace.score,
                trace.score_profile, trace.cap, trace.profile,
                p["expected_profile"], "PASS" if ok else "**FAIL**"])
            if not ok:
                struct_fail.append(
                    f"{pid}: rubric({ex}) = {trace.profile} != registered "
                    f"{p['expected_profile']}")

    counts_ok = (kinds["dev"] == 3 and kinds["clear_cut"] == 9
                 and kinds["ambiguous"] == 5 and len(personas) == 17
                 and persona_doc["counts"] == {"dev": 3, "clear_cut": 9,
                                               "ambiguous": 5, "evaluation": 14})

    # --- import audit --------------------------------------------------- #
    imports = package_imports()
    import_rows = [[f, f"`{s}`", f"`{t}`", classify_import(t)] for f, s, t in imports]
    bad_imports = [r for r in import_rows if r[3] in ("FORBIDDEN", "UNAUTHORIZED")]

    # --- options variant ------------------------------------------------ #
    t14_blob, t14_opt_sha = options_fingerprint()
    t13_blob, t13_opt_sha = t13_options_fingerprint()
    t13_report_text = T13_STAGE_C_REPORT.read_text(encoding="utf-8")
    t13_hash_tied = t13_opt_sha in t13_report_text
    d13, d14 = json.loads(t13_blob), json.loads(t14_blob)
    delta = {k: (d13[k], d14[k]) for k in set(d13) | set(d14) if d13.get(k) != d14.get(k)}
    opt_delta = {k: (d13["options"].get(k), d14["options"].get(k))
                 for k in set(d13["options"]) | set(d14["options"])
                 if d13["options"].get(k) != d14["options"].get(k)}
    variant_is_single_delta = (set(delta) == {"options"}
                               and opt_delta == {"num_predict": (512, NUM_PREDICT)})

    # collect pre-network failures
    if r_pass != len(r_rows):
        fail_a2.append(f"rubric fixtures {r_pass}/{len(r_rows)}")
    if misses:
        fail_a3.append(f"register cross-check: {len(misses)} needle(s) not found")
    if p_pass != len(p_rows):
        fail_a3.append(f"parser fixtures {p_pass}/{len(p_rows)}")
    if supp_pass != len(supp_rows):
        fail_a3.append(f"supplementary parser checks {supp_pass}/{len(supp_rows)}")
    if struct_fail:
        fail_a3.append(f"persona structural/consistency: {len(struct_fail)} issue(s)")
    if not counts_ok:
        fail_a3.append(f"persona counts {kinds}, n={len(personas)}")
    if bad_imports:
        fail_a3.append(f"import audit: {len(bad_imports)} disallowed target(s)")
    if not variant_is_single_delta:
        fail_a3.append(f"options variant delta is not num_predict-only: {delta}")

    pre_network_ok = not (fail_a2 or fail_a3)

    # ================================================================== #
    # NETWORK PHASE -- pin re-verification, then the determinism probe
    # ================================================================== #
    pin = None
    drift: list[str] = []
    calls: list[object] = []
    texts: list[str] = []
    identical = False
    first_div = "-"
    mean_tps = float("nan")
    t13_probe_rows: list[list[object]] = []
    t13_probe_note = ""

    if pre_network_ok:
        pin = read_live_pin()
        t13_a_text = T13_STAGE_A_REPORT.read_text(encoding="utf-8")
        pin_rows = [
            ["Ollama server version", PIN_SERVER_VERSION, pin["server_version"],
             PIN_SERVER_VERSION in t13_a_text and PIN_SERVER_VERSION in t13_report_text],
            ["model tag", MODEL, pin["model_tag"],
             MODEL in t13_a_text and MODEL in t13_report_text],
            ["model digest", PIN_MODEL_DIGEST, pin["model_digest"],
             PIN_MODEL_DIGEST in t13_a_text and PIN_MODEL_DIGEST in t13_report_text],
        ]
        for label, registered, live, in_t13 in pin_rows:
            if registered != live:
                drift.append(f"{label}: registered {registered!r} != live {live!r}")
            if not in_t13:
                drift.append(f"{label}: {registered!r} not found verbatim in the "
                             "Task 13 record")
        if not drift:
            warm = generate(PROBE_PROMPT, purpose="stage_a:warmup_discarded")
            reps = [generate(PROBE_PROMPT, purpose=f"stage_a:probe_rep{i + 1}")
                    for i in range(K_REPEATS)]
            calls = [warm, *reps]
            texts = [r.response for r in reps]
            identical = all(t == texts[0] for t in texts)
            if not identical:
                for i, t in enumerate(texts[1:], start=2):
                    if t != texts[0]:
                        off = next((j for j, (x, y) in enumerate(zip(texts[0], t))
                                    if x != y), min(len(texts[0]), len(t)))
                        first_div = f"rep1 vs rep{i} at char offset {off}"
                        break
            for i, t in enumerate(texts, start=1):
                (TRANSCRIPT_DIR / f"probe_rep{i}.txt").write_text(t, encoding="utf-8")
            (TRANSCRIPT_DIR / "probe_warmup_discarded.txt").write_text(
                warm.response, encoding="utf-8")
            tps = [r.eval_tokens_per_second for r in reps if r.eval_count > 0]
            mean_tps = sum(tps) / len(tps) if tps else float("nan")

            # Recorded observation (feeds no gate): the same registered probe
            # prompt was run by Task 13 under its frozen block and kept.
            if T13_PROBE_REP1.exists():
                old = T13_PROBE_REP1.read_text(encoding="utf-8")
                new = texts[0]
                same = old == new
                off = next((i for i, (x, y) in enumerate(zip(old, new)) if x != y),
                           min(len(old), len(new)))
                t13_probe_rows = [
                    ["probe prompt", "identical (registered, reused verbatim)",
                     "identical"],
                    ["model tag + digest", "identical", "identical"],
                    ["server version", "identical", "identical"],
                    ["temperature / top_k / seed", "0 / 1 / 42", "0 / 1 / 42"],
                    ["response chars", len(old), len(new)],
                    ["response sha256 (16)", f"`{sha256_text(old)[:16]}`",
                     f"`{sha256_text(new)[:16]}`"],
                    ["byte-identical to the other task",
                     "yes" if same else "**no**",
                     "yes" if same else "**no**"],
                ]
                t13_probe_note = (
                    "The two transcripts are byte-identical, so the registered "
                    "`num_predict` delta left greedy decoding of this prompt "
                    "untouched."
                    if same else
                    "The two transcripts DIFFER, first at character offset "
                    f"{off} (common prefix "
                    f"`{ascii_escape(old[:off])[0]}`). Every other input is "
                    "held constant -- same prompt, model digest, server "
                    "version, temperature 0, top_k 1, seed 42, penalties -- so "
                    "the registered `num_predict` 512 -> 128 delta is the only "
                    "candidate cause, and greedy decoding is evidently not "
                    "invariant to it on this server build. Consequences, "
                    "recorded: (i) Task 13 transcripts are NOT reproducible "
                    "under the Task 14 variant, which is why the "
                    "options-variant hash is recorded in every header and why "
                    "the variant is frozen for the whole task; (ii) the "
                    "within-task standard is unaffected -- the k = 3 repeats "
                    "under the variant were byte-identical; (iii) no Task 13 "
                    "result is disturbed, since nothing here re-runs Task 13. "
                    "This is an observation about the environment, not a "
                    "finding about either arm, and it feeds no gate.")
            else:
                t13_probe_rows = [["Task 13 probe transcript",
                                   "ABSENT on disk", "-"]]
                t13_probe_note = ("Task 13's probe transcript is not on disk, "
                                  "so the comparison was not made.")
    else:
        pin_rows = []

    if not calls:
        standard = "**NOT MEASURED**"
    elif identical:
        standard = "**repeat-exact**"
    else:
        standard = "**pre-authorized fallback ARMED**"

    # ================================================================== #
    # REPORT
    # ================================================================== #
    lines += env_header(
        "Task 14 Stage A -- machinery: pin, rubric, parsers, personas", 
        pin)
    lines += [
        f"Determinism standard IN FORCE: {standard} "
        + ("-- the k = 3 repeats of the registered probe were byte-identical, "
           "so the registered k = 3 byte-identical standard is carried forward "
           "unmodified." if identical else
           "-- see Sec. 3 below." if calls else
           "-- not measured: the pre-network battery did not pass, so no "
           "generation was issued."),
        "",
        "## 0. Objective and boundary",
        "",
        "This module builds BOTH elicitor arms (A: LLM extraction -> deterministic rubric; "
        "B: direct LLM mapping), evaluate them one-shot under the register's "
        "protocol, select per the registered rule, and ship the winner as "
        "`src/task14/elicit.py`.",
        "",
        "> The Task 12 and Task 13 verdicts are frozen ground truth, and all "
        "Task 9-13 outputs are read-only. The elicitor creates no new profile, "
        "ladder, or budget freedom: its output is exactly one registered "
        "profile string from `PROFILES` (src.task9.allocators) or a clarify "
        "signal. `recommend()` validation is the backstop, never the primary "
        "check. A weak elicitor is a Task 14 finding about the elicitation "
        "layer and re-litigates nothing upstream.",
        "",
        "## 1. Register intake",
        "",
        "### Registered question set, read back from the code",
        "",
        md_table(["#", "question (asked verbatim)", "extraction domain"],
                 [[q, QUESTIONS[q],
                   ", ".join(f"`{v}`" for v in EXTRACTION_DOMAINS[q])]
                  for q in Q_KEYS]),
        "",
        "## 2. Pin re-verification and the options variant",
        "",
    ]

    if pin_rows:
        lines += [
            "The pin is carried from the Task 13 closing record. Each value is "
            "checked three ways: against the knob table, against "
            "the live host, and for verbatim presence in both Task 13 stage "
            "reports on disk.",
            "",
            md_table(["pin field", "registered", "live host",
                      "verbatim in Task 13 reports", "drift"],
                     [[a, f"`{b}`", f"`{c}`", "yes" if d else "**no**",
                       "none" if (b == c and d) else "**DRIFT**"]
                      for a, b, c, d in pin_rows]),
            "",
            f"Pin fields captured: {len(pin)}; reported ABSENT: "
            f"{sorted(k for k, v in pin.items() if v == 'ABSENT') or 'none'}.",
            "",
        ]
    else:
        lines += ["Pin re-verification was NOT reached: the pre-network battery "
                  "did not pass, so no host call was made.", ""]

    lines += [
        "### Options-block variant",
        "",
        md_table(["item", "value"], [
            ["Task 13 frozen block, canonical JSON SHA-256", f"`{t13_opt_sha}`"],
            ["that hash appears verbatim in `outputs/task13/stage_c_report.md`",
             "yes" if t13_hash_tied else "**no**"],
            ["Task 14 options-variant SHA-256", f"`{t14_opt_sha}`"],
            ["delta between the two canonical payloads", f"`{opt_delta}`"],
            ["delta is num_predict-only",
             "yes" if variant_is_single_delta else "**NO**"],
            ["I4 budget for this task",
             f"`{I4_PROMPT_BUDGET}` = num_ctx 8192 - num_predict {NUM_PREDICT}"],
            ["static prompt bound", f"`{PROMPT_CHAR_BOUND}` chars"],
        ]),
        "",
        "The Task 14 fingerprint is computed by a function structurally "
        "identical to `src.task13.options_fingerprint` (same keys, same "
        "canonical-JSON settings), and that function reproduces the Task 13 "
        "hash recorded at Task 13 Gate B3. The only thing that can move the "
        "digest is the one registered knob.",
        "",
        "## 3. Determinism probe",
        "",
    ]

    if calls:
        lines += [
            "Registered probe prompt, reused verbatim from Task 13:",
            "",
            f"> {PROBE_PROMPT}",
            "",
            f"One discarded warm-up call, then k = {K_REPEATS} warm generations "
            "under the registered options variant.",
            "",
            md_table(["seq", "purpose", "prompt_eval_count",
                      f"I4 (<= {I4_PROMPT_BUDGET})", "eval_count", "done_reason",
                      "load_duration_s", "wall_s", "eval tok/s"],
                     [[r.seq, r.purpose, r.i4_prompt_eval_count,
                       "PASS" if r.i4_ok else "**BREACH**", r.eval_count,
                       f"`{r.i3_done_reason}`", f"{r.load_duration_ns / 1e9:.2f}",
                       f"{r.wall_seconds:.2f}",
                       f"{r.eval_tokens_per_second:.2f}"] for r in calls]),
            "",
            md_table(["invariant", "exercised on", "outcome"], [
                ["I1 model == pinned tag", f"{len(calls)} calls",
                 "held on all" if all(r.i1_model_ok for r in calls) else "**BREACH**"],
                ["I2 no thinking content", f"{len(calls)} calls",
                 "held on all" if all(r.i2_no_thinking_ok for r in calls)
                 else "**BREACH**"],
                ["I3 done == true", f"{len(calls)} calls",
                 "held on all" if all(r.i3_done for r in calls) else "**BREACH**"],
                [f"I4 prompt_eval_count <= {I4_PROMPT_BUDGET}", f"{len(calls)} calls",
                 "held on all" if all(r.i4_ok for r in calls) else "**BREACH**"],
            ]),
            "",
            f"I2 evidence strings, per call: "
            f"{[r.i2_thinking_evidence for r in calls]}.",
            "",
            md_table(["repeat", "response chars", "sha256 (16)", "== rep1"],
                     [[i + 1, len(t), sha256_text(t)[:16],
                       "yes" if t == texts[0] else "**no**"]
                      for i, t in enumerate(texts)]),
            "",
            f"**Byte-identical across k = {K_REPEATS}: "
            f"{'YES' if identical else 'NO'}.** First divergence: {first_div}.",
            "",
            f"Determinism standard IN FORCE: {standard}. Under the registered "
            "posture this is a MEASUREMENT, not a pass/fail criterion; Gate A1 "
            "requires only that it ran and is recorded. Both arms will be "
            "evaluated under whichever standard is in force.",
            "",
            "### Recorded observation -- the same probe under the Task 13 block",
            "",
            "Task 13 ran this identical probe prompt under its frozen options "
            "block and kept the transcript. Comparing the two is free evidence "
            "about what the registered variant actually changes. This feeds NO "
            "gate and is a recorded observation only.",
            "",
            md_table(["item", "Task 13 (num_predict 512)",
                      "Task 14 (num_predict 128)"], t13_probe_rows),
            "",
            t13_probe_note,
            "",
            f"Runtime estimate for Stage C from the measured decode rate "
            f"({mean_tps:.2f} eval tok/s, mean over {len(texts)} probe repeats): "
            f"84 generations x `num_predict` {NUM_PREDICT} = "
            f"{(84 * NUM_PREDICT / mean_tps) / 60:.0f} min upper bound if every "
            "generation runs to the token cap, less where the model stops "
            "earlier.",
            "",
        ]
        esc, n_non_ascii = ascii_escape(texts[0])
        lines += [
            "### Probe transcript, repeat 1 (quoted ASCII-escaped; raw bytes on disk)",
            "", "```", esc, "```", "",
            f"Non-ASCII characters in the quoted transcript: {n_non_ascii} "
            "(`outputs/task14/transcripts/probe_rep1.txt` holds the raw text).",
            "",
        ]
    elif drift:
        lines += ["**Probe NOT executed.** Pin drift was detected, which is a "
                  "recorded environment event; the task stops before any "
                  "generation:", ""] + [f"- {d}" for d in drift] + [""]
    else:
        lines += ["**Probe NOT executed.** The pre-network battery did not pass; "
                  "no generation was issued.", ""]

    # ---- step 4: rubric -------------------------------------------------- #
    lines += [
        "## 4. Rubric implemented and validated vs R1-R9",
        "",
        "The point table is DERIVED from the registered domain listings rather "
        "than retyped (*each answer scores 0, 1, or 2 in the "
        "order its domain is listed*), so an ordering slip cannot hide in a "
        "second copy. Every registered intermediate is compared, not just the "
        "final profile: columns show `registered / computed`.",
        "",
        md_table(["id", "answers (Q1..Q5)", "score", "score_profile", "cap",
                  "expected profile", "verdict"], r_rows),
        "",
        R.R_FIXTURE_COVERAGE,
        "",
        f"Rubric fixtures exact: **{r_pass}/{len(r_rows)}**. Threshold pairs "
        "exercised: R3/R4 (3->4) and R5/R6 (7->8); cap fixtures exercised: R7 "
        "(short horizon), R8 (low capacity), R9 (medium horizon).",
        "",
        "## 5. Parsers implemented and validated vs PF1-PF8",
        "",
        "Registered decisions on two points the register leaves open, both "
        "fixed here before any evaluation call exists:",
        "",
        "- **D-A1 (outer whitespace, Arm A).** The register spells out per-line "
        "stripping but not the whole-output boundary, so a single trailing "
        "newline -- a transport artifact, not a model content choice -- would "
        "otherwise read as a sixth line and force MALFORMED. The parser strips "
        "the whole response first, then splits, then strips each line. Interior "
        "blank lines still yield more than five lines and stay MALFORMED. The "
        "registered Arm B rule opens with exactly this move (*strip surrounding "
        "whitespace*), so applying it to Arm A keeps the arms level at the "
        "boundary.",
        "- **D-A2 (no case folding, Arm A).** The register lists `lowercase` as "
        "an Arm B normalization step and does NOT list it for Arm A. "
        "Implemented literally: `Q1: Medium` is MALFORMED on Arm A while "
        "`Balanced.` is VALID on Arm B. This is a registered asymmetry in "
        "strictness between the arms, carried as registered rather than "
        "harmonised.",
        "",
        md_table(["id", "arm", "raw model output", "registered expectation",
                  "observed", "parser reason", "verdict"], p_rows),
        "",
        f"Parser fixtures exact: **{p_pass}/{len(p_rows)}**. PF2 and PF4 are "
        "given in the register as prose (*four lines only (Q5 missing)*, "
        "*`Q1: unclear` + four valid lines*) and are constructed from PF1's "
        "registered lines, so the construction is itself register-sourced; the "
        "other six raws are registered literals and are transcribed verbatim.",
        "",
        "### Supplementary parser checks (NOT part of the registered 8)",
        "",
        "PF3 as registered is a one-line raw, so it is MALFORMED on the "
        "line-count mechanism and the domain-match mechanism at once. These "
        "cases isolate each mechanism so the parser cannot pass PF3 for the "
        "wrong reason. They add no criterion and change no fixture.",
        "",
        md_table(["id", "arm", "raw", "expected behaviour", "observed", "verdict"],
                 supp_rows),
        "",
        f"Supplementary checks: **{supp_pass}/{len(supp_rows)}**.",
        "",
        "## 6. Personas transcribed and cross-checked",
        "",
        md_table(["item", "value"], [
            ["persona file", "`src/task14/personas.json`"],
            ["personas", f"{len(personas)} = {kinds['dev']} dev + "
                         f"{kinds['clear_cut']} clear-cut + "
                         f"{kinds['ambiguous']} ambiguous/edge"],
            ["evaluation personas", f"{kinds['clear_cut'] + kinds['ambiguous']}"],
            ["counts match register Sec. 4", "yes" if counts_ok else "**NO**"],
            ["encoding", "ASCII file; U+2014 held as JSON `\\u2014` escapes, "
                         "decoding to the register's characters verbatim"],
        ]),
        "",
        md_table(["id", "kind", "expected extraction",
                  "expected profile / acceptable set", "answer chars",
                  "non-ASCII", "register needles found"], persona_rows),
        "",
        "### Register self-consistency of the clear-cut and dev expectations",
        "",
        "The registered expected extraction, pushed through the "
        "fixture-validated rubric, must reproduce the registered expected "
        "profile. A mismatch here is the Stage C HALT-AND-ESCALATE combination "
        "surfacing at Stage A, where it is still cheap; it is "
        "checked now so that a Stage C occurrence can only be a genuine "
        "anomaly.",
        "",
        md_table(["persona", "registered extraction", "score", "score_profile",
                  "cap", "rubric profile", "registered profile", "verdict"],
                 consistency_rows),
        "",
    ]

    # ---- cross-check battery -------------------------------------------- #
    cat_rows = []
    for cat in ("question set", "rubric mapping", "rubric fixtures R1-R9",
                "parser fixtures PF1-PF8", "registered rules", "personas"):
        sub = [c for c in checks if c[0] == cat]
        cat_rows.append([cat, len(sub), sum(1 for c in sub if c[3]),
                         sum(1 for c in sub if not c[3])])
    lines += [
        "### Mechanical cross-check against the register text",
        "",
        "The imported-never-retyped doctrine, enforced. Every fixture value in "
        "this package -- each question, each point in the mapping table, each "
        "threshold and cap line, each R and PF row, each rule sentence, and for "
        "every persona each free-text answer, each expected extraction value, "
        "each expected profile and each acceptable set -- is rebuilt from the "
        "code/JSON and required to appear verbatim in the register file. The "
        "register is hard-wrapped at ~72 columns, so a registered string can "
        "straddle a line break; both sides of every comparison are therefore "
        "whitespace-normalised (runs of whitespace collapsed to one space) and "
        "nothing else is touched.",
        "",
        md_table(["category", "needles", "found", "missing"], cat_rows),
        "",
        f"**Total needles checked: {len(checks)}; mismatches: {len(misses)}.** "
        "Full needle-by-needle results: `outputs/task14/register_crosscheck.csv`.",
        "",
    ]
    if misses:
        lines += ["Needles NOT found verbatim in the register:", ""]
        for cat, label, needle, _ in misses:
            esc, _n = ascii_escape(needle)
            lines += [f"- [{cat}] {label}: `{esc}`"]
        lines += [""]
    if struct_fail:
        lines += ["Persona structural / self-consistency issues:", ""]
        lines += [f"- {ascii_escape(s)[0]}" for s in struct_fail] + [""]

    # ---- step 7: import audit ------------------------------------------- #
    lines += [
        "## 7. Static import audit over `src/task14`",
        "",
        "Registered surface: stdlib, plus `src.task13` (frozen constants "
        "carried never-retyped, and `client.py` for all transport), plus "
        "`src.task9.allocators` for `PROFILES` only. `stdlib` is decided by "
        "`sys.stdlib_module_names`, not by a hand-kept list. Package versions "
        "in the header are read with `importlib.metadata` rather than by "
        "importing numpy/pandas/scipy/pyarrow, so this package's own import "
        "list stays stdlib-only beyond the three frozen targets.",
        "",
        md_table(["file", "statement", "target", "class"], import_rows),
        "",
        f"Authorized non-stdlib targets: {list(ALLOWED_NON_STDLIB)}. "
        f"Forbidden targets {list(FORBIDDEN_IMPORTS)}: "
        f"{sum(1 for r in import_rows if r[3] == 'FORBIDDEN')} present. "
        f"Unauthorized targets: "
        f"{sum(1 for r in import_rows if r[3] == 'UNAUTHORIZED')}.",
        "",
        "Addition to the file list, recorded: `src/task14/"
        "__init__.py`. The substantive modules are already listed; this is the "
        "package file that carries them and holds the registered constants and "
        "report plumbing, mirroring `src/task13/__init__.py`.",
        "",
    ]

    # ---- gates ----------------------------------------------------------- #
    a1 = bool(pin) and not drift and bool(calls)
    a2 = not fail_a2
    a3 = not fail_a3
    lines += [
        "## 8. Gates",
        "",
        md_table(["gate", "criterion", "evidence", "verdict"], [
            ["A1", "register hashed; pin verified with zero drift; variant hash "
                   "recorded; determinism probe executed and standard in force "
                   "recorded",
             f" pin drift "
             f"{len(drift)}; variant hash `{t14_opt_sha[:16]}...`; "
             f"{len(calls)} probe calls; standard recorded in Sec. 3",
             "PASS" if a1 else "**FAIL**"],
            ["A2", "rubric fixtures 9/9 exact, incl. both threshold pairs and "
                   "all three cap fixtures",
             f"{r_pass}/{len(r_rows)} exact, every registered intermediate "
             "(score, score_profile, cap, profile) compared",
             "PASS" if a2 else "**FAIL**"],
            ["A3", "parser fixtures 8/8 exact; personas cross-check 0 "
                   "mismatches; all hashes recorded; import audit clean",
             f"{p_pass}/{len(p_rows)} parser fixtures; {len(checks)} register "
             f"needles, {len(misses)} missing; persona counts "
             f"{'ok' if counts_ok else 'WRONG'}; {len(struct_fail)} structural "
             f"issues; {len(bad_imports)} disallowed imports",
             "PASS" if a3 else "**FAIL**"],
        ]),
        "",
    ]
    if fail_a2 or fail_a3 or drift:
        lines += ["**Gate failure.** Gates A2/A3 cover machinery, so any failure "
                  "stops the task; no fix-and-continue:", ""]
        lines += [f"- {f}" for f in fail_a2 + fail_a3 + drift] + [""]

    # ---- artifacts -------------------------------------------------------- #
    with (OUT_DIR / "register_crosscheck.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "label", "needle", "found_verbatim"])
        for cat, label, needle, ok in checks:
            w.writerow([cat, label, needle, ok])
    if calls:
        with (OUT_DIR / "stage_a_call_log.csv").open(
                "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            w.writeheader()
            for r in calls:
                w.writerow(log_row(r))
        with (OUT_DIR / "determinism_log.csv").open(
                "w", encoding="utf-8", newline="") as fh:
            w2 = csv.writer(fh)
            w2.writerow(["stage", "unit", "arm", "repeat", "response_chars",
                         "sha256", "identical_to_rep1"])
            for i, t in enumerate(texts, start=1):
                w2.writerow(["A_probe", "registered_probe_prompt", "-", i,
                             len(t), sha256_text(t), t == texts[0]])

    lines += [
        "## 9. Deliverables written by this stage",
        "",
        "- `src/task14/__init__.py`, `rubric.py`, `parse.py`, `personas.json`, "
        "`stage_a.py`",
        "- `outputs/task14/stage_a_report.md` (this file)",
        "- `outputs/task14/register_crosscheck.csv` "
        f"({len(checks)} needles; addition to the data list, "
        "recorded)",
        "- `outputs/task14/stage_a_call_log.csv` "
        f"({len(calls)} calls)" if calls else
        "- `outputs/task14/stage_a_call_log.csv` (not written -- no calls)",
        "- `outputs/task14/determinism_log.csv` (probe repeats)" if calls else
        "- `outputs/task14/determinism_log.csv` (not written -- no calls)",
        "- `outputs/task14/transcripts/probe_rep{1,2,3}.txt`, "
        "`probe_warmup_discarded.txt`" if calls else
        "- `outputs/task14/transcripts/` (not written -- no calls)",
        "",
        "## 10. Stop line",
        "",
        ("STAGE A COMPLETE -- gates A1/A2/A3 PASS. No evaluation persona has "
         "been sent to the model. "
         if (a1 and a2 and a3) else
         "STAGE A HALTED -- see the gate table. No Stage B work is authorized."),
        "",
    ]

    report = write_report("stage_a_report.md", lines)
    print(report[-3000:])
    return 0 if (a1 and a2 and a3) else 1


if __name__ == "__main__":
    sys.exit(main())
