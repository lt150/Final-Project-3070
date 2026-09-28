"""Registered output formats and parsers,
plus fixtures PF1-PF8 and battery S1-S7.

- **Arm A extraction call**: one call per persona, all five Q&A pairs in the
  prompt; required output is exactly five lines, ``Qk: <domain value>``
  (lowercase), k = 1..5 in order.  Parser: strip whitespace per line; exact
  match against the domain incl. ``unclear``; five valid lines -> VALID;
  anything else -> MALFORMED.
- **Arm B mapping call**: one call per persona; required output is exactly one
  word from ``{conservative, balanced, growth, clarify}``.  Parser
  normalization, registered: strip surrounding whitespace, strip one trailing
  ``.``, lowercase; then exact match; anything else -> MALFORMED.
- MALFORMED is always nonconformant and never coerced.

**Amendment T14-A1** . Arm A per-line parsing is harmonised 
with Arm B's registered normalization: for each of the five lines,
split at the first ``:``; the label must equal ``Qk`` case-insensitively after
whitespace stripping; the value token then receives exactly the registered Arm
B normalization sequence (strip surrounding whitespace, strip one trailing
``.``, lowercase) before exact domain match.  Extra words remain MALFORMED.
Rationale, recorded: the strictness asymmetry was a register drafting
inconsistency, not an experimental contrast; parser-strictness differences
would confound the architectural comparison.

Decision record:

**D-A1 -- outer whitespace on Arm A.**  The
register spells out per-line stripping but not the whole-output boundary, so a
single trailing newline (a transport artifact rather than a model content
choice) would otherwise read as a sixth line and force MALFORMED.  This parser
strips the whole response first, then splits on newlines, then strips each
line.  Interior blank lines still produce more than five lines and are
MALFORMED.

**D-A2 -- no case folding on Arm A.  SUPERSEDED BY T14-A1.**  Stage A read the
register literally (``lowercase`` listed for Arm B, not for Arm A) and made Arm
A case-sensitive.  T14-A1 reverses that: both arms now run the identical
registered normalization sequence.  The pre-amendment behaviour and its
literal-reading fixture result are preserved in
``outputs/task14/stage_a_report.md``, which is the pre-amendment record and is
not rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ARM_B_VOCAB, EXTRACTION_DOMAINS, Q_KEYS
from . import rubric as _rubric

MALFORMED = "MALFORMED"
VALID = "VALID"


def registered_normalization(token: str) -> str:
    """The registered normalization sequence, in the registered order and no
    other: strip surrounding whitespace, strip ONE trailing ``.``, lowercase.

    Deliberately does NOT re-strip after removing the ``.`` -- the register
    fixes the order, and fixture S4/S7 pin that consequence.  Under T14-A1 this
    single function serves BOTH arms, so no strictness difference between the
    arms can survive in the parser layer.
    """
    t = token.strip()
    if t.endswith("."):
        t = t[:-1]
    return t.lower()


@dataclass(frozen=True)
class ArmAParse:
    """Result of parsing one Arm A extraction response."""
    verdict: str                       # VALID | MALFORMED
    extraction: dict[str, str] | None
    reason: str

    @property
    def valid(self) -> bool:
        return self.verdict == VALID


@dataclass(frozen=True)
class ArmBParse:
    """Result of parsing one Arm B mapping response."""
    verdict: str                       # VALID | MALFORMED
    value: str | None                  # a PROFILES string or 'clarify'
    reason: str

    @property
    def valid(self) -> bool:
        return self.verdict == VALID


def parse_arm_a(raw: str) -> ArmAParse:
    """Five lines ``Qk: <domain value>``, k = 1..5 in order.  T14-A1 + D-A1."""
    lines = [ln.strip() for ln in raw.strip().split("\n")]
    if len(lines) != 5:
        return ArmAParse(MALFORMED, None, f"line count {len(lines)} != 5")

    extraction: dict[str, str] = {}
    for i, (key, line) in enumerate(zip(Q_KEYS, lines), start=1):
        if ":" not in line:
            return ArmAParse(MALFORMED, None, f"line {i} has no ':' separator")
        label, value = line.split(":", 1)          # split at the FIRST ':'
        if label.strip().lower() != key.lower():   # 'Qk', case-insensitive
            return ArmAParse(MALFORMED, None,
                             f"line {i} label {label.strip()!r} != {key!r} "
                             "(case-insensitive)")
        token = registered_normalization(value)
        if token not in EXTRACTION_DOMAINS[key]:
            return ArmAParse(MALFORMED, None,
                             f"line {i} value {token!r} not in domain "
                             f"{EXTRACTION_DOMAINS[key]}")
        extraction[key] = token
    return ArmAParse(VALID, extraction, "five valid lines")


def parse_arm_b(raw: str) -> ArmBParse:
    """Registered normalization, applied in the registered order and no other."""
    token = registered_normalization(raw)
    if token not in ARM_B_VOCAB:
        return ArmBParse(MALFORMED, None,
                         f"normalized {token!r} not in {ARM_B_VOCAB}")
    return ArmBParse(VALID, token, "exact match after registered normalization")


# --------------------------------------------------------------------------- #
# Registered fixtures PF1-PF8.  T14-A1 leaves all eight
# expectations unchanged; they are re-run against the amended parser.
# --------------------------------------------------------------------------- #
# PF1, PF3, PF5-PF8 are given as literal raw strings in the register and are
# transcribed verbatim (the register renders PF1's newlines as literal `\n`).
# PF2 and PF4 are given in prose -- "four lines only (Q5 missing)" and
# "`Q1: unclear` + four valid lines" -- and are CONSTRUCTED from PF1's
# registered lines so the construction itself is register-sourced.
PF1_RAW = "Q1: medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some"
PF1_LINES = PF1_RAW.split("\n")
PF2_RAW = "\n".join(PF1_LINES[:4])                       # Q5 missing
PF3_RAW = "Q1: about five years so medium"
PF4_RAW = "\n".join(["Q1: unclear", *PF1_LINES[1:]])     # + four valid lines

PF1_EXTRACTION = {"Q1": "medium", "Q2": "hold", "Q3": "balanced",
                  "Q4": "medium", "Q5": "some"}
PF4_EXTRACTION = {**PF1_EXTRACTION, "Q1": "unclear"}

# (id, arm, raw, register's "expected parse" cell, expected verdict,
#  expected payload, expected clarify index)
PF_FIXTURES: tuple[tuple[str, str, str, str, str, object, int | None], ...] = (
    ("PF1", "A", PF1_RAW, "VALID extraction", VALID, PF1_EXTRACTION, None),
    ("PF2", "A", PF2_RAW, "MALFORMED", MALFORMED, None, None),
    ("PF3", "A", PF3_RAW, "MALFORMED", MALFORMED, None, None),
    ("PF4", "A", PF4_RAW, "VALID -> clarify(Q1)", VALID, PF4_EXTRACTION, 1),
    ("PF5", "B", "balanced", "VALID: balanced", VALID, "balanced", None),
    ("PF6", "B", "Balanced.", "VALID: balanced (normalization)", VALID,
     "balanced", None),
    ("PF7", "B", "I would say balanced", MALFORMED, MALFORMED, None, None),
    ("PF8", "B", "balanced or growth", MALFORMED, MALFORMED, None, None),
)


def fixture_rows() -> list[list[object]]:
    """Run PF1-PF8.  PF4 additionally exercises the registered clarify routing."""
    rows: list[list[object]] = []
    for pid, arm, raw, expected_cell, verdict, payload, clarify_k in PF_FIXTURES:
        if arm == "A":
            got = parse_arm_a(raw)
            ok = got.verdict == verdict and got.extraction == payload
            observed = got.verdict
            if got.valid:
                routing = _rubric.route(got.extraction)
                if clarify_k is not None:
                    ok = ok and routing.clarify_question == clarify_k
                    observed = f"{got.verdict} -> clarify(Q{routing.clarify_question})"
                else:
                    observed = f"{got.verdict} extraction"
            detail = got.reason
        else:
            got = parse_arm_b(raw)
            ok = got.verdict == verdict and got.value == payload
            observed = got.verdict if not got.valid else f"{got.verdict}: {got.value}"
            detail = got.reason
        shown = raw.replace("\n", "\\n")
        rows.append([pid, arm, f"`{shown}`", expected_cell, observed, detail,
                     "PASS" if ok else "**FAIL**"])
    return rows


def fixtures_pass() -> bool:
    return all(r[-1] == "PASS" for r in fixture_rows())


# --------------------------------------------------------------------------- #
# Post-amendment battery S1-S7.
#
#   status "T14-A1"        -- registered by amendment T14-A1; S3's expectation
#                             is FLIPPED relative to the Stage A record.
#   status "supplementary" -- Stage A isolation check that survives T14-A1
#                             unchanged.
#
# Each entry: (id, arm, raw, expected verdict, expected payload, status, note)
# --------------------------------------------------------------------------- #
def _swap_q1(first_line: str) -> str:
    return "\n".join([first_line, *PF1_LINES[1:]])


EXTRA_FIXTURES: tuple[
    tuple[str, str, str, str, object, str, str], ...] = (
    ("S1", "A", _swap_q1("Q1: about five years so medium"), MALFORMED, None,
     "supplementary",
     "five lines, Q1 value outside the domain after normalization; isolates "
     "the domain-match mechanism from PF3's line count. Extra words remain "
     "MALFORMED under T14-A1."),
    ("S2", "A", PF1_RAW + "\n", VALID, PF1_EXTRACTION, "supplementary",
     "one trailing newline is a transport artifact, absorbed by the D-A1 "
     "whole-output strip (D-A1 stands, ratified as S14-E1)."),
    ("S3", "A", _swap_q1("Q1: Medium"), VALID, PF1_EXTRACTION, "T14-A1",
     "capitalised value. Expectation FLIPPED by T14-A1: the value token now "
     "receives the registered Arm B normalization, so it lowercases to "
     "'medium'. The Stage A literal-reading result (MALFORMED under D-A2) is "
     "preserved in the Stage A report as the pre-amendment record."),
    ("S4", "B", "balanced .", MALFORMED, None, "supplementary",
     "the registered order strips whitespace once, then one trailing '.', and "
     "does not re-strip."),
    ("S5", "A", _swap_q1("q1: medium"), VALID, PF1_EXTRACTION, "T14-A1",
     "lowercase label; the label matches 'Qk' case-insensitively under "
     "T14-A1."),
    ("S6", "A", _swap_q1("Q1: medium."), VALID, PF1_EXTRACTION, "T14-A1",
     "one trailing '.' on the value, stripped by the registered "
     "normalization."),
    ("S7", "A", _swap_q1("Q1: medium ."), MALFORMED, None, "T14-A1",
     "mirrors S4: strip whitespace once, strip one trailing '.', no re-strip, "
     "so the residual trailing space fails the exact domain match."),
)


def extra_fixture_rows() -> list[list[object]]:
    """Run S1-S7 against the amended parser."""
    rows: list[list[object]] = []
    for sid, arm, raw, verdict, payload, status, note in EXTRA_FIXTURES:
        if arm == "A":
            got = parse_arm_a(raw)
            ok = got.verdict == verdict and got.extraction == payload
            observed = (f"{got.verdict} " + "/".join(
                got.extraction[q] for q in Q_KEYS)) if got.valid else got.verdict
        else:
            got = parse_arm_b(raw)
            ok = got.verdict == verdict and got.value == payload
            observed = f"{got.verdict}: {got.value}" if got.valid else got.verdict
        expected = verdict if payload is None else (
            f"{verdict} " + "/".join(payload[q] for q in Q_KEYS)
            if arm == "A" else f"{verdict}: {payload}")
        rows.append([sid, status, arm, f"`{raw.replace(chr(10), chr(92) + 'n')}`",
                     expected, observed, note, "PASS" if ok else "**FAIL**"])
    return rows


def extra_fixtures_pass() -> bool:
    return all(r[-1] == "PASS" for r in extra_fixture_rows())


# Backwards-compatible alias: the Stage A driver (the PRE-amendment driver that
# produced the preserved stage_a_report.md) calls this name.
def supplementary_rows() -> list[list[object]]:
    return extra_fixture_rows()
