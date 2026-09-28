"""The registered deterministic rubric -- Arm A's decider.

Stated twice in the register (words, then mapping) per the double-statement
convention.  Implemented here ONCE, from the mapping block, and validated
against fixtures R1-R9 before any LLM call exists.

Mapping block, verbatim:

    points: Q1 short=0 medium=1 long=2 | Q2 sell=0 hold=1 buy_more=2
            Q3 preserve=0 balanced=1 grow=2 | Q4 low=0 medium=1 high=2
            Q5 none=0 some=1 experienced=2
    score = sum(points)                     # 0..10
    score_profile = conservative if score <= 3
                    balanced     if 4 <= score <= 7
                    growth       if score >= 8
    cap = conservative if Q1 == short
          balanced     if Q1 == medium or Q4 == low
          growth       otherwise
    profile = min(score_profile, cap)       # conservative < balanced < growth

Routing: *Any ``unclear`` in Arm A's extraction routes the
persona to ``clarify(Qk)`` (first unclear question).*  So the rubric is applied
only to extractions with no ``unclear``; the routing decision is made first.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import CLARIFY, DOMAINS, PROFILE_ORDER, Q_KEYS, UNCLEAR

# points = the index of the answer inside its registered domain listing.  The
# register states the rule as "each answer scores 0, 1, or 2 in the order its
# domain is listed", so the point table is DERIVED from DOMAINS rather than
# retyped -- a transcription slip in the ordering cannot hide here.
POINTS: dict[str, dict[str, int]] = {
    q: {value: i for i, value in enumerate(DOMAINS[q])} for q in Q_KEYS
}

CONSERVATIVE, BALANCED, GROWTH = PROFILE_ORDER
_RANK = {p: i for i, p in enumerate(PROFILE_ORDER)}   # conservative < balanced < growth


@dataclass(frozen=True)
class RubricTrace:
    """Every intermediate the register names, kept for the audit trail."""
    per_question: tuple[int, ...]
    score: int
    score_profile: str
    cap: str
    profile: str

    def as_text(self) -> str:
        pts = "+".join(str(p) for p in self.per_question)
        return (f"{pts}={self.score}; score_profile={self.score_profile}; "
                f"cap={self.cap}; min -> {self.profile}")


def score_of(extraction: dict[str, str]) -> tuple[tuple[int, ...], int]:
    per = tuple(POINTS[q][extraction[q]] for q in Q_KEYS)
    return per, sum(per)


def score_profile_of(score: int) -> str:
    if score <= 3:
        return CONSERVATIVE
    if 4 <= score <= 7:
        return BALANCED
    return GROWTH                     # score >= 8


def cap_of(extraction: dict[str, str]) -> str:
    if extraction["Q1"] == "short":
        return CONSERVATIVE
    if extraction["Q1"] == "medium" or extraction["Q4"] == "low":
        return BALANCED
    return GROWTH


def _min_profile(a: str, b: str) -> str:
    return a if _RANK[a] <= _RANK[b] else b


def apply(extraction: dict[str, str]) -> RubricTrace:
    """The registered mapping.  Requires a fully-scored extraction (no unclear)."""
    missing = [q for q in Q_KEYS if q not in extraction]
    if missing:
        raise ValueError(f"rubric.apply: extraction missing {missing} -- STOP")
    bad = [(q, extraction[q]) for q in Q_KEYS if extraction[q] not in POINTS[q]]
    if bad:
        raise ValueError(
            f"rubric.apply: unscoreable answers {bad}; the rubric is only "
            "applied to extractions with no 'unclear' -- STOP"
        )
    per, score = score_of(extraction)
    sp = score_profile_of(score)
    cap = cap_of(extraction)
    return RubricTrace(per, score, sp, cap, _min_profile(sp, cap))


def first_unclear(extraction: dict[str, str]) -> int | None:
    """Index (1..5) of the first ``unclear`` answer in registered question order."""
    for i, q in enumerate(Q_KEYS, start=1):
        if extraction.get(q) == UNCLEAR:
            return i
    return None


@dataclass(frozen=True)
class Routing:
    """Arm A's decision: exactly one of ``profile`` / ``clarify_question``."""
    profile: str | None
    clarify_question: int | None
    trace: RubricTrace | None

    @property
    def outcome(self) -> str:
        """The comparable outcome token: a PROFILES string or ``clarify``."""
        return self.profile if self.profile is not None else CLARIFY


def route(extraction: dict[str, str]) -> Routing:
    """Register Sec. 1 routing, then the Sec. 2 rubric."""
    k = first_unclear(extraction)
    if k is not None:
        return Routing(None, k, None)
    trace = apply(extraction)
    return Routing(trace.profile, None, trace)


# --------------------------------------------------------------------------- #
# Registered fixtures R1-R9, transcribed verbatim.
# Cross-checked against the register text by stage_a.py before they are used.
# --------------------------------------------------------------------------- #
# (id, answers Q1..Q5, score, score_profile, cap, expected profile)
R_FIXTURES: tuple[tuple[str, tuple[str, ...], int, str, str, str], ...] = (
    ("R1", ("short", "sell", "preserve", "low", "none"),
     0, "conservative", "conservative", "conservative"),
    ("R2", ("long", "buy_more", "grow", "high", "experienced"),
     10, "growth", "growth", "growth"),
    ("R3", ("medium", "hold", "preserve", "medium", "none"),
     3, "conservative", "balanced", "conservative"),
    ("R4", ("medium", "hold", "preserve", "medium", "some"),
     4, "balanced", "balanced", "balanced"),
    ("R5", ("long", "hold", "balanced", "high", "some"),
     7, "balanced", "growth", "balanced"),
    ("R6", ("long", "buy_more", "balanced", "high", "some"),
     8, "growth", "growth", "growth"),
    ("R7", ("short", "buy_more", "grow", "high", "experienced"),
     8, "growth", "conservative", "conservative"),
    ("R8", ("long", "buy_more", "grow", "low", "experienced"),
     8, "growth", "balanced", "balanced"),
    ("R9", ("medium", "buy_more", "grow", "high", "experienced"),
     9, "growth", "balanced", "balanced"),
)

# The register's own note on what the fixtures exercise.
R_FIXTURE_COVERAGE = (
    "R3/R4 exercise the 3/4 threshold; R5/R6 the 7/8 threshold; R7-R9 the caps."
)


def fixture_rows() -> list[list[object]]:
    """Run R1-R9.  Returns report rows; every registered column is compared."""
    rows: list[list[object]] = []
    for rid, answers, score, sp, cap, expected in R_FIXTURES:
        extraction = dict(zip(Q_KEYS, answers))
        t = apply(extraction)
        ok = (t.score == score and t.score_profile == sp
              and t.cap == cap and t.profile == expected)
        rows.append([
            rid, ", ".join(answers),
            f"{score} / {t.score}",
            f"{sp} / {t.score_profile}",
            f"{cap} / {t.cap}",
            f"{expected} / {t.profile}",
            "PASS" if ok else "**FAIL**",
        ])
    return rows


def fixtures_pass() -> bool:
    return all(r[-1] == "PASS" for r in fixture_rows())
