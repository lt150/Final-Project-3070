"""Both elicitor arms: rendering, generation, parsing, and the arm outcome.

- **Arm A (hybrid)**: free text -> LLM extracts structured answers only ->
  deterministic rubric maps answers -> profile.  The LLM parses; the rubric
  decides.
- **Arm B (direct)**: free text -> LLM maps directly to a profile.

Both arms are kept here for the record after selection; ``elicit.py`` exposes
only the SELECTED arm.

Prompt discipline:  Arm A's prompt presents the five registered
questions verbatim with the persona's answers, instructs extraction to the
registered five-line format, instructs ``unclear`` whenever the text does not
support a registered value, and contains NO profile semantics and NO mapping
guidance -- the rubric decides.  Arm B's prompt presents the same questions and
answers and instructs exactly one word from the registered vocabulary; its
content MAY describe profile semantics, which is Arm B's registered design
freedom.  :func:`template_issues` enforces the shared half of that mechanically
(questions verbatim, placeholders present) and enforces on Arm A that no
profile name appears outside the registered Q3 domain value ``balanced``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import (
    ARM_B_VOCAB,
    CLARIFY,
    PACKAGE_DIR,
    PROFILE_ORDER,
    QUESTIONS,
    Q_KEYS,
    generate,
    sha256_file,
)
from . import parse as P
from . import rubric as R

PROMPT_PATHS = {
    "A": PACKAGE_DIR / "prompt_arm_a.txt",
    "B": PACKAGE_DIR / "prompt_arm_b.txt",
}
PLACEHOLDERS = tuple(f"{{{{A{i}}}}}" for i in range(1, 6))   # {{A1}} .. {{A5}}


def load_template(arm: str) -> str:
    return PROMPT_PATHS[arm].read_text(encoding="utf-8")


def prompt_sha(arm: str) -> str:
    return sha256_file(PROMPT_PATHS[arm])


def render(arm: str, answers: dict[str, str]) -> str:
    """Substitute the five answers into the arm's template.

    Substitution is positional and total: every placeholder must be present and
    every answer must be consumed, so a template edit cannot silently drop a
    question.
    """
    text = load_template(arm)
    for i, key in enumerate(Q_KEYS):
        ph = PLACEHOLDERS[i]
        if ph not in text:
            raise ValueError(f"arm {arm} template is missing {ph} -- STOP")
        text = text.replace(ph, answers[key])
    left = [ph for ph in PLACEHOLDERS if ph in text]
    if left:
        raise ValueError(f"arm {arm} template still holds {left} -- STOP")
    return text


def template_issues(arm: str) -> list[str]:
    """Mechanical checks on a template, run before any generation."""
    text = load_template(arm)
    issues: list[str] = []
    for i, key in enumerate(Q_KEYS):
        if QUESTIONS[key] not in text:
            issues.append(f"arm {arm}: registered question {key} not present "
                          "verbatim")
        if PLACEHOLDERS[i] not in text:
            issues.append(f"arm {arm}: placeholder {PLACEHOLDERS[i]} missing")
    if arm == "A":
        # No profile semantics on Arm A.  'balanced' is exempt: it is a
        # registered Q3 domain value, not a profile reference here.
        low = text.lower()
        for name in PROFILE_ORDER:
            if name == "balanced":
                continue
            if name in low:
                issues.append(f"arm A: profile name {name!r} appears in the "
                              "prompt; Arm A carries no profile semantics")
        if CLARIFY in low:
            issues.append("arm A: 'clarify' appears in the prompt; Arm A routes "
                          "to clarify through the rubric, not the model")
    if arm == "B":
        for word in ARM_B_VOCAB:
            if word not in text:
                issues.append(f"arm B: vocabulary word {word!r} not offered in "
                              "the prompt")
    return issues


# --------------------------------------------------------------------------- #
# Arm results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArmOutcome:
    """One arm's result for one persona, on one generation.

    ``outcome`` is the comparable token: a PROFILES string, ``clarify``, or
    ``MALFORMED``.  Exactly one of {profile, clarify_question, malformed}.
    """
    arm: str
    persona: str
    raw: str
    malformed: bool
    profile: str | None
    clarify_question: int | None
    extraction: dict[str, str] | None
    rubric_trace: str
    parse_reason: str

    @property
    def outcome(self) -> str:
        if self.malformed:
            return P.MALFORMED
        return self.profile if self.profile is not None else CLARIFY


def interpret_arm_a(persona: str, raw: str) -> ArmOutcome:
    """Parse -> (rubric | clarify routing).  No generation."""
    parsed = P.parse_arm_a(raw)
    if not parsed.valid:
        return ArmOutcome("A", persona, raw, True, None, None, None, "-",
                          parsed.reason)
    routing = R.route(parsed.extraction)
    return ArmOutcome(
        "A", persona, raw, False, routing.profile, routing.clarify_question,
        parsed.extraction,
        routing.trace.as_text() if routing.trace else
        f"unclear at Q{routing.clarify_question} -> clarify",
        parsed.reason)


def interpret_arm_b(persona: str, raw: str) -> ArmOutcome:
    """Parse only.  No intermediate exists, by construction."""
    parsed = P.parse_arm_b(raw)
    if not parsed.valid:
        return ArmOutcome("B", persona, raw, True, None, None, None, "-",
                          parsed.reason)
    if parsed.value == CLARIFY:
        # Arm B's clarify carries no question index: it has no per-question
        # intermediate to point at.  Recorded as the registered asymmetry.
        return ArmOutcome("B", persona, raw, False, None, None, None, "-",
                          parsed.reason)
    return ArmOutcome("B", persona, raw, False, parsed.value, None, None, "-",
                      parsed.reason)


INTERPRET = {"A": interpret_arm_a, "B": interpret_arm_b}


def run(arm: str, persona: str, answers: dict[str, str], *, purpose: str):
    """Render, generate once, interpret.  Returns (GenerationResult, ArmOutcome)."""
    prompt = render(arm, answers)
    res = generate(prompt, purpose=purpose)
    return res, INTERPRET[arm](persona, res.response)


def write_pair(directory: Path, stem: str, prompt: str, response: str) -> None:
    """Persist the exact prompt sent and the exact response received."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}_prompt.txt").write_text(prompt, encoding="utf-8")
    (directory / f"{stem}_response.txt").write_text(response, encoding="utf-8")
