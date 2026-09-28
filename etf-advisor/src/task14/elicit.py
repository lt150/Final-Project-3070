"""The shipped elicitor. Task 15 consumes this module.

``elicit()`` runs ONLY the arm selected at Stage C under the registered rule.
The losing arm's code remains in ``elicit_arms.py`` for the record but is not
exposed here.  Which arm won is read from ``outputs/task14/selection.json``,
the frozen selection record written by the one-shot Stage C run; it is data of
record, not a runtime knob, and this module refuses to run without it.

Output contract: the elicitor emits exactly
one registered profile string from ``PROFILES`` (src.task9.allocators) or a
clarify signal.  ``profile``, when set, is validated against ``PROFILES``
before return -- the contract check lives here; ``recommend()`` remains the
backstop, never the primary check.

Registered disclosure, carried: the selected arm is certified on the
registered personas.  Live free-text input at runtime is OUT-OF-DISTRIBUTION
relative to that evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import CLARIFY, OUT_DIR, PROFILE_ORDER, Q_KEYS
from . import elicit_arms as _arms

SELECTION_PATH = OUT_DIR / "selection.json"


@dataclass(frozen=True)
class ElicitationResult:
    """Exactly one of {profile set, clarify_question set, malformed True} holds.

    The one registered exception is Arm B signalling clarify: Arm B has no
    per-question intermediate to point at, so ``clarify_question`` is
    None there and the clarify state is carried by :attr:`clarify`.  When Arm A
    is the shipped arm the invariant holds strictly, and :func:`elicit` asserts
    it.
    """
    arm: str                       # 'A' or 'B' -- the SELECTED arm
    profile: str | None            # a PROFILES string, or None
    clarify_question: int | None   # 1..5, or None
    malformed: bool
    extraction: dict | None        # Arm A intermediate {Q1..Q5}; None for Arm B

    @property
    def clarify(self) -> bool:
        """True when the elicitor abstained rather than naming a profile."""
        return not self.malformed and self.profile is None


def selection() -> dict:
    if not SELECTION_PATH.exists():
        raise RuntimeError(
            "elicit: no selection record at "
            f"{SELECTION_PATH.as_posix()}; the Stage C one-shot evaluation "
            "has not shipped an arm yet -- STOP"
        )
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def selected_arm() -> str:
    arm = selection()["selected_arm"]
    if arm not in ("A", "B"):
        raise RuntimeError(f"elicit: selection record names arm {arm!r} -- STOP")
    return arm


def elicit(answers: dict[str, str]) -> ElicitationResult:
    """Run the selected arm once over the five registered answers."""
    missing = [q for q in Q_KEYS if q not in answers]
    if missing:
        raise ValueError(f"elicit: answers missing {missing}; the registered "
                         f"question set is {list(Q_KEYS)} -- STOP")
    arm = selected_arm()
    _res, out = _arms.run(arm, "runtime", answers, purpose="elicit:runtime")
    return _finalize(arm, out)


def _finalize(arm: str, out) -> ElicitationResult:
    """Interpretation -> the registered result shape, with the contract check."""
    if out.malformed:
        return ElicitationResult(arm, None, None, True, None)

    profile = out.profile
    if profile is not None and profile not in PROFILE_ORDER:
        # The contract check lives here.  An off-contract string never leaves
        # this function, so recommend() is never asked to be the primary check.
        raise RuntimeError(
            f"elicit: arm {arm} produced profile {profile!r}, which is not in "
            f"PROFILES {PROFILE_ORDER} -- STOP"
        )

    result = ElicitationResult(arm, profile, out.clarify_question, False,
                               out.extraction)
    if arm == "A":
        exactly_one = ((result.profile is not None)
                       + (result.clarify_question is not None)
                       + result.malformed)
        if exactly_one != 1:
            raise RuntimeError(
                f"elicit: Arm A result violates the registered invariant "
                f"(profile={result.profile!r}, "
                f"clarify_question={result.clarify_question!r}, "
                f"malformed={result.malformed!r}) -- STOP"
            )
    return result


def interpret_only(raw: str) -> ElicitationResult:
    """Parse/route a stored transcript through the selected arm, no generation.

    Used by the Stage C Gate C3 check so the shipped interface can be exercised
    without spending a call, and available to Task 15 for replay.
    """
    arm = selected_arm()
    return _finalize(arm, _arms.INTERPRET[arm]("replay", raw))


__all__ = ["ElicitationResult", "elicit", "interpret_only", "selected_arm",
           "selection", "CLARIFY"]
