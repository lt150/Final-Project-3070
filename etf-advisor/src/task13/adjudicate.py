"""Adjudication of extracted claims against the frozen artifacts.

The numeric matching rule:

    A numeric claim PASSES iff the frozen full-precision value, rounded to the
    precision stated in the claim text, equals the claimed value. Rounding is
    banker's rounding via ``decimal.Decimal.quantize(..., ROUND_HALF_EVEN)``
    applied to the full-precision value in the claim's stated units (percent
    for "x%", percentage points for "x percentage points" = weight-units x 100,
    raw units otherwise). Example: frozen weight 0.337415; claim "33.7%" (1 dp
    percent) -> quantize(33.7415, 0.1) = 33.7 -> PASS; claim "34%" (0 dp) ->
    quantize -> 34 -> PASS; claim "33.8%" -> FAIL.

The same rule in code (second statement): see :func:`numeric_matches`.
The two statements are kept adjacent deliberately -- this is the
leakage-sensitive rule of the task and the convention requires it stated twice.

Full precision is preserved end to end: frozen values enter ``Decimal`` from
their exact CSV token (Task 10 and Task 12 recorded as ``repr()``, so the
token is the value), and every permitted derivation -- differences, column
sums, the x100 to percent/percentage points -- is carried out in ``Decimal``
arithmetic on those tokens rather than in binary floating point.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

from . import REPO_ROOT
from .extract import Claim, Extraction
from .grounding import Cell, Grounding

DERIVATIONS = (
    "identity", "sign", "ranking_by_magnitude", "column_sum", "difference",
    "times_100",
)


def numeric_matches(frozen: Decimal, claimed: str, precision: str) -> bool:
    """The registered matching rule, in code (second statement)."""
    unit, decimals = precision.split(".")
    scaled = frozen * 100 if unit in ("pct", "pp") else frozen
    quantum = Decimal(1).scaleb(-int(decimals))
    return scaled.quantize(quantum, rounding=ROUND_HALF_EVEN) == Decimal(claimed)


def frozen_covariance_source() -> str:
    """The COVARIANCE_SOURCE literal, parsed out of the frozen Task 10 module.
    """
    src = (REPO_ROOT / "src" / "task10" / "recommend.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "COVARIANCE_SOURCE":
                    return ast.literal_eval(node.value)
    raise RuntimeError("COVARIANCE_SOURCE not found in src/task10/recommend.py -- STOP")


# --------------------------------------------------------------------------- #
# Decimal views of the frozen surfaces
# --------------------------------------------------------------------------- #
def d_advisor(g: Grounding, c: Cell, t: str) -> Decimal:
    return Decimal(g.advisor_w_raw[(c.date, c.profile)][t])


def d_classical(g: Grounding, c: Cell, t: str) -> Decimal:
    return Decimal(g.classical_w_raw[(c.date, c.profile)][t])


def d_vol(g: Grounding, c: Cell, t: str) -> Decimal:
    return Decimal(g.forecast_vols_raw[(c.date, c.profile)][t])


def d_phi(g: Grounding, c: Cell, f: str, o: str) -> Decimal:
    return Decimal(g.phi_raw[(c.date, c.profile)][(f, o)])


def d_diff(g: Grounding, c: Cell, t: str) -> Decimal:
    """Permitted derivation: difference (w_advisor - w_classical), in Decimal."""
    return d_advisor(g, c, t) - d_classical(g, c, t)


def d_colsum(g: Grounding, c: Cell, o: str) -> Decimal:
    """Permitted derivation: column sum of attribution values, in Decimal."""
    return sum((d_phi(g, c, f, o) for f in g.tickers), Decimal(0))


def _extremes(values: dict[str, Decimal], direction: str) -> list[str]:
    """Argmax / argmin, returning EVERY tied extreme (ties pass on membership)."""
    target = max(values.values()) if direction == "max" else min(values.values())
    return sorted(k for k, v in values.items() if v == target)


# --------------------------------------------------------------------------- #
# Adjudication
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    claim: Claim
    passed: bool
    reference_repr: str
    derivation: str
    note: str


HARD_FAIL_TYPES = ("T7",)


def adjudicate(ex: Extraction, g: Grounding, c: Cell) -> list[Verdict]:
    out: list[Verdict] = []
    cov = frozen_covariance_source()
    for cl in ex.claims:
        out.append(_one(cl, ex, g, c, cov))
    return out


def _one(cl: Claim, ex: Extraction, g: Grounding, c: Cell, cov: str) -> Verdict:
    T = g.tickers

    if cl.ctype == "T1":
        if cl.subject not in T:
            return Verdict(cl, False, "-", "identity",
                           f"subject {cl.subject!r} is not a universe ticker")
        ref = d_advisor(g, c, cl.subject)
        ok = numeric_matches(ref, cl.value, cl.precision)
        return Verdict(cl, ok, str(ref), "identity+times_100",
                       f"frozen advisor weight for {cl.subject}")

    if cl.ctype == "T2":
        if cl.subject not in T:
            return Verdict(cl, False, "-", "identity",
                           f"subject {cl.subject!r} is not a universe ticker")
        ref = d_vol(g, c, cl.subject)
        ok = numeric_matches(ref, cl.value, cl.precision)
        return Verdict(cl, ok, str(ref), "identity+times_100",
                       f"frozen forecast_vol for {cl.subject} (raw daily units)")

    if cl.ctype == "T3":
        if cl.subject not in T or cl.basis not in T:
            return Verdict(cl, False, "-", "identity",
                           f"input {cl.subject!r} / output {cl.basis!r} not both "
                           "universe tickers")
        ref = d_phi(g, c, cl.subject, cl.basis)
        ok = numeric_matches(ref, cl.value, cl.precision)
        return Verdict(cl, ok, str(ref), "identity+times_100",
                       f"phi[feature={cl.subject}, output={cl.basis}]")

    if cl.ctype == "T4":
        if cl.subject not in T:
            return Verdict(cl, False, "-", "sign",
                           f"subject {cl.subject!r} is not a universe ticker")
        if cl.basis == "attribution_sign":
            ref, deriv = d_colsum(g, c, cl.subject), "column_sum+sign"
        else:
            ref, deriv = d_diff(g, c, cl.subject), "difference+sign"
        sign = "positive" if ref > 0 else ("negative" if ref < 0 else "zero")
        return Verdict(cl, sign == cl.value, str(ref), deriv,
                       f"sign({cl.basis}[{cl.subject}]) = {sign}")

    if cl.ctype == "T5":
        qset = cl.value
        direction = "max" if any(cl.basis.startswith(w) for w in
                                 ("largest", "biggest", "greatest", "highest",
                                  "top", "most")) else "min"
        if qset == "advisor_weights":
            values = {t: d_advisor(g, c, t) for t in T}
            deriv = "ranking_by_magnitude"
        elif qset == "forecast_vols":
            values = {t: d_vol(g, c, t) for t in T}
            deriv = "ranking_by_magnitude"
        elif qset.startswith("attributions[output="):
            o = qset[len("attributions[output="):-1]
            if o not in T:
                return Verdict(cl, False, "-", "ranking_by_magnitude",
                               f"unresolved attribution output {o!r}")
            values = {t: abs(d_phi(g, c, t, o)) for t in T}
            deriv = "ranking_by_magnitude (absolute value)"
        else:
            return Verdict(cl, False, "-", "ranking_by_magnitude",
                           f"unrecognised quantity set {qset!r}")
        winners = _extremes(values, direction)
        return Verdict(cl, cl.subject in winners,
                       "{" + ", ".join(f"{k}={v}" for k, v in values.items()) + "}",
                       deriv, f"{direction} of {qset} = {winners}")

    if cl.ctype == "T6":
        if cl.subject == "profile":
            return Verdict(cl, cl.value == c.profile, c.profile, "identity",
                           "cell profile")
        if cl.subject == "date":
            return Verdict(cl, cl.value == c.date, c.date, "identity", "cell date")
        ok = cov in cl.value
        return Verdict(cl, ok, cov, "identity",
                       "sentence must contain the frozen COVARIANCE_SOURCE "
                       "literal verbatim; any other description of the "
                       "covariance construction fails")

    if cl.ctype == "T7":
        return Verdict(cl, False, "-", "n/a",
                       f"FABRICATED: {cl.basis} {cl.subject!r} has no artifact "
                       "referent")

    return Verdict(cl, False, "-", "n/a", f"unrecognised claim type {cl.ctype!r}")


# --------------------------------------------------------------------------- #
# Unaccounted-numeric rule, in code
# --------------------------------------------------------------------------- #
def unaccounted_numerics(ex: Extraction, c: Cell) -> list[tuple[int, int, str, int, str]]:
    """Every numeric token not consumed by exactly one claim, minus the whitelist.

    The registered whitelist is the CELL DATE TOKENS ONLY.  Because every date
    token is consumed by a T6 date claim (and therefore adjudicated against the
    cell date rather than waved through), the whitelist is a belt-and-braces
    fallback: a date token that somehow escaped consumption is still checked
    against the cell date here, and a WRONG date is reported as unaccounted
    rather than silently forgiven.
    """
    rows = []
    for start, end, token, sidx in ex.unaccounted:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token) and token == c.date:
            continue
        rows.append((start, end, token, sidx, "T7 hard-fail path"))
    return rows
