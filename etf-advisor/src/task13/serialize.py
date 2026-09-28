"""Serialisation of a frozen cell into the narration prompt.

Serializer freedom is bounded by:
identity, sign, ranking by magnitude, column sums of attribution values,
differences ``(w_advisor - w_classical)``, and multiplication by 100 for
percent / percentage-point presentation.  **Nothing else is computed.** Every
serialized number is traceable to a frozen value under that list, and every
rounding here uses the SAME banker's rounding as the adjudicator, so a number
the template emits and the model copies faithfully must adjudicate PASS.

Registered display precisions:

- weights -- 1 dp percent (``times_100`` + quantize 0.1)
- forecast_vols -- 2 significant figures.  Under clarification S13-E2 the
  percent presentation is selected, explicitly labelled *daily volatility*;
  the ``times_100`` derivation is the only transformation applied.
- attributions -- 1 dp percentage points (``times_100`` + quantize 0.1)

"Largest overweight" / "largest underweight" are composed strictly from the
permitted list: rank by MAGNITUDE of the difference within the positive-sign
subset and the negative-sign subset respectively (ranking by magnitude + sign).
No ranking on signed values is performed.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

from . import PACKAGE_DIR
from .adjudicate import d_advisor, d_diff, d_phi, d_vol
from .grounding import Cell, Grounding

TEMPLATE_PATH = PACKAGE_DIR / "template.txt"


# --------------------------------------------------------------------------- #
# Formatters -- banker's rounding, identical to the adjudicator's
# --------------------------------------------------------------------------- #
def _q(x: Decimal, decimals: int) -> Decimal:
    out = x.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)
    return Decimal(0).quantize(Decimal(1).scaleb(-decimals)) if out == 0 else out


def pct1(x: Decimal) -> str:
    """Weight as percent, 1 decimal place."""
    return f"{_q(x * 100, 1)}%"


def pp1(x: Decimal) -> str:
    """Attribution as percentage points, 1 decimal place (unit word added by caller)."""
    return f"{_q(x * 100, 1)}"


def pct_2sf(x: Decimal) -> str:
    """Forecast volatility as percent, 2 significant figures."""
    y = x * 100
    if y == 0:
        return "0.0%"
    decimals = 1 - y.adjusted()          # adjusted() = exponent of the leading digit
    if decimals < 0:
        decimals = 0
    return f"{_q(y, decimals)}%"


# --------------------------------------------------------------------------- #
# Permitted derivations, named
# --------------------------------------------------------------------------- #
def largest_position(g: Grounding, c: Cell) -> str:
    vals = {t: d_advisor(g, c, t) for t in g.tickers}
    return max(sorted(vals), key=lambda t: vals[t])


def lowest_vol(g: Grounding, c: Cell) -> str:
    vals = {t: d_vol(g, c, t) for t in g.tickers}
    return min(sorted(vals), key=lambda t: vals[t])


def largest_overweight(g: Grounding, c: Cell) -> str:
    pos = [t for t in g.tickers if d_diff(g, c, t) > 0]
    return max(sorted(pos), key=lambda t: abs(d_diff(g, c, t)))


def largest_underweight(g: Grounding, c: Cell) -> str:
    neg = [t for t in g.tickers if d_diff(g, c, t) < 0]
    return max(sorted(neg), key=lambda t: abs(d_diff(g, c, t)))


def attribution_output(g: Grounding, c: Cell) -> str:
    return max(sorted(g.tickers), key=lambda t: abs(d_diff(g, c, t)))


def attribution_input(g: Grounding, c: Cell, out: str) -> str:
    return max(sorted(g.tickers), key=lambda f: abs(d_phi(g, c, f, out)))


# --------------------------------------------------------------------------- #
# Fact block and prompt
# --------------------------------------------------------------------------- #
def fact_block(g: Grounding, c: Cell) -> str:
    """Exactly the numbers the sentence plan needs.

    Withholding unused numbers is deliberate: every number placed in front of
    the model is a number it can emit, and every emitted number must be
    consumed by an extracted claim or become an unaccounted numeric.
    """
    low = lowest_vol(g, c)
    out = attribution_output(g, c)
    inp = attribution_input(g, c, out)
    weights = "\n".join(f"  {t} {pct1(d_advisor(g, c, t))}" for t in g.tickers)
    return "\n".join([
        f"date: {c.date}",
        f"profile: {c.profile}",
        "",
        "advisor allocation, percent of portfolio:",
        weights,
        "",
        f"largest position: {largest_position(g, c)}",
        "",
        f"lowest forecast daily volatility: {low} at {pct_2sf(d_vol(g, c, low))}",
        "",
        "versus the classical benchmark:",
        f"  largest overweight: {largest_overweight(g, c)}",
        f"  largest underweight: {largest_underweight(g, c)}",
        "",
        "attribution:",
        f"  attribution target ticker: {out}",
        f"  attribution source ticker: {inp}",
        f"  attribution value: {pp1(d_phi(g, c, inp, out))} percentage points",
    ])


def render_prompt(g: Grounding, c: Cell, template: str | None = None) -> str:
    tpl = template if template is not None else TEMPLATE_PATH.read_text(encoding="utf-8")
    return tpl.replace("{{FACTS}}", fact_block(g, c))
