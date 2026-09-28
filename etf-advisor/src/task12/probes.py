"""Registered stability probes.

The explicand only is perturbed; the background is a constant of the method and
is never touched. eps = 0.01, multiplicative.

Monitored pair (SPY, QQQ) -- gated:
    p1: SPY x(1+eps)
    p2: QQQ x(1+eps)
    p3: SPY x(1+eps), QQQ x(1-eps)
    p4: SPY x(1-eps), QQQ x(1+eps)

Control pair (SPY, AGG) -- descriptive comparator, never gated:
    p1 (reused), k1: AGG x(1+eps), k2: SPY x(1+eps), AGG x(1-eps),
    k3: SPY x(1-eps), AGG x(1+eps)

p1 belongs to both probe lists; its enumeration is computed once per cell and
its cached values feed both pairs' metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .fixtures import EPS, TICKERS


@dataclass(frozen=True)
class Probe:
    """A registered perturbation: a name and its multiplicative legs."""

    name: str
    legs: tuple[tuple[str, int], ...]   # (ticker, sign), sign in {+1, -1}

    def describe(self, eps: float = EPS) -> str:
        return ", ".join(
            f"{tk} x(1{'+' if s > 0 else '-'}{eps:g})" for tk, s in self.legs
        )


P1 = Probe("p1", (("SPY", +1),))

MONITORED_PROBES: tuple[Probe, ...] = (
    P1,
    Probe("p2", (("QQQ", +1),)),
    Probe("p3", (("SPY", +1), ("QQQ", -1))),
    Probe("p4", (("SPY", -1), ("QQQ", +1))),
)

CONTROL_PROBES: tuple[Probe, ...] = (
    P1,
    Probe("k1", (("AGG", +1),)),
    Probe("k2", (("SPY", +1), ("AGG", -1))),
    Probe("k3", (("SPY", -1), ("AGG", +1))),
)

# The distinct enumerations a cell needs: p1 appears once, not twice.
ALL_PROBES: tuple[Probe, ...] = MONITORED_PROBES + tuple(
    p for p in CONTROL_PROBES if p.name != P1.name
)


def perturb(x: Sequence[float], probe: Probe, eps: float = EPS) -> np.ndarray:
    """Apply a probe to the explicand. Multiplicative, one factor per leg."""
    out = np.array(x, dtype=float, copy=True)
    for ticker, sign in probe.legs:
        i = TICKERS.index(ticker)
        out[i] = out[i] * (1.0 + sign * eps)
    return out
