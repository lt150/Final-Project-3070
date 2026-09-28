"""Exact interventional Shapley by full coalition enumeration.

New arithmetic, registered: the coalition weights, the hybrid construction, the
enumeration, and the attribution sum. Nothing in this module knows what the
explained function is; Stages A-C supply it.

For a feature set N = {0..n-1}, an explicand x, a background b and a
vector-valued f, the hybrid at coalition S is h_i = x_i for i in S and h_i = b_i
otherwise, and v(S) = f(h(S)). All 2^n coalitions are evaluated ONCE each,
cached by bitmask (bit i <-> feature i, LSB = feature 0); the cached vectors are
the sole numerical inputs to the attribution arithmetic:

    phi[i, j] = sum over S subseteq N \\ {i} of
                ( |S|! (n - |S| - 1)! / n! ) ( v_j(S + {i}) - v_j(S) )

Two properties are used as gates elsewhere and are computed here, never
enforced here: v(N) and v(empty) are exposed unchanged, and the efficiency
residual max_j |sum_i phi[i, j] - (v_j(N) - v_j(empty))| is reported by the
caller against its own registered tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Callable, Sequence

import numpy as np


def shapley_weights(n: int) -> tuple[float, ...]:
    """``w[s] = s! (n - s - 1)! / n!`` -- the kernel for a coalition of size s.

    Factorials are exact integers; a single float division per size.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    denom = factorial(n)
    return tuple(factorial(s) * factorial(n - s - 1) / denom for s in range(n))


def hybrid(mask: int, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``h_i = x_i`` if bit i of ``mask`` is set, else ``b_i``.

    Selection only: every entry of the result is a bit-identical copy of an
    entry of x or of b. No arithmetic is performed on the inputs.
    """
    n = x.shape[0]
    present = np.fromiter(((mask >> i) & 1 for i in range(n)), dtype=bool, count=n)
    return np.where(present, x, b)


@dataclass(frozen=True)
class Enumeration:
    """One completed 2^n enumeration and the attributions read off from it."""

    x: np.ndarray            # explicand, shape (n,)
    b: np.ndarray            # background, shape (n,)
    values: np.ndarray       # shape (2^n, m); row index IS the coalition bitmask
    phi: np.ndarray          # shape (n, m)

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def m(self) -> int:
        return int(self.values.shape[1])

    @property
    def v_full(self) -> np.ndarray:
        """v(N) -- every feature taken from the explicand."""
        return self.values[(1 << self.n) - 1]

    @property
    def v_empty(self) -> np.ndarray:
        """v(empty) -- every feature taken from the background."""
        return self.values[0]

    @property
    def column_sums(self) -> np.ndarray:
        """sum_i phi[i, j], one per output."""
        return self.phi.sum(axis=0)

    def efficiency_residuals(self) -> np.ndarray:
        """|sum_i phi[i, j] - (v_j(N) - v_j(empty))| per output j."""
        return np.abs(self.column_sums - (self.v_full - self.v_empty))

    def efficiency_residual(self) -> float:
        """The worst per-output efficiency residual."""
        return float(self.efficiency_residuals().max())


def shapley_from_values(values: np.ndarray, n: int) -> np.ndarray:
    """Attribution matrix from the cached coalition values. Fixed sum order."""
    if values.shape[0] != 1 << n:
        raise ValueError(f"expected {1 << n} coalition rows, got {values.shape[0]}")
    weights = shapley_weights(n)
    m = values.shape[1]
    phi = np.zeros((n, m), dtype=float)
    for i in range(n):
        bit = 1 << i
        acc = np.zeros(m, dtype=float)
        for mask in range(1 << n):
            if mask & bit:
                continue
            acc += weights[mask.bit_count()] * (values[mask | bit] - values[mask])
        phi[i] = acc
    return phi


def enumerate_exact(
    f: Callable[[np.ndarray], Sequence[float]],
    x: Sequence[float],
    b: Sequence[float],
) -> Enumeration:
    """Evaluate all 2^n coalitions once each, then read off the attributions."""
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)
    if x.ndim != 1 or b.shape != x.shape:
        raise ValueError(f"x {x.shape} and b {b.shape} must be equal-length 1-D")
    n = int(x.shape[0])

    cache: list[np.ndarray] = []
    for mask in range(1 << n):
        out = np.asarray(f(hybrid(mask, x, b)), dtype=float)
        if out.ndim != 1:
            raise ValueError(f"f must return a 1-D vector, got shape {out.shape}")
        cache.append(out)

    values = np.vstack(cache)
    return Enumeration(x=x, b=b, values=values, phi=shapley_from_values(values, n))
