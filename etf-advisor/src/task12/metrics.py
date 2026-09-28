"""Stability arithmetic. New numerics, registered.

    dphi = phi_perturbed - phi_unperturbed          (attribution movement)
    dw   = v_perturbed(N) - v_unperturbed(N)        (output movement)

    rho = max_j max(|dphi_A,j|, |dphi_B,j|) / max(max_j |dw_j|, 1e-6)

per (cell, probe) and pair (A, B). The denominator floor is the solver band: it
prevents division by solver noise while ensuring that the worst instability
signature -- attributions moving above the band while the output barely moves --
registers as a LARGE rho, never as a degenerate exclusion. No cell or probe is
ever excluded from rho_max.

dphi and dw are MEASUREMENTS, not gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixtures import RHO_DENOM_FLOOR


@dataclass(frozen=True)
class Stability:
    """One (cell, probe, pair) stability measurement, full precision."""

    dphi_pair_max: float     # max_j max(|dphi_A,j|, |dphi_B,j|)
    dw_max: float            # max_j |dw_j|
    rho: float
    dphi_pair_sum_max: float  # max_j |d(phi_A,j + phi_B,j)| -- soft observation


def stability(
    phi_base: np.ndarray,
    phi_probe: np.ndarray,
    w_base: np.ndarray,
    w_probe: np.ndarray,
    pair_index: tuple[int, int],
    floor: float = RHO_DENOM_FLOOR,
) -> Stability:
    """rho and its two components for one pair under one probe."""
    dphi = phi_probe - phi_base
    dw = w_probe - w_base
    rows = np.array(pair_index, dtype=int)
    dphi_pair_max = float(np.abs(dphi[rows, :]).max())
    dw_max = float(np.abs(dw).max())
    rho = dphi_pair_max / max(dw_max, floor)
    pair_sum_max = float(np.abs(dphi[rows[0], :] + dphi[rows[1], :]).max())
    return Stability(
        dphi_pair_max=dphi_pair_max,
        dw_max=dw_max,
        rho=rho,
        dphi_pair_sum_max=pair_sum_max,
    )
