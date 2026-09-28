"""Test-only parallel covariance path with an explicit correlation window.

Stage D variants V1 (R = 60) and V2 (R = 120) need a correlation window other
than 250. The 250-day window is a module constant inside `src/task9/covariance.py`
and `src/task9/` is read-only — it is deliberately *not* a parameter of
any public Task 9 function. This module is the parallel implementation 
and it exists for Stage D only.

**What is re-implemented, and what is not.** Only the thing that must change is
new code:

  * ``corr_matrix_w(t, window)`` — Pearson correlation of daily simple returns
    over the ``window`` trading days ending at and including t, full 8-ticker
    panel, no pairwise exclusions. This mirrors `task9.covariance.corr_matrix`
    with the window lifted to a parameter.

  * ``sigma_w(t, variant, window)`` — Σ = D·R·D assembled as R_ij·d_i·d_j,
    exactly as Task 9 assembles it. **D is not re-implemented**: the vectors
    come from the registered `task9.forecaster_deploy.forecast_vol` (advisor)
    and `vol_trail_20` (classical), called directly.

  * The solver is **not** re-implemented either. `task9.allocators.solve_risk_budget`
    takes Σ as an argument, so the registered objective, gradient, bounds,
    gtol/ftol/maxiter and the Gate C1 assertion are reached by direct call.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..data.config import UNIVERSE
from ..task9.allocators import RISK_BUDGETS, solve_risk_budget
from ..task9.covariance import CORR_WINDOW
from ..task9.forecaster_deploy import _returns, forecast_vol, vol_trail_20

HEADLINE_WINDOW = CORR_WINDOW

SENSITIVITY_WINDOWS = (60, 120)


def corr_matrix_w(t: pd.Timestamp, window: int) -> pd.DataFrame:
    """Pearson correlation over the ``window`` trading days ending at t."""
    t = pd.Timestamp(t)
    ret = _returns()
    if t not in ret.index:
        raise ValueError(f"{t.date()} is not a trading day on the panel index")
    w = ret.loc[:t].tail(window)
    if len(w) != window or w.isna().any().any():
        raise ValueError(f"insufficient return history for R({window}) at {t.date()}")
    return w.corr()


def sigma_w(
    t: pd.Timestamp,
    variant: Literal["advisor", "classical"],
    window: int,
) -> pd.DataFrame:
    """Σ = D·R·D in raw daily variance, with R on an explicit window."""
    t = pd.Timestamp(t)
    if variant == "advisor":
        d = forecast_vol(t)
    elif variant == "classical":
        d = vol_trail_20(t)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    R = corr_matrix_w(t, window)
    d = d.reindex(R.index)
    assert not d.isna().any(), "vol vector does not cover the panel tickers"
    return R * np.outer(d.to_numpy(), d.to_numpy())


def rb_weight_fn(variant: str, profile: str, window: int):
    """Weight function for a risk-budget allocator on an overridden window."""
    budgets = RISK_BUDGETS[profile]
    return lambda t: solve_risk_budget(sigma_w(pd.Timestamp(t), variant, window),
                                       budgets)


def one_over_n_weight_fn(*_args, **_kwargs):
    """1/N is Σ-free"""
    return lambda t: pd.Series(1.0 / len(UNIVERSE), index=list(UNIVERSE))
