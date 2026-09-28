"""Independent reconciliation path for the Task 11 engine.

Two implementations of the same accounting, compared: the
engine's units formulation (``engine.run_backtest``) and a gross-return
formulation of drift with direct value recomputation.

The two routes share no arithmetic:

  * ``engine`` holds fixed unit counts and marks them at adjusted closes; it
    obtains drifted weights from a single two-point price ratio
    P[t_k] / P[t_prev].
  * ``reconcile`` never forms units or touches a price level. It compounds the
    portfolio's daily simple return day by day,
    V_d = V_{d-1} * (1 + w_{d-1} . r_d), and evolves the weight vector one day
    at a time, w_d = w_{d-1} * (1 + r_d) / (1 + w_{d-1} . r_d), so the drifted
    weights at a grid date are the product of ~21 daily steps rather than one
    ratio.

They agree analytically; disagreement beyond float accumulation means one of
them is wrong. Gate B2 is the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import COST_RATE, V_CASH, WeightFn


@dataclass(frozen=True)
class ReconResult:
    values: pd.Series
    v_pre: pd.Series
    v_post: pd.Series
    drift_weights: pd.DataFrame
    target_weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


def reconcile_backtest(
    weight_fn: WeightFn,
    returns: pd.DataFrame,
    grid: pd.DatetimeIndex,
    days: pd.DatetimeIndex,
    cost_rate: float = COST_RATE,
) -> ReconResult:
    """Daily-compounding replica of ``engine.run_backtest``.

    ``days`` is the trading-day span (taken from the price index by the caller,
    so both paths cover exactly the same dates); ``returns`` supplies daily
    simple returns on those days.
    """
    days = pd.DatetimeIndex(days)
    tickers = list(returns.columns)
    grid = pd.DatetimeIndex(grid)
    on_grid = set(grid)

    if days[0] not in on_grid:
        raise ValueError("first trading day of the span must be the inception grid date")

    values = pd.Series(np.nan, index=days, dtype=float, name="value")
    v_pre_rec, v_post_rec, turn_rec, cost_rec = {}, {}, {}, {}
    drift_rec, tgt_rec = {}, {}

    v = float(V_CASH)
    w = pd.Series(0.0, index=tickers)   # cash before the inception trade

    for i, d in enumerate(days):
        if i > 0:
            r = returns.loc[d].reindex(tickers)
            if r.isna().any():
                raise ValueError(f"missing daily returns at {d.date()}")
            growth = 1.0 + float(w @ r)
            v = v * growth
            w = w * (1.0 + r) / growth      # one-day weight drift, renormalised

        if d in on_grid:
            v_pre_rec[d] = v
            drift_rec[d] = w.copy()
            w_tgt = pd.Series(weight_fn(d), dtype=float).reindex(tickers)
            turnover = float((w_tgt - w).abs().sum())
            v = v * (1.0 - cost_rate * turnover)
            w = w_tgt
            tgt_rec[d], turn_rec[d] = w_tgt, turnover
            cost_rec[d], v_post_rec[d] = cost_rate * turnover, v

        values.loc[d] = v

    gidx = pd.DatetimeIndex(grid, name="date")
    frame = lambda rec: pd.DataFrame(rec).T.reindex(gidx)[tickers]
    series = lambda rec, nm: pd.Series(rec, name=nm).reindex(gidx)

    return ReconResult(
        values=values,
        v_pre=series(v_pre_rec, "v_pre"),
        v_post=series(v_post_rec, "v_post"),
        drift_weights=frame(drift_rec),
        target_weights=frame(tgt_rec),
        turnover=series(turn_rec, "turnover"),
        costs=series(cost_rec, "cost"),
    )
