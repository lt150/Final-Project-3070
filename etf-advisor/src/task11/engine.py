"""Task 11 backtest engine.

Run indirectly; this module is pure accounting: given a
price panel, a rebalance grid and a point-in-time weight function, it produces
a net-of-cost daily value path plus the per-rebalance weight/turnover/cost
record. It knows nothing about metrics, nothing about H1, 
and nothing about which allocator it is running.

Registered conventions implemented here, verbatim in intent:

  1.1 Execution timing — ``weight_fn(t)`` is evaluated with data realised
       through the close of t; the portfolio is rebalanced at the close of t;
       the first return earned under those weights is the t -> t+1
       close-to-close return. No return attributed to any weight vector
       precedes the data used to compute it.

  1.2 Drift, turnover, costs — between rebalances the portfolio buys and
       holds: unit holdings are fixed at each execution close and portfolio
       value is the mark-to-market of those units at adjusted closes
       (dividends implicit in the adjusted price series). At grid date t_k:

           G_i     = P[t_k, i] / P[t_prev, i]
           w_drift = (w_prev_target * G) / sum(w_prev_target * G)
           T       = sum_i |w_target_i - w_drift_i|
           cost    = cost_rate * T
           V_post  = V_pre * (1 - cost)
           units_i = V_post * w_target_i / P[t_k, i]
           V(d)    = sum_i units_i * P[d, i]   for d in (t_k, t_{k+1}]

       T is the traded fraction = buy notional + sell notional, each dollar
       traded paying the one-way rate once (the DeMiguel-style convention;
       round-trip 20 bps at the registered 10 bps headline). The cost is a
       multiplicative value haircut at the rebalance close, so post-cost
       weights equal target weights exactly and drifted weights are
       cost-invariant.

  1.3 Inception — the first grid date is funded from cash. That is expressed
       here without a special case: the "previous target" entering the first
       rebalance is the zero vector, so w_drift = 0 and
       T = sum_i |w_target_i| = 1, giving cost = one-way rate and
       V_post(t_0) = 1 - rate. No branch, no hand-set constant.

Value-path convention: ``values[d]`` is the portfolio value 
at the CLOSE of d, net of any cost charged at d. On a rebalance
date that is V_post, not V_pre, the recorded inception value is fixed at
V_post = 0.999, and the path cannot switch convention at later rebalances.
V_pre is retained separately in the result for reconciliation and reporting.

The cost rate is a parameter whose registered headline value is
``COST_RATE`` = 10 bps. Nothing else in this module is tunable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Headline transaction costs: 10 bps one-way, proportional to turnover.
COST_RATE = 0.0010

# Pre-inception cash. Every portfolio is funded with 1.0 and pays the inception
# haircut, so all paths start at 1 - COST_RATE = 0.999.
V_CASH = 1.0

WeightFn = Callable[[pd.Timestamp], pd.Series]


@dataclass(frozen=True)
class BacktestResult:
    """Everything the engine produces."""

    values: pd.Series           # net-of-cost value at every trading day in span
    v_pre: pd.Series            # value at each grid date BEFORE the cost haircut
    v_post: pd.Series           # value at each grid date AFTER the cost haircut
    target_weights: pd.DataFrame     # grid dates x tickers, allocator output
    drift_weights: pd.DataFrame      # grid dates x tickers, zero row at inception
    post_cost_weights: pd.DataFrame  # grid dates x tickers, units * P / V_post
    turnover: pd.Series         # T at each grid date (inception = 1.0)
    costs: pd.Series            # cost_rate * T at each grid date
    cost_paid: pd.Series        # cost * V_pre, in value units
    units: pd.DataFrame         # grid dates x tickers, holdings set at that close
    cost_rate: float

    @property
    def tickers(self) -> list[str]:
        return list(self.target_weights.columns)


def run_backtest(
    weight_fn: WeightFn,
    prices: pd.DataFrame,
    grid: pd.DatetimeIndex,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    cost_rate: float = COST_RATE,
) -> BacktestResult:
    """Run one portfolio over ``[start, end]`` rebalancing on ``grid``.
    
    ``prices`` are adjusted closes indexed by trading day; ``grid`` must begin
    at the first trading day of the span (the inception funding date) and every
    grid date must be a trading day.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    tickers = list(prices.columns)

    days = prices.index[(prices.index >= start) & (prices.index <= end)]
    if len(days) == 0:
        raise ValueError(f"no trading days in span {start.date()}..{end.date()}")

    grid = pd.DatetimeIndex(grid)
    grid = grid[(grid >= start) & (grid <= end)]
    if len(grid) == 0:
        raise ValueError("rebalance grid is empty within the span")
    if grid[0] != days[0]:
        raise ValueError(
            f"grid starts {grid[0].date()} but the span's first trading day is "
            f"{days[0].date()}: inception funding must coincide with the span start"
        )
    missing = [d for d in grid if d not in prices.index]
    if missing:
        raise ValueError(f"grid dates absent from the price index: {missing}")

    P = prices.loc[days]

    values = pd.Series(np.nan, index=days, dtype=float, name="value")
    v_pre_rec, v_post_rec = {}, {}
    tgt_rec, drift_rec, post_rec, units_rec = {}, {}, {}, {}
    turn_rec, cost_rec, paid_rec = {}, {}, {}

    # The portfolio enters the first rebalance holding cash, i.e. the
    # previous target is the zero vector. This makes T(t_0) = sum|w| = 1 an
    # identity rather than a hand-set constant.
    w_prev = pd.Series(0.0, index=tickers)
    t_prev: pd.Timestamp | None = None
    units = pd.Series(0.0, index=tickers)

    for k, t in enumerate(grid):
        px_t = P.loc[t]

        if t_prev is None:
            v_pre = float(V_CASH)
            w_drift = pd.Series(0.0, index=tickers)
        else:
            v_pre = float(units @ px_t)
            gross = px_t / P.loc[t_prev]
            drifted = w_prev * gross
            w_drift = drifted / drifted.sum()

        w_tgt = pd.Series(weight_fn(t), dtype=float).reindex(tickers)
        if w_tgt.isna().any():
            raise ValueError(f"weight_fn({t.date()}) does not cover {tickers}")

        turnover = float((w_tgt - w_drift).abs().sum())   # traded fraction
        cost = cost_rate * turnover
        v_post = v_pre * (1.0 - cost)
        units = v_post * w_tgt / px_t                     # held until t_{k+1}

        # Value at the close of t is net of the cost charged at t (see module docstring).
        values.loc[t] = v_post

        t_next = grid[k + 1] if k + 1 < len(grid) else None
        seg_end = t_next if t_next is not None else days[-1]
        seg = days[(days > t) & (days <= seg_end)]
        if len(seg) > 0:
            values.loc[seg] = P.loc[seg].to_numpy() @ units.to_numpy()

        v_pre_rec[t], v_post_rec[t] = v_pre, v_post
        tgt_rec[t], drift_rec[t] = w_tgt, w_drift
        post_rec[t] = units * px_t / v_post
        units_rec[t] = units.copy()
        turn_rec[t], cost_rec[t], paid_rec[t] = turnover, cost, v_pre * cost

        w_prev, t_prev = w_tgt, t

    gidx = pd.DatetimeIndex(grid, name="date")
    frame = lambda rec: pd.DataFrame(rec).T.reindex(gidx)[tickers]
    series = lambda rec, nm: pd.Series(rec, name=nm).reindex(gidx)

    return BacktestResult(
        values=values,
        v_pre=series(v_pre_rec, "v_pre"),
        v_post=series(v_post_rec, "v_post"),
        target_weights=frame(tgt_rec),
        drift_weights=frame(drift_rec),
        post_cost_weights=frame(post_rec),
        turnover=series(turn_rec, "turnover"),
        costs=series(cost_rec, "cost"),
        cost_paid=series(paid_rec, "cost_paid"),
        units=frame(units_rec),
        cost_rate=cost_rate,
    )


def rerun_values_at_cost_rate(
    result: BacktestResult,
    prices: pd.DataFrame,
    cost_rate: float,
) -> pd.Series:
    """Recompute the daily value path at a different cost rate, weights frozen.

    Drifted weights and therefore target weights,
    turnover and every H1 quantity are invariant to the cost rate, because
    the haircut is proportional and uniform. The sensitivity runs reuse 
    the frozen headline weights and only the value path 
    is recomputed. This function is the mechanisation of that
    invariance claim; Stage B proves the claim itself by rerunning the engine.
    """
    days = result.values.index
    P = prices.loc[days, result.tickers]
    grid = result.turnover.index

    values = pd.Series(np.nan, index=days, dtype=float, name="value")
    v = float(V_CASH)
    units = pd.Series(0.0, index=result.tickers)
    for k, t in enumerate(grid):
        if k > 0:
            v = float(units @ P.loc[t])
        v = v * (1.0 - cost_rate * float(result.turnover.loc[t]))
        units = v * result.target_weights.loc[t] / P.loc[t]
        values.loc[t] = v
        t_next = grid[k + 1] if k + 1 < len(grid) else None
        seg_end = t_next if t_next is not None else days[-1]
        seg = days[(days > t) & (days <= seg_end)]
        if len(seg) > 0:
            values.loc[seg] = P.loc[seg].to_numpy() @ units.to_numpy()
    return values
