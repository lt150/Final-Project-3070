"""Task 11 performance metrics.

All metrics are computed on the net-of-cost daily value path produced by
``engine.run_backtest``. Daily returns are r_d = V(d)/V(d−1) − 1 over
consecutive trading days; N is the number of return days in the window. Because
a window's returns are formed by shifting *within* the window slice, the
registered sub-period rule — sub-period returns use r_d where both d and d−1
lie in the window — holds by construction: the window's first day
contributes a level, not a return.

Registered definitions, verbatim in intent:

  ann. return  geometric, (V_end / V_start)^(252/N) − 1
  ann. vol     √252 · std(r, ddof=1)
  Sharpe       √252 · mean(r) / std(r, ddof=1)   — arithmetic, rf = 0
  Sortino      √252 · mean(r) / DD, DD = sqrt(mean(min(r, 0)²)) over ALL N
               days (full-N denominator), MAR = 0
  max DD       min_d V(d)/max_{s<=d} V(s) − 1, on the daily net path
  T            mean traded fraction over the counted grid dates
  cost drag    total costs in bps per year

Geometric annualised return and arithmetic Sharpe are reported as separate
columns and are never mixed.

Cost drag is defined as the annualised sum of the proportional haircuts,
10⁴ · Σ_k cost_k / years — equivalently, the one-way rate times turnover per
year. Reported in bps/yr.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# The single annualisation constant in the codebase.
SQRT_TRADING_DAYS = math.sqrt(TRADING_DAYS_PER_YEAR)

BPS = 1e4


def daily_returns(values: pd.Series) -> pd.Series:
    """r_d = V(d)/V(d−1) − 1 over consecutive trading days of ``values``.

    The first day of the series yields no return, which is exactly the
    rule when ``values`` has already been sliced to a sub-period window.
    """
    return (values / values.shift(1) - 1.0).iloc[1:]


def drawdown_curve(values: pd.Series) -> pd.Series:
    """V(d)/max_{s<=d} V(s) − 1, running peak taken within ``values``."""
    return values / values.cummax() - 1.0


def max_drawdown(values: pd.Series) -> float:
    return float(drawdown_curve(values).min())


def compute_metrics(
    values: pd.Series,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    cumulative_only: bool = False,
) -> dict[str, float]:
    """Metric set for one portfolio over one window.

    ``values`` is the net-of-cost path already sliced to the window;
    ``turnover``/``costs`` are already restricted to the counted grid dates.
    ``cumulative_only=True`` considers COVID — 
    cumulative return and max drawdown only, every annualised field
    left as NaN so no unregistered number can be quoted from the artifacts.
    """
    v = values.dropna()
    if len(v) < 2:
        raise ValueError("window holds fewer than two trading days")

    r = daily_returns(v)
    n = int(len(r))
    years = n / TRADING_DAYS_PER_YEAR
    cumulative = float(v.iloc[-1] / v.iloc[0] - 1.0)

    out: dict[str, float] = {
        "n_days": float(n),
        "years": years,
        "cumulative_return": cumulative,
        "max_drawdown": max_drawdown(v),
    }

    if cumulative_only:
        out.update({
            "ann_return": np.nan, "ann_vol": np.nan,
            "sharpe": np.nan, "sortino": np.nan,
        })
    else:
        sd = float(r.std(ddof=1))
        mean = float(r.mean())
        downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)))
        out.update({
            "ann_return": float((v.iloc[-1] / v.iloc[0]) ** (TRADING_DAYS_PER_YEAR / n) - 1.0),
            "ann_vol": SQRT_TRADING_DAYS * sd,
            "sharpe": SQRT_TRADING_DAYS * mean / sd,
            "sortino": SQRT_TRADING_DAYS * mean / downside,
        })

    if turnover is not None and len(turnover) > 0:
        out["mean_turnover"] = float(turnover.mean())
        out["n_rebalances_counted"] = float(len(turnover))
    else:
        out["mean_turnover"] = np.nan
        out["n_rebalances_counted"] = 0.0

    if costs is not None and len(costs) > 0:
        out["total_cost"] = float(costs.sum())
        out["cost_drag_bps_per_year"] = BPS * float(costs.sum()) / years
    else:
        out["total_cost"] = np.nan
        out["cost_drag_bps_per_year"] = np.nan

    return out


def counted_grid_dates(
    grid: pd.DatetimeIndex, window_days: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Grid dates a window counts for T and cost drag.

    Grid dates inside the window, minus one coinciding with the window's first
    day. For the full period this is exactly the 155 post-inception dates.
    """
    inside = grid[(grid >= window_days[0]) & (grid <= window_days[-1])]
    return inside[inside != window_days[0]]
