"""Task 11 Stage B fixtures.

Two fixtures:

  Fixture 1 (synthetic) — three tickers A, B, C; three abstract consecutive
  dates D0, D1, D2 with no intermediate days; prices A (100, 110, 121),
  B (100, 100, 100), C (100, 90, 81); targets w(D0) = (0.5, 0.3, 0.2),
  w(D1) = (0.4, 0.4, 0.2). The expected results are expressed as exact
  expressions, so they are coded here with ``fractions.Fraction`` and only
  converted to float at the comparison. ``fixture1_expected`` additionally
  derives every quantity from the fixture inputs by an independent Fraction
  computation and asserts the two agree, so a transcription slip in either the
  brief's arithmetic or this module's transcription is caught before the gate
  is evaluated.

  Fixture 2 (real data) — the restricted window 2012-01-03 .. 2012-07-02, seven
  grid dates, four portfolios: one_over_n, static_bucket balanced, rb_classical
  balanced, rb_advisor balanced. The window is already partially revealed by
  the Task 9 2012-02-01 smoke table, so nothing new leaks.
"""

from __future__ import annotations

from fractions import Fraction as F

import pandas as pd

from ..data.config import PROCESSED_DIR, UNIVERSE
from ..task9.allocators import get_allocator
from .engine import COST_RATE

# --------------------------------------------------------------------------- #
# Fixture 1 — synthetic
# --------------------------------------------------------------------------- #
FIXTURE1_TICKERS = ["A", "B", "C"]

# Abstract consecutive dates; the labels D0/D1/D2 are what the brief names.
FIXTURE1_DATES = pd.DatetimeIndex(
    [pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-04"), pd.Timestamp("2000-01-05")],
    name="Date",
)
D0, D1, D2 = FIXTURE1_DATES

FIXTURE1_PRICES = pd.DataFrame(
    [[100.0, 100.0, 100.0],
     [110.0, 100.0, 90.0],
     [121.0, 100.0, 81.0]],
    index=FIXTURE1_DATES,
    columns=FIXTURE1_TICKERS,
)

FIXTURE1_TARGETS = {
    D0: pd.Series([0.5, 0.3, 0.2], index=FIXTURE1_TICKERS),
    D1: pd.Series([0.4, 0.4, 0.2], index=FIXTURE1_TICKERS),
}

# Grid = {D0, D1}; D2 is the terminal valuation day.
FIXTURE1_GRID = pd.DatetimeIndex([D0, D1])


def fixture1_weight_fn(t: pd.Timestamp) -> pd.Series:
    return FIXTURE1_TARGETS[pd.Timestamp(t)].copy()


def fixture1_expected() -> dict[str, object]:
    """Registered expressions, plus an independent re-derivation.

    Returns exact ``Fraction`` values.
    """
    rate = F(1, 1000)  # 10 bps, exact

    reg = {
        "T_D0": F(1),
        "V_post_D0": F(999, 1000),
        "w_drift_D1": (F(55, 103), F(30, 103), F(18, 103)),
        "T_D1": F(276, 10) / F(103),                       # 27.6 / 103
        "cost_D1": rate * (F(276, 10) / F(103)),           # 0.001 * 27.6/103
        "V_pre_D1": F(999, 1000) * F(103, 100),            # 0.999 * 1.03
    }
    reg["V_post_D1"] = reg["V_pre_D1"] * (1 - reg["cost_D1"])
    reg["V_D2"] = reg["V_post_D1"] * F(102, 100)           # gross 1.02

    # --- independently derived from the fixture inputs -------------
    px = {d: [F(int(v)) for v in FIXTURE1_PRICES.loc[d]] for d in FIXTURE1_DATES}
    w0 = [F(1, 2), F(3, 10), F(1, 5)]
    w1 = [F(2, 5), F(2, 5), F(1, 5)]

    t_d0 = sum(abs(x - F(0)) for x in w0)                  # funded from cash
    v_post_d0 = F(1) * (1 - rate * t_d0)

    gross = [px[D1][i] / px[D0][i] for i in range(3)]
    drifted = [w0[i] * gross[i] for i in range(3)]
    s = sum(drifted)
    w_drift_d1 = tuple(x / s for x in drifted)
    t_d1 = sum(abs(w1[i] - w_drift_d1[i]) for i in range(3))
    cost_d1 = rate * t_d1
    units_d0 = [v_post_d0 * w0[i] / px[D0][i] for i in range(3)]
    v_pre_d1 = sum(units_d0[i] * px[D1][i] for i in range(3))
    v_post_d1 = v_pre_d1 * (1 - cost_d1)
    units_d1 = [v_post_d1 * w1[i] / px[D1][i] for i in range(3)]
    v_d2 = sum(units_d1[i] * px[D2][i] for i in range(3))

    der = {
        "T_D0": t_d0, "V_post_D0": v_post_d0, "w_drift_D1": w_drift_d1,
        "T_D1": t_d1, "cost_D1": cost_d1, "V_pre_D1": v_pre_d1,
        "V_post_D1": v_post_d1, "V_D2": v_d2,
    }

    for k in reg:
        assert reg[k] == der[k], (
            f"Fixture 1 self-consistency: registered {k} = {reg[k]} but the "
            f"§1.2 re-derivation gives {der[k]} -- transcription error, STOP"
        )
    reg["cost_rate"] = rate
    return reg


# --------------------------------------------------------------------------- #
# Fixture 2 — real data, restricted window
# --------------------------------------------------------------------------- #
FIXTURE2_START = "2012-01-03"
FIXTURE2_END = "2012-07-02"
FIXTURE2_GRID_COUNT = 7

# (allocator_id, profile) -> portfolio key.
FIXTURE2_PORTFOLIOS = (
    ("one_over_n", None),
    ("static_bucket", "balanced"),
    ("rb_classical", "balanced"),
    ("rb_advisor", "balanced"),
)


def portfolio_key(allocator_id: str, profile: str | None) -> str:
    return allocator_id if profile is None else f"{allocator_id}:{profile}"


def allocator_weight_fn(allocator_id: str, profile: str | None):
    """Adapter from the Task 9 interface to the engine's ``weight_fn``."""
    alloc = get_allocator(allocator_id, profile)
    return lambda t: alloc.weights(pd.Timestamp(t))


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "prices.parquet")[list(UNIVERSE)]


def load_returns() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "returns.parquet")[list(UNIVERSE)]


__all__ = [
    "COST_RATE", "D0", "D1", "D2", "FIXTURE1_DATES", "FIXTURE1_GRID",
    "FIXTURE1_PRICES", "FIXTURE1_TARGETS", "FIXTURE1_TICKERS",
    "FIXTURE2_END", "FIXTURE2_GRID_COUNT", "FIXTURE2_PORTFOLIOS",
    "FIXTURE2_START", "allocator_weight_fn", "fixture1_expected",
    "fixture1_weight_fn", "load_prices", "load_returns", "portfolio_key",
]
