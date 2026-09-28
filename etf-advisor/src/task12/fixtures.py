"""Registered constants and the imported code paths they name.

Everything in this module was fixed before any attribution
existed. Nothing here is derived from a result. The upstream symbols are
IMPORTED, never restated: this module contains no reimplementation of any
Task 9 or Task 10 numeric.

Import map of record -- Gate A1 verifies every entry by object identity against
a freshly imported module. The list is closed: no other upstream symbol is used
anywhere in ``src/task12``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..data.config import ROOT
from ..task9.allocators import PROFILES
from ..task9.forecaster_deploy import fit_coeffs, forecast_vol, vol_trail_20
from ..task10.recommend import TICKERS, allocate

OUT_DIR = ROOT / "outputs" / "task12"

# (symbol, module of record, purpose) -- Gate A1's table.
IMPORT_MAP: tuple[tuple[str, str, str], ...] = (
    ("allocate", "src.task10.recommend", "the explained function f(d) = allocate(p, d, t).weights"),
    ("TICKERS", "src.task10.recommend", "canonical ticker ordering (task10's own source)"),
    ("PROFILES", "src.task9.allocators", "the three registered profile strings"),
    ("forecast_vol", "src.task9.forecaster_deploy", "explicand d_forecast(t) -- walk-forward OLS forecast vols"),
    ("vol_trail_20", "src.task9.forecaster_deploy", "background d_classical(t) -- the ablation twin's D"),
    ("fit_coeffs", "src.task9.forecaster_deploy", "walk-forward per-ticker (a_i, b_i) for the Sec. 1.3 check"),
    ("ROOT", "src.data.config", "repo root for output paths"),
)

# stage D only
IMPORT_MAP_STAGE_D: tuple[tuple[str, str, str], ...] = (
    ("build_model", "src.task8.modeling", "rebuild the frozen LSTM from its checkpoint"),
    ("PROCESSED_DIR", "src.data.config", "frozen window arrays (train/val npz)"),
    ("FEATURE_ORDER", "src.data.config", "input channel names of the frozen schema"),
    ("WINDOW", "src.data.config", "lag count of the frozen schema, asserted against the arrays"),
)

# --------------------------------------------------------------------------- #
# Registered constants
# --------------------------------------------------------------------------- #
SMOKE_DATES: tuple[str, ...] = ("2012-02-01", "2020-03-02", "2022-06-01", "2024-12-02")
ANCHOR_CELL: tuple[str, str] = ("2024-12-02", "balanced")

EPS = 0.01                      # multiplicative, explicand only
MONITORED_PAIR: tuple[str, str] = ("SPY", "QQQ")
CONTROL_PAIR: tuple[str, str] = ("SPY", "AGG")

RHO_THRESHOLD = 10.0            # registered judgment constant, fixed pre-computation
RHO_DENOM_FLOOR = 1e-6          # the solver band; prevents division by solver noise

# Engine tolerances
TOL_EFFICIENCY = 1e-12          # efficiency identity, engine arithmetic
TOL_CLOSED_FORM_DIAG = 1e-12    # diagonal vs closed form
TOL_TOY = 1e-12                 # Gate A2 toy function
TOL_UNPLANNED_SOLVE_PAIR = 1e-6  # only if an unplanned solve-vs-solve check arises

# linear-layer exactness check: calm vs crisis, maximally separated
# registered inputs, so the check is non-trivial.
LINEAR_EXPLICAND_DATE = "2024-12-02"
LINEAR_BACKGROUND_DATE = "2020-03-02"

# Gate A2 -- registered toy
TOY_EXACT_PHI: tuple[float, float, float] = (2.0, 1.5, 1.5)
TOY_EXPLICAND: tuple[float, float, float] = (1.0, 1.0, 1.0)
TOY_BACKGROUND: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Files of record. Every reference value is READ from these at execution time.
TASK9_DIR = ROOT / "outputs" / "task9"
TASK10_DIR = ROOT / "outputs" / "task10"
RECORD_TASK9_STAGE_B = TASK9_DIR / "stage_b_report.md"
RECORD_TASK9_STAGE_C = TASK9_DIR / "stage_c_report.md"
RECORD_TASK9_SMOKE = TASK9_DIR / "smoke_weights.csv"
RECORD_TASK10_SMOKE = TASK10_DIR / "smoke_recommendations.csv"



def cells() -> tuple[tuple[str, str], ...]:
    """The 12 registered cells: 4 smoke dates x 3 profiles."""
    return tuple((date, profile) for date in SMOKE_DATES for profile in PROFILES)


def ticker_index(ticker: str) -> int:
    """Position of a ticker in the canonical order."""
    return TICKERS.index(ticker)


def _as_vector(series: pd.Series, what: str) -> np.ndarray:
    """Take a Task 9 vol Series to an ndarray after asserting its axis order.

    Validation, not arithmetic: the values are passed through untouched.
    """
    if tuple(series.index) != TICKERS:
        raise RuntimeError(
            f"{what} axis order {tuple(series.index)} != canonical {TICKERS} -- STOP"
        )
    return series.to_numpy()


def explicand(date: str) -> np.ndarray:
    """d_forecast(t) via the imported Task 9 deployment."""
    return _as_vector(forecast_vol(pd.Timestamp(date)), "forecast_vol")


def trailing_vol(date: str) -> np.ndarray:
    """``vol_trail_20(t)`` via the imported Task 9 deployment, canonical order."""
    return _as_vector(vol_trail_20(pd.Timestamp(date)), "vol_trail_20")


def background(date: str) -> np.ndarray:
    """d_classical(t): the trailing vol vector -- the ablation twin's D.

    Reached through the same imported Task 9 code path that feeds
    ``sigma(t, "classical")``, which calls ``vol_trail_20(t)`` and reindexes it
    onto the correlation matrix's axis; that reindex is a verified no-op here
    because ``vol_trail_20`` is already in canonical order.
    Gate B2 certifies the identity end-to-end against the rb_classical record.
    """
    return trailing_vol(date)


def weights_fn(profile: str, date: str) -> Callable[[Sequence[float]], np.ndarray]:
    """f(d) = allocate(p, d, t).weights at fixed (t, p) -- the explained function.

    Profile is a conditioning variable, never a feature.
    """

    def f(d: Sequence[float]) -> np.ndarray:
        return np.asarray(allocate(profile, d, date).weights, dtype=float)

    return f


def linear_coefficients(date: str) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) at a fixed date through the imported Task 9 walk-forward fit path.

    ``fit_coeffs`` -> ``_fit_coeffs_cached`` -> closed-form OLS on every sample
    whose 20-day label window is realised by t. Never retyped.
    """
    coeffs = fit_coeffs(pd.Timestamp(date))
    a = np.array([coeffs[tk][0] for tk in TICKERS], dtype=float)
    b = np.array([coeffs[tk][1] for tk in TICKERS], dtype=float)
    return a, b


def linear_map(date: str) -> tuple[Callable[[Sequence[float]], np.ndarray], np.ndarray, np.ndarray]:
    """g(x)_i = a_i(t*) + b_i(t*) * x_i -- the explained function.

    It cannot be reached through any Task 9
    callable: ``forecast_vol`` evaluates the same map but fixes
    x = ``vol_trail_20(t)``, whereas the exactness check must evaluate it at
    2^8 hybrid inputs. Declared at Gate A1 rather than absorbed silently.
    """
    a, b = linear_coefficients(date)

    def g(x: Sequence[float]) -> np.ndarray:
        return a + b * np.asarray(x, dtype=float)

    return g, a, b


def toy_function(x: Sequence[float]) -> np.ndarray:
    """Gate A2's registered toy: f(x1, x2, x3) = 2*x1 + 3*x2*x3, as a 1-vector."""
    x = np.asarray(x, dtype=float)
    return np.array([2.0 * x[0] + 3.0 * x[1] * x[2]], dtype=float)
