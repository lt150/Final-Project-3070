"""Persistence baseline for Task 8: trailing 20-day realised volatility.

For a sample whose input window ends on trading day t (``window_end``),
``vol_trail_20`` is the sample std (ddof=1) of that ETF's daily returns over
the 20 trading days ending at and including t: ret[t-19 .. t], raw daily
units. Every value in that window is known at the close of day t.

The label ``y_vol`` spans ret[t+1 .. t+20]; the two windows are adjacent and
disjoint, so the baseline is structurally symmetric with the label but
strictly backward-looking.

This one series serves double duty, it is the persistence forecast, 
and it is the denominator of the T3 log-ratio target.
"""

from __future__ import annotations

import pandas as pd

from ..data.config import PROCESSED_DIR

TRAIL_WINDOW = 20
SPLITS = ("train", "val", "test")


def build_vol_trail_20(
    returns: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    """Join ``vol_trail_20`` onto meta's split rows (``split='drop'`` excluded).

    Returns meta's split rows (stored order preserved, which is the npz row
    order: sorted by window_end then ticker within each split) with a
    ``vol_trail_20`` column.
    """
    # value indexed at date t uses returns up to and including t
    trail = returns.rolling(TRAIL_WINDOW).std(ddof=1)

    long = trail.stack().rename("vol_trail_20").reset_index()
    long.columns = ["window_end", "ticker", "vol_trail_20"]

    split_rows = meta[meta["split"].isin(SPLITS)].copy()
    out = split_rows.merge(long, on=["ticker", "window_end"], how="left")

    n_missing = out["vol_trail_20"].isna().sum()
    if n_missing:
        raise ValueError(f"vol_trail_20 missing for {n_missing} split rows")
    return out


def load_baseline_frame() -> pd.DataFrame:
    """Load frozen artifacts and return split rows with ``vol_trail_20``."""
    returns = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    meta = pd.read_parquet(PROCESSED_DIR / "meta.parquet")
    return build_vol_trail_20(returns, meta)
