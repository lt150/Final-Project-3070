"""Feature engineering: the six features, from adjusted close only.

Every value at day t is a function of ``Adj Close`` up to and including t, so a
window ending at t contains no information from t+1 onward. This is the single
property the whole leakage story rests on.

The definitions are carried over unchanged from the Phase 0 prototype.
(``src/prototype/data_pipeline.py::compute_features``)
"""

from __future__ import annotations

import pandas as pd

from . import config


def _adj_close(prices: "pd.DataFrame | pd.Series") -> pd.Series:
    """Accept either a Series of adjusted closes or a frame containing one."""
    if isinstance(prices, pd.Series):
        return prices.astype("float64")
    if isinstance(prices, pd.DataFrame):
        for col in ("Adj Close", "adj_close"):
            if col in prices.columns:
                return prices[col].astype("float64")
        if prices.shape[1] == 1:
            return prices.iloc[:, 0].astype("float64")
    raise TypeError(
        "build_features expects a Series of adjusted closes or a DataFrame "
        f"with an 'Adj Close' column; got columns={getattr(prices, 'columns', None)}"
    )


def build_features(prices: "pd.DataFrame") -> pd.DataFrame:
    """One ETF's aligned adjusted-close series -> DataFrame with FEATURE_ORDER columns.

    Leading rows are NaN while the rolling lookbacks fill (the longest is the
    50-day MA); they are left as NaN here and dropped during windowing.
    """
    close = _adj_close(prices)
    if not close.index.is_monotonic_increasing:
        raise ValueError("build_features requires a date-sorted price series")

    ret = close.pct_change(fill_method=None)

    feat = pd.DataFrame(index=close.index)
    feat["ret"] = ret
    feat["vol_10"] = ret.rolling(config.VOL_LOOKBACK).std(ddof=1)
    feat["mom_5"] = ret.rolling(config.MOM_SHORT).sum()
    feat["mom_20"] = ret.rolling(config.MOM_LONG).sum()
    feat["px_ma20"] = close / close.rolling(config.MA_SHORT).mean()
    feat["px_ma50"] = close / close.rolling(config.MA_LONG).mean()

    feat.index.name = close.index.name or "Date"
    return feat[config.FEATURE_ORDER]


def build_features_by_ticker(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply :func:`build_features` to every column of an aligned price panel."""
    return {ticker: build_features(panel[ticker]) for ticker in panel.columns}
