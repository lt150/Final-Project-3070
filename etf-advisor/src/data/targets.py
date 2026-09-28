"""Prediction targets: next-day return and forward realised volatility.

Both are indexed by the day ``t`` at which the prediction is made (the window
end), but their information content lives strictly in the future of ``t``:

    ret_next[t] = ret[t+1]                          -> label date t+1
    fwd_vol[t]  = std(ret[t+1 .. t+vol_horizon])    -> label date t+vol_horizon

The 20-day horizon on ``fwd_vol`` is what makes the purged split in
``windows.assign_splits`` necessary: a sample taken at t only becomes
observable 20 trading days later.
"""

from __future__ import annotations

import pandas as pd

from .features import _adj_close


def build_targets(prices: "pd.DataFrame", vol_horizon: int) -> pd.DataFrame:
    """One ETF's adjusted-close series -> DataFrame with ``ret_next`` and ``fwd_vol``.

    Rows without enough forward data (the last ``vol_horizon`` rows for
    ``fwd_vol``, the last row for ``ret_next``) are NaN and are dropped during
    windowing.

    Note on the ``fwd_vol`` alignment: a plain ``ret.shift(-1).rolling(h).std()``
    is still a BACKWARD-looking window on a shifted series -- at t it would
    cover ret[t-h+2 .. t+1], i.e. mostly returns already known at t. Rolling
    first and then shifting by ``-h`` is what actually yields ret[t+1 .. t+h].
    """
    close = _adj_close(prices)
    if not close.index.is_monotonic_increasing:
        raise ValueError("build_targets requires a date-sorted price series")

    ret = close.pct_change(fill_method=None)

    tgt = pd.DataFrame(index=close.index)
    tgt["ret_next"] = ret.shift(-1)
    # std over ret[t+1 .. t+vol_horizon], in raw daily units (not annualised).
    tgt["fwd_vol"] = ret.rolling(vol_horizon).std(ddof=1).shift(-vol_horizon)

    tgt.index.name = close.index.name or "Date"
    return tgt


def build_targets_by_ticker(panel: pd.DataFrame, vol_horizon: int) -> dict[str, pd.DataFrame]:
    """Apply :func:`build_targets` to every column of an aligned price panel."""
    return {t: build_targets(panel[t], vol_horizon) for t in panel.columns}
