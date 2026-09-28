"""Windowing, purge-based split assignment, and the train-only scaler.

One sample = one ETF's 30-day feature block. Samples are pooled across the
eight ETFs, so the LSTM sees ~8x the data a single-sleeve prototype would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

# Region labels in chronological order. Only the middle three are usable.
REGIONS = ("pre", "train", "val", "test", "post")
USABLE_REGIONS = ("train", "val", "test")


def make_windows(
    features_by_ticker: dict[str, "pd.DataFrame"],
    targets_by_ticker: dict[str, "pd.DataFrame"],
    window: int,
    vol_horizon: int,
) -> dict:
    """Build pooled samples from per-ticker feature/target frames.

    A sample is emitted for ticker/day t only when the whole feature window
    ``[t-window+1 .. t]`` is NaN-free AND both targets at t are non-NaN (which
    implies t+1 and t+vol_horizon both exist on the shared trading calendar).

    Returns ``{'X': (N, window, 6), 'y_ret': (N,), 'y_vol': (N,), 'meta': DataFrame}``
    with samples ordered deterministically by ``(window_end, ticker)``.
    """
    n_features = len(config.FEATURE_ORDER)
    X_parts, yr_parts, yv_parts, meta_parts = [], [], [], []

    for ticker in sorted(features_by_ticker):
        feat = features_by_ticker[ticker][config.FEATURE_ORDER]
        tgt = targets_by_ticker[ticker]
        if not feat.index.equals(tgt.index):
            raise ValueError(f"{ticker}: feature and target indices differ")

        index = feat.index
        values = feat.to_numpy(dtype=np.float64)
        n = len(index)

        row_ok = ~np.isnan(values).any(axis=1)
        # A window ending at i is usable iff the last `window` rows are all ok.
        window_ok = (
            pd.Series(row_ok).rolling(window).sum().to_numpy() == window
        )

        y_ret = tgt["ret_next"].to_numpy(dtype=np.float64)
        y_vol = tgt["fwd_vol"].to_numpy(dtype=np.float64)
        target_ok = ~np.isnan(y_ret) & ~np.isnan(y_vol)

        # Guard the label-date lookups; non-NaN targets should already imply this.
        has_labels = np.zeros(n, dtype=bool)
        has_labels[: max(n - vol_horizon, 0)] = True

        keep = np.flatnonzero(window_ok & target_ok & has_labels)
        if keep.size == 0:
            continue

        # (n_keep, window, n_features) via a strided gather -- one row per window.
        offsets = np.arange(-window + 1, 1)
        X_parts.append(values[keep[:, None] + offsets[None, :], :])
        yr_parts.append(y_ret[keep])
        yv_parts.append(y_vol[keep])
        meta_parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "window_end": index[keep],
                    "ret_label_date": index[keep + 1],
                    "vol_label_date": index[keep + vol_horizon],
                }
            )
        )

    if not X_parts:
        raise ValueError("make_windows produced no samples")

    X = np.concatenate(X_parts, axis=0)
    y_ret = np.concatenate(yr_parts)
    y_vol = np.concatenate(yv_parts)
    meta = pd.concat(meta_parts, ignore_index=True)

    # Deterministic ordering: window_end, then ticker. Stable sort, no randomness.
    order = np.lexsort((meta["ticker"].to_numpy(), meta["window_end"].to_numpy()))
    X, y_ret, y_vol = X[order], y_ret[order], y_vol[order]
    meta = meta.iloc[order].reset_index(drop=True)

    assert X.shape == (len(meta), window, n_features), X.shape
    return {"X": X, "y_ret": y_ret, "y_vol": y_vol, "meta": meta}


def region_of(dates: "pd.Series | pd.DatetimeIndex") -> pd.Series:
    """Map dates to {pre, train, val, test, post} using the config boundaries."""
    d = pd.to_datetime(pd.Series(dates).to_numpy())
    bounds = [
        pd.Timestamp(config.EVAL_START),
        pd.Timestamp(config.TRAIN_END),
        pd.Timestamp(config.VAL_END),
        pd.Timestamp(config.TEST_END),
    ]
    out = pd.Series("post", index=range(len(d)), dtype=object)
    out[d <= bounds[3]] = "test"
    out[d <= bounds[2]] = "val"
    out[d <= bounds[1]] = "train"
    out[d < bounds[0]] = "pre"
    return out


def assign_splits(meta: "pd.DataFrame") -> pd.Series:
    """Purge-based split assignment keyed to the binding (20-day) label horizon.

    A sample spans ``[window_end, vol_label_date]``. It is kept only if BOTH
    ends fall in the same usable region -- so no training sample's label is
    drawn from the validation period, and so on. Samples straddling an internal
    boundary are purged. Both targets share this split; that is deliberately
    conservative for the 1-day return target, and keeps the two heads directly
    comparable on identical samples.
    """
    ra = region_of(meta["window_end"]).to_numpy()
    rb = region_of(meta["vol_label_date"]).to_numpy()

    split = np.where((ra == rb) & np.isin(ra, USABLE_REGIONS), ra, "drop")
    return pd.Series(split, index=meta.index, name="split")


def purge_report(meta: "pd.DataFrame", split: "pd.Series") -> pd.DataFrame:
    """Break the dropped samples down by which boundary they straddle."""
    ra = region_of(meta["window_end"]).to_numpy()
    rb = region_of(meta["vol_label_date"]).to_numpy()
    dropped = (split == "drop").to_numpy()

    rows = []
    for a in REGIONS:
        for b in REGIONS:
            n = int((dropped & (ra == a) & (rb == b)).sum())
            if n:
                reason = (
                    "window ends outside the evaluation range"
                    if a in ("pre", "post")
                    else f"label crosses the {a}->{b} boundary"
                )
                rows.append(
                    {"window_end_region": a, "vol_label_region": b, "n_dropped": n,
                     "reason": reason}
                )
    return pd.DataFrame(rows)


def fit_scale(X_train: "np.ndarray") -> StandardScaler:
    """Fit a StandardScaler on train windows flattened to (n_train*window, 6)."""
    if X_train.ndim != 3:
        raise ValueError(f"expected (N, window, n_features), got {X_train.shape}")
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    return scaler


def apply_scale(X: "np.ndarray", scaler: StandardScaler) -> np.ndarray:
    """Transform each window with a train-fitted scaler; preserve (N, window, 6)."""
    if X.shape[0] == 0:
        return X.copy()
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[2])).reshape(shape)
