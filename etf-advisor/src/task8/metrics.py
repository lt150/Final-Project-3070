"""Evaluation metrics for Task 8, all in raw daily vol units.

Every model and the persistence baseline is scored through these same
functions after inversion to raw vol space, so numbers are comparable by
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def basic_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, Pearson r, and std ratio (pred std / realised std)."""
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    err = pred - actual
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson_r": float(np.corrcoef(pred, actual)[0, 1]),
        "std_ratio": float(np.std(pred, ddof=1) / np.std(actual, ddof=1)),
    }


def rmse_skill(rmse_model: float, rmse_persistence: float) -> float:
    """1 - RMSE_model / RMSE_persistence. Positive = beats persistence."""
    return 1.0 - rmse_model / rmse_persistence


def per_ticker_table(
    pred: np.ndarray,
    actual: np.ndarray,
    tickers: pd.Series | np.ndarray,
    pred_persistence: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-ticker RMSE (and skill vs persistence when a baseline is given)."""
    df = pd.DataFrame(
        {"ticker": np.asarray(tickers), "pred": pred, "actual": actual}
    )
    if pred_persistence is not None:
        df["pred_persist"] = pred_persistence

    rows = []
    for ticker, g in df.groupby("ticker", sort=True):
        row = {
            "ticker": ticker,
            "n": len(g),
            "rmse": float(np.sqrt(np.mean((g["pred"] - g["actual"]) ** 2))),
        }
        if pred_persistence is not None:
            rmse_p = float(np.sqrt(np.mean((g["pred_persist"] - g["actual"]) ** 2)))
            row["rmse_persist"] = rmse_p
            row["skill"] = rmse_skill(row["rmse"], rmse_p)
        rows.append(row)
    return pd.DataFrame(rows)


def decile_table(pred: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    """Mean predicted vs mean realised vol by predicted decile.

    A model that collapses to the mean shows a flat ``mean_pred`` column
    against a steep ``mean_actual`` column; a calibrated one tracks it.
    """
    df = pd.DataFrame({"pred": pred, "actual": actual})
    # rank-based bucketing so ties (e.g. a constant prediction) never crash
    df["decile"] = pd.qcut(df["pred"].rank(method="first"), 10, labels=False) + 1
    out = (
        df.groupby("decile")
        .agg(n=("pred", "size"), mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        .reset_index()
    )
    return out
