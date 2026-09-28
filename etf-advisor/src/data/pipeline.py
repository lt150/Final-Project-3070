"""End-to-end orchestration: cache -> aligned panels -> windows -> artifacts.

CLI::

    python -m src.data.pipeline [--refresh]

``--refresh`` re-downloads every ticker; without it the pipeline runs entirely
off ``data/cache/`` and makes no network calls.

Artifacts written to ``data/processed/``::

    prices.parquet     aligned adjusted-close panel (dates x 8)
    returns.parquet    daily simple returns  (optimiser / baselines / backtester)
    train.npz          X (n, 30, 6), y_ret (n,), y_vol (n,)   -- scaled
    val.npz            same
    test.npz           same
    scaler.joblib      StandardScaler fitted on TRAIN windows only
    meta.parquet       every candidate sample: ticker, window_end, label dates, split
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import build_features_by_ticker
from .fetch import coverage_report, fetch_prices
from .targets import build_targets_by_ticker
from .windows import apply_scale, assign_splits, fit_scale, make_windows, purge_report

SPLITS = ("train", "val", "test")


def align_prices(prices: dict[str, "pd.DataFrame"], universe: list[str]) -> pd.DataFrame:
    """Inner-join the per-ticker 'Adj Close' series onto their common trading dates.

    Features and targets are computed on this panel, so all eight ETFs share one
    date index and a window ending at t means the same calendar day for each.
    Column order follows ``universe``.
    """
    series = {t: prices[t]["Adj Close"].astype("float64") for t in universe}
    panel = pd.concat(series, axis=1, join="inner")
    panel = panel[universe]
    panel.index.name = "Date"
    panel.columns.name = None

    if not panel.index.is_monotonic_increasing:
        panel = panel.sort_index()

    # An inner join across eight funds should already be NaN-free; assert it
    # rather than silently carrying holes into the features.
    if panel.isna().any().any():
        bad = panel.columns[panel.isna().any()].tolist()
        raise ValueError(f"aligned panel has NaNs in {bad} -- inner join did not hold")
    return panel


def build_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily simple-return panel for the optimiser / baselines / backtester."""
    rets = panel.pct_change(fill_method=None)
    rets.index.name = "Date"
    return rets


def _validate(splits: dict, meta: pd.DataFrame, split: pd.Series) -> None:
    """Fail loudly on anything that would quietly poison downstream stages."""
    n_features = len(config.FEATURE_ORDER)
    for name, d in splits.items():
        n = d["X"].shape[0]
        assert d["X"].shape == (n, config.WINDOW, n_features), f"{name}: {d['X'].shape}"
        assert d["y_ret"].shape == (n,), f"{name}: y_ret {d['y_ret'].shape}"
        assert d["y_vol"].shape == (n,), f"{name}: y_vol {d['y_vol'].shape}"
        assert n == int((split == name).sum()), f"{name}: N != meta rows"
        for key in ("X", "y_ret", "y_vol"):
            assert np.isfinite(d[key]).all(), f"{name}: non-finite values in {key}"

    kept = split != "drop"
    assert kept.sum() > 0, "every sample was purged"
    # The guard that matters: no kept sample straddles a split boundary.
    from .windows import region_of

    ra = region_of(meta.loc[kept, "window_end"]).to_numpy()
    rb = region_of(meta.loc[kept, "vol_label_date"]).to_numpy()
    assert (ra == rb).all(), "a kept sample straddles a split boundary"


def run_pipeline(refresh: bool = False) -> None:
    """fetch -> align -> features + targets -> windows -> splits -> scale -> persist."""
    np.random.seed(config.SEED)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"ETF DATA PIPELINE  |  universe={config.UNIVERSE}  seed={config.SEED}")
    print("=" * 78)

    # 1. Fetch (the only networked step, and only when --refresh) ------------
    print(f"\n[1/6] Prices {config.FETCH_START} .. {config.FETCH_END} (refresh={refresh})")
    prices = fetch_prices(
        config.UNIVERSE, config.FETCH_START, config.FETCH_END,
        config.CACHE_DIR, refresh=refresh,
    )
    cov = coverage_report(prices)
    print(cov.drop(columns=["gap_detail"]).to_string())

    # 2. Align ---------------------------------------------------------------
    print("\n[2/6] Aligning on common trading dates")
    panel = align_prices(prices, config.UNIVERSE)
    union = len(set().union(*[set(df.index) for df in prices.values()]))
    print(f"  panel {panel.shape}  {panel.index.min().date()} .. {panel.index.max().date()}")
    print(f"  dates dropped by inner join: {union - len(panel)}")

    returns = build_returns(panel)
    panel.to_parquet(config.PROCESSED_DIR / "prices.parquet")
    returns.to_parquet(config.PROCESSED_DIR / "returns.parquet")
    print(f"  wrote prices.parquet {panel.shape}, returns.parquet {returns.shape}")

    # 3. Features + targets --------------------------------------------------
    print("\n[3/6] Features + targets")
    feats = build_features_by_ticker(panel)
    tgts = build_targets_by_ticker(panel, config.VOL_HORIZON)
    warmup = int(next(iter(feats.values())).isna().any(axis=1).sum())
    print(f"  features {config.FEATURE_ORDER}  (warmup rows dropped later: {warmup})")
    print(f"  targets  ['ret_next', 'fwd_vol']  vol_horizon={config.VOL_HORIZON}d")

    # 4. Windows + purged splits --------------------------------------------
    print(f"\n[4/6] Windowing (window={config.WINDOW}, pooled over {len(config.UNIVERSE)} ETFs)")
    w = make_windows(feats, tgts, config.WINDOW, config.VOL_HORIZON)
    meta, X = w["meta"], w["X"]
    print(f"  candidate samples: {X.shape}")

    split = assign_splits(meta)
    meta = meta.assign(split=split.to_numpy())
    counts = split.value_counts().reindex(["train", "val", "test", "drop"]).fillna(0).astype(int)
    print("  split counts:")
    for k, v in counts.items():
        print(f"    {k:6s} {v:7d}")
    print("  purge breakdown:")
    print("    " + purge_report(w["meta"], split).to_string(index=False).replace("\n", "\n    "))

    # 5. Scale (train-fit only) ---------------------------------------------
    print("\n[5/6] Scaling (StandardScaler fitted on TRAIN windows only)")
    masks = {s: (split == s).to_numpy() for s in SPLITS}
    scaler = fit_scale(X[masks["train"]])
    print(f"  fitted on {int(masks['train'].sum())} x {config.WINDOW} = "
          f"{int(scaler.n_samples_seen_)} rows")
    for i, f in enumerate(config.FEATURE_ORDER):
        print(f"    {f:8s} mean={scaler.mean_[i]: .6e}  var={scaler.var_[i]: .6e}")

    splits = {
        s: {
            "X": apply_scale(X[masks[s]], scaler),
            "y_ret": w["y_ret"][masks[s]],
            "y_vol": w["y_vol"][masks[s]],
        }
        for s in SPLITS
    }

    # 6. Validate + persist --------------------------------------------------
    print("\n[6/6] Validating and persisting")
    _validate(splits, meta, split)
    print("  all shape / finiteness / no-straddle assertions passed")

    for s in SPLITS:
        path = config.PROCESSED_DIR / f"{s}.npz"
        np.savez_compressed(path, **splits[s])
        print(f"  wrote {path.name:12s} X={splits[s]['X'].shape} "
              f"y_ret={splits[s]['y_ret'].shape} y_vol={splits[s]['y_vol'].shape}")

    joblib.dump(scaler, config.PROCESSED_DIR / "scaler.joblib")
    meta.to_parquet(config.PROCESSED_DIR / "meta.parquet")
    print(f"  wrote scaler.joblib")
    print(f"  wrote meta.parquet   {meta.shape}  (all candidates, incl. split='drop')")

    print("\n" + "=" * 78)
    print("DONE. X in the .npz files is SCALED; invert with scaler.inverse_transform.")
    print("=" * 78)


def load_split(name: str) -> dict:
    """Read one persisted split back as ``{'X', 'y_ret', 'y_vol'}``."""
    with np.load(config.PROCESSED_DIR / f"{name}.npz") as z:
        return {k: z[k] for k in ("X", "y_ret", "y_vol")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ETF dataset.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download every ticker instead of reading data/cache/",
    )
    args = parser.parse_args()
    run_pipeline(refresh=args.refresh)


if __name__ == "__main__":
    main()
