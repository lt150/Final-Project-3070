"""Network layer: per-ticker yfinance download with a parquet cache.

This is the only module in ``src.data`` that touches the network. Everything
downstream reads the cache, so a Yahoo outage is a non-event and the whole
dataset is rebuildable from a frozen snapshot of ``data/cache/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

# Columns we keep from the raw download, in a stable order.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Reduce yfinance's (field, ticker) MultiIndex columns to plain field names.

    yfinance returns a column MultiIndex even for a single ticker in recent
    versions; older versions return a flat index. Handle both.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Drop whichever level holds the ticker symbol.
        levels = [
            i for i in range(df.columns.nlevels)
            if set(df.columns.get_level_values(i)) == {ticker}
        ]
        if levels:
            df = df.droplevel(levels[0], axis=1)
        else:  # pragma: no cover - unexpected layout, flatten defensively
            df.columns = [c[0] for c in df.columns]
    return df


def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Single-ticker ``yf.download``; retain 'Adj Close'. Raise on empty frame.

    ``end`` is treated as INCLUSIVE (yfinance's own ``end`` is exclusive, so one
    day is added internally). ``auto_adjust=False`` keeps the raw 'Close' and
    the separate split/dividend-adjusted 'Adj Close' column.
    """
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: yfinance returned an empty frame for {start}..{end}")

    df = _flatten_columns(df, ticker)

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{ticker}: download is missing expected column(s) {missing}. "
            f"Got {list(df.columns)}. Check the installed yfinance version."
        )

    df = df[OHLCV_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV + Adj Close per ticker, one parquet per ticker in cache_dir.

    Reads the cache unless ``refresh=True``. Returns ``{ticker: DataFrame}``
    indexed by date.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker}.parquet"
        if path.exists() and not refresh:
            df = pd.read_parquet(path)
            print(f"  [cache hit ] {ticker:5s} {path.name}")
        else:
            reason = "refresh" if path.exists() else "cache miss"
            print(f"  [{reason:10s}] {ticker:5s} downloading {start}..{end} ...")
            df = _download_one(ticker, start, end)
            df.to_parquet(path)
        out[ticker] = df
    return out


def coverage_report(prices: dict[str, pd.DataFrame], gap_threshold: int = 5) -> pd.DataFrame:
    """Per-ticker first/last date, row count, and the largest calendar gap.

    ``gap_threshold`` is in trading days; gaps longer than that (after allowing
    for weekends) are listed so genuine missing stretches stand out from normal
    market holidays.
    """
    rows = []
    for ticker, df in prices.items():
        idx = df.index
        # Count business days between consecutive observations; a normal
        # consecutive pair is 1, a long weekend/holiday run is a few more.
        bdays = pd.Series(idx).diff().dt.days.dropna()
        gaps = [
            (idx[i], idx[i + 1], int((pd.bdate_range(idx[i], idx[i + 1]).size - 1)))
            for i in range(len(idx) - 1)
            if pd.bdate_range(idx[i], idx[i + 1]).size - 1 > gap_threshold
        ]
        rows.append(
            {
                "ticker": ticker,
                "first_date": idx.min().date(),
                "last_date": idx.max().date(),
                "rows": len(df),
                "max_cal_gap_days": int(bdays.max()) if len(bdays) else 0,
                f"gaps_gt_{gap_threshold}bd": len(gaps),
                "gap_detail": "; ".join(
                    f"{a.date()}->{b.date()} ({n}bd)" for a, b, n in gaps
                ) or "-",
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
