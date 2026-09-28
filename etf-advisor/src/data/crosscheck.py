"""Independent checks on the cached prices -- catching bad ticks.

Two layers:

1. :func:`stooq_compare` -- a second vendor. Stooq now gates its CSV endpoint
   behind a JavaScript bot-check, and ``pandas-datareader`` 0.11.1 dropped its
   ``stooq`` module, so this returns ``status='unavailable'`` rather than
   failing. It is deliberately fail-soft: a cross-check is a nice-to-have and
   must never break a pipeline that is otherwise fully offline.

2. :func:`tick_screen` -- self-contained, no network. A genuinely bad tick in a
   single ETF shows up two ways: the adjusted and unadjusted series disagree on
   the day's return when no corporate action occurred, and the move is not
   corroborated by the other seven sleeves. Both are checked here.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Second vendor (best-effort)
# --------------------------------------------------------------------------- #
def _fetch_stooq(ticker: str, start: str, end: str, timeout: int = 20) -> pd.DataFrame:
    """Pull one ticker's daily history from Stooq's CSV endpoint.

    Raises ``RuntimeError`` if Stooq answers with anything other than CSV
    (currently it answers with a bot-check page).
    """
    url = (
        "https://stooq.com/q/d/l/"
        f"?s={ticker.lower()}.us&i=d"
        f"&d1={pd.Timestamp(start):%Y%m%d}&d2={pd.Timestamp(end):%Y%m%d}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")

    if not raw.lstrip().lower().startswith("date,"):
        head = raw.lstrip()[:60].replace("\n", " ")
        raise RuntimeError(f"Stooq did not return CSV (got: {head!r})")

    df = pd.read_csv(io.StringIO(raw), parse_dates=["Date"]).set_index("Date").sort_index()
    if df.empty:
        raise RuntimeError("Stooq returned an empty series")
    return df


def stooq_compare(
    panel: pd.DataFrame,
    tickers: list[str],
    start: str,
    end: str,
    tol: float = 0.005,
) -> dict:
    """Compare our adjusted closes against Stooq's on ``tickers``.

    Comparison is on DAILY RETURNS, not price levels: the two vendors
    back-adjust from different bases, so levels differ by a constant factor
    even when the underlying data agrees perfectly. Returns are also what
    actually feeds the model. A date is flagged when the two returns differ by
    more than ``tol`` (default 50 bp).

    Returns ``{'status': 'ok'|'unavailable', 'summary': DataFrame, 'flagged':
    DataFrame, 'error': str|None}``. Never raises.
    """
    rows, flagged, errors = [], [], []

    for ticker in tickers:
        try:
            other = _fetch_stooq(ticker, start, end)
        except (urllib.error.URLError, RuntimeError, ValueError, OSError) as exc:
            errors.append(f"{ticker}: {exc}")
            continue

        ours = panel[ticker].pct_change(fill_method=None)
        theirs = other["Close"].pct_change(fill_method=None)
        joined = pd.concat({"ours": ours, "theirs": theirs}, axis=1, join="inner").dropna()
        if joined.empty:
            errors.append(f"{ticker}: no overlapping dates with Stooq")
            continue

        diff = (joined["ours"] - joined["theirs"]).abs()
        hits = joined[diff > tol].assign(abs_diff=diff[diff > tol], ticker=ticker)
        rows.append(
            {
                "ticker": ticker,
                "overlap_days": len(joined),
                "max_abs_ret_diff": float(diff.max()),
                "median_abs_ret_diff": float(diff.median()),
                f"n_flagged_gt_{tol:g}": int((diff > tol).sum()),
            }
        )
        flagged.append(hits.reset_index())

    if not rows:
        return {
            "status": "unavailable",
            "summary": pd.DataFrame(),
            "flagged": pd.DataFrame(),
            "error": "; ".join(errors) or "no tickers requested",
        }

    return {
        "status": "ok",
        "summary": pd.DataFrame(rows).set_index("ticker"),
        "flagged": pd.concat(flagged, ignore_index=True) if flagged else pd.DataFrame(),
        "error": "; ".join(errors) or None,
    }


# --------------------------------------------------------------------------- #
# 2. Self-contained bad-tick screen (no network)
# --------------------------------------------------------------------------- #
def tick_screen(
    raw: dict[str, "pd.DataFrame"],
    panel: pd.DataFrame,
    z_threshold: float = 8.0,
    max_distribution: float = 0.05,
    eps: float = 1e-4,
) -> dict:
    """Flag suspicious observations using only the cached data.

    ``adj_vs_close``
        ``ret_adj - ret_close`` is the day's implied distribution yield: it is
        zero on ordinary days and positive on an ex-dividend date, because the
        back-adjustment scales the pre-event history down. So a raw difference
        is NOT itself a defect -- what would be a defect is a difference that
        cannot be a distribution: a NEGATIVE one (beyond float noise ``eps``),
        or one larger than ``max_distribution``. Only those are listed.

    ``isolated_moves``
        A move beyond ``z_threshold`` robust sigmas (median / MAD, so the
        estimate is not inflated by the outlier itself) that no other sleeve
        echoes. Real market shocks -- 2010 flash crash, 2020 COVID -- move
        several sleeves at once; a data error moves exactly one.
    """
    adj_rows = []
    n_distributions = {}
    for ticker, df in raw.items():
        r_adj = df["Adj Close"].pct_change(fill_method=None)
        r_raw = df["Close"].pct_change(fill_method=None)
        implied = (r_adj - r_raw).dropna()
        n_distributions[ticker] = int((implied > eps).sum())

        bad = implied[(implied < -eps) | (implied > max_distribution)]
        for date, value in bad.items():
            adj_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "implied_distribution": float(value),
                    "reason": "negative" if value < 0 else "implausibly large",
                    "ret_adj": float(r_adj[date]),
                    "ret_close": float(r_raw[date]),
                }
            )

    rets = panel.pct_change(fill_method=None).dropna()
    med = rets.median()
    mad = (rets - med).abs().median() * 1.4826  # -> robust sigma estimate
    z = (rets - med) / mad

    iso_rows = []
    for ticker in panel.columns:
        others = [c for c in panel.columns if c != ticker]
        extreme = z.index[z[ticker].abs() > z_threshold]
        for date in extreme:
            corroborating = int((z.loc[date, others].abs() > z_threshold / 2).sum())
            if corroborating == 0:
                iso_rows.append(
                    {"ticker": ticker, "date": date, "ret": float(rets.loc[date, ticker]),
                     "robust_z": float(z.loc[date, ticker]),
                     "other_sleeves_moving": corroborating}
                )

    n_extreme = int((z.abs() > z_threshold).sum().sum())
    return {
        "adj_vs_close": pd.DataFrame(adj_rows),
        "n_distributions": pd.Series(n_distributions, name="n_distributions"),
        "isolated_moves": pd.DataFrame(iso_rows),
        "n_extreme_total": n_extreme,
        "z_threshold": z_threshold,
        "max_distribution": max_distribution,
    }
