"""Task 9 Stage A: deployment of the production vol forecaster (OLS-per-ticker).

The production forecaster is the Task 8 winner: per-ticker OLS
``y_vol = a_i + b_i * vol_trail_20``, fit on realised samples only, raw daily
vol units throughout (never annualised). This module provides its deployment
form for the Task 11 backtester:

  * ``train_fit``      -- A.1 frozen reference: pooled + per-ticker OLS on the
                          train split rows, Task 8 Stage C construction.
  * ``vol_trail_20``   -- A.2 on-demand trailing vol at any trading day t,
                          computed from returns.parquet, never read from a
                          stored feature parquet at rebalance time.
  * ``fit_coeffs``     -- A.3 walk-forward refit: per-ticker OLS on all
                          realised samples at t (memoised per t).
  * ``forecast_vol``   -- A.3 deployment forecast a_i(t) + b_i(t) * trail_i(t).

Leakage clause: a sample with window_end = s has label y_vol spanning
[s+1, s+20]; it may enter a fit at time t only once the
whole span is realised, i.e. ``vol_label_date <= t``. ``window_end < t`` is
NOT sufficient (leaks up to 20 days of label overlap).

Fit universe (interpretive note, recorded in the Stage A report): every fit --
frozen and walk-forward -- draws from the split rows (train/val/test) of the
Task 8 construction: x = vol_trail_20 from the frozen baseline.parquet, y =
raw y_vol from the split npz files, joined positionally per split (both are
stored in meta's (window_end, ticker) row order). The 896 split='drop' rows
never enter any fit: they were purged in Task 7 (pre-2010 warm-up or
boundary-straddling labels) and carry no y_vol in any artifact. This is also
what makes Gate A3 exact: at t* = max train vol_label_date the eligible set
is precisely the train rows (576 drop rows have vol_label_date <= t* and
would otherwise contaminate the tie-out).

"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

from ..data.config import PROCESSED_DIR, ROOT, SEED, UNIVERSE

TASK8_DIR = ROOT / "outputs" / "task8"
OUT_DIR = ROOT / "outputs" / "task9"

TRAIL_WINDOW = 20
SPLITS = ("train", "val", "test")

# Task 8 Stage C, Gate C3 printed reference (%.6g) -- the A1 checksum.
GATE_A1_REF_A = "0.00293256"
GATE_A1_REF_B = "0.659492"
GATE_A2_N, GATE_A2_TOL = 30, 1e-10
GATE_A3_TOL = 1e-12
GATE_A4_XCHECK_TOL = 1e-10

CONTINUITY_START, CONTINUITY_END = "2023-01-03", "2024-12-02"

_CACHE: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Frozen inputs (loaded once; all read-only)
# --------------------------------------------------------------------------- #
def _returns() -> pd.DataFrame:
    if "returns" not in _CACHE:
        _CACHE["returns"] = pd.read_parquet(PROCESSED_DIR / "returns.parquet")[UNIVERSE]
    return _CACHE["returns"]


def _prices_index() -> pd.DatetimeIndex:
    if "prices_index" not in _CACHE:
        _CACHE["prices_index"] = pd.read_parquet(PROCESSED_DIR / "prices.parquet").index
    return _CACHE["prices_index"]


def _fit_frame() -> pd.DataFrame:
    """Realised-sample universe for every OLS fit (frozen and walk-forward).

    Task 8 Stage C construction, reused: baseline.parquet's split rows carry
    vol_trail_20 (the x); the split npz files carry raw y_vol (the y) in the
    same stored row order. The per-split (ticker, window_end) identity against
    meta is asserted before the positional join -- a mismatch means the frozen
    artifacts drifted and nothing downstream can be trusted.
    """
    if "fit_frame" in _CACHE:
        return _CACHE["fit_frame"]

    base = pd.read_parquet(TASK8_DIR / "baseline.parquet")
    meta = pd.read_parquet(PROCESSED_DIR / "meta.parquet")

    out = base.copy()
    out["y_vol"] = np.nan
    for s in SPLITS:
        mask = (out["split"] == s).to_numpy()
        b = out.loc[mask]
        m = meta[meta["split"] == s]
        with np.load(PROCESSED_DIR / f"{s}.npz") as z:
            y = z["y_vol"].astype(np.float64)
        assert len(b) == len(m) == len(y), f"{s}: row counts disagree"
        assert (b["ticker"].to_numpy() == m["ticker"].to_numpy()).all() and (
            b["window_end"].to_numpy() == m["window_end"].to_numpy()
        ).all(), f"{s}: baseline.parquet rows do not match meta rows"
        out.loc[mask, "y_vol"] = y

    assert not out["y_vol"].isna().any(), "y_vol join left holes"
    _CACHE["fit_frame"] = out
    return out


# --------------------------------------------------------------------------- #
# A.1 frozen train-fit reference
# --------------------------------------------------------------------------- #
def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form simple OLS: b = S_xy / S_xx, a = ybar - b*xbar."""
    xbar, ybar = x.mean(), y.mean()
    dx = x - xbar
    b = float(np.dot(dx, y - ybar) / np.dot(dx, dx))
    return float(ybar - b * xbar), b


def train_fit() -> dict:
    """Pooled + per-ticker OLS on the train split rows (the frozen reference)."""
    tr = _fit_frame()
    tr = tr[tr["split"] == "train"]
    a, b = _ols(tr["vol_trail_20"].to_numpy(), tr["y_vol"].to_numpy())
    per = {}
    for tk in UNIVERSE:
        rows = tr[tr["ticker"] == tk]
        ai, bi = _ols(rows["vol_trail_20"].to_numpy(), rows["y_vol"].to_numpy())
        per[tk] = {"a": ai, "b": bi, "n": int(len(rows))}
    return {"pooled": {"a": a, "b": b}, "per_ticker": per}


# --------------------------------------------------------------------------- #
# A.2 on-demand trailing vol
# --------------------------------------------------------------------------- #
def vol_trail_20(t: pd.Timestamp) -> pd.Series:
    """Sample std (ddof=1) of the 20 daily returns ending at and including t.

    Raw daily units. Uses only data realised by t; ddof=1 is important
    (numpy's ddof=0 default fails Gate A2).
    """
    t = pd.Timestamp(t)
    ret = _returns()
    if t not in ret.index:
        raise ValueError(f"{t.date()} is not a trading day on the panel index")
    window = ret.loc[:t].tail(TRAIL_WINDOW)
    if len(window) != TRAIL_WINDOW or window.isna().any().any():
        raise ValueError(f"insufficient return history at {t.date()}")
    out = window.std(ddof=1)
    out.name = "vol_trail_20"
    return out


# --------------------------------------------------------------------------- #
# A.3 walk-forward refit
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _fit_coeffs_cached(t: pd.Timestamp) -> dict[str, tuple[float, float]]:
    f = _fit_frame()
    elig = f[f["vol_label_date"] <= t]
    out = {}
    for tk in UNIVERSE:
        rows = elig[elig["ticker"] == tk]
        if len(rows) < 2:
            raise ValueError(f"{tk}: {len(rows)} realised samples at {t.date()}, cannot fit")
        out[tk] = _ols(rows["vol_trail_20"].to_numpy(), rows["y_vol"].to_numpy())
    return out


def fit_coeffs(t: pd.Timestamp) -> dict[str, tuple[float, float]]:
    """Per-ticker (a, b) fit on every sample whose label is realised by t."""
    return dict(_fit_coeffs_cached(pd.Timestamp(t)))


def forecast_vol(t: pd.Timestamp) -> pd.Series:
    """Walk-forward forecast a_i(t) + b_i(t) * vol_trail_20_i(t), raw daily."""
    t = pd.Timestamp(t)
    coeffs = fit_coeffs(t)
    trail = vol_trail_20(t)
    out = pd.Series(
        {tk: coeffs[tk][0] + coeffs[tk][1] * trail[tk] for tk in UNIVERSE},
        name="forecast_vol",
    )
    return out


# --------------------------------------------------------------------------- #
# Monthly grid (Stage A private; the public pre-registered one lands in
# allocators.py at Stage C -- same rule, derived from the prices index)
# --------------------------------------------------------------------------- #
def _month_firsts() -> pd.DatetimeIndex:
    idx = _prices_index()
    firsts = idx.to_series().groupby(idx.to_period("M")).min()
    return pd.DatetimeIndex(firsts.to_numpy())


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def gate_a1(lines: list[str]) -> dict:
    fit = train_fit()
    a, b = fit["pooled"]["a"], fit["pooled"]["b"]
    got_a, got_b = f"{a:.6g}", f"{b:.6g}"
    assert got_a == GATE_A1_REF_A and got_b == GATE_A1_REF_B, (
        f"pooled fit a={got_a}, b={got_b} does not reproduce Task 8 Stage C "
        f"Gate C3 reference a={GATE_A1_REF_A}, b={GATE_A1_REF_B} -- "
        "construction drift from Stage C, STOP"
    )
    lines.append(
        f"**GATE A1 PASS** — pooled train fit reproduces the Task 8 Stage C "
        f"Gate C3 checksum to all printed digits: a={got_a} (full: {a!r}), "
        f"b={got_b} (full: {b!r}). Fit basis: 21992 train rows "
        f"(2749 per ticker), Stage C construction."
    )
    return fit


def gate_a2(lines: list[str]) -> None:
    base = pd.read_parquet(TASK8_DIR / "baseline.parquet")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(base), size=GATE_A2_N, replace=False)
    diffs = []
    for i in idx:
        row = base.iloc[i]
        got = vol_trail_20(row["window_end"])[row["ticker"]]
        diffs.append(abs(got - row["vol_trail_20"]))
    worst = max(diffs)
    assert worst < GATE_A2_TOL, (
        f"on-demand vol_trail_20 max abs diff {worst:.3e} >= {GATE_A2_TOL} "
        "vs frozen baseline.parquet -- STOP"
    )
    lines.append(
        f"**GATE A2 PASS** — on-demand vol_trail_20 vs frozen baseline.parquet "
        f"at {GATE_A2_N} random (split, window_end, ticker) points (seed {SEED}): "
        f"max abs diff {worst:.3e} < {GATE_A2_TOL:g}."
    )


def gate_a3(lines: list[str], frozen: dict) -> pd.Timestamp:
    f = _fit_frame()
    t_star = f.loc[f["split"] == "train", "vol_label_date"].max()

    elig = f[f["vol_label_date"] <= t_star]
    n_elig, n_train = len(elig), int((f["split"] == "train").sum())
    all_train = bool((elig["split"] == "train").all())

    wf = fit_coeffs(t_star)
    worst = max(
        abs(delta)
        for tk in UNIVERSE
        for delta in (
            wf[tk][0] - frozen["per_ticker"][tk]["a"],
            wf[tk][1] - frozen["per_ticker"][tk]["b"],
        )
    )
    assert worst < GATE_A3_TOL, (
        f"fit_coeffs(t*={t_star.date()}) vs frozen per-ticker coefficients: "
        f"max abs diff {worst:.3e} >= {GATE_A3_TOL} -- STOP"
    )
    lines.append(
        f"**GATE A3 PASS** — walk-forward ↔ frozen tie-out at derived "
        f"t* = {t_star.date()}: max elementwise |diff| {worst:.3e} < "
        f"{GATE_A3_TOL:g}. Eligible set at t*: {n_elig} rows, all split='train': "
        f"{all_train} (train rows: {n_train}) — exactly the frozen fit basis. "
        f"Earliest val label realises "
        f"{f.loc[f['split'] == 'val', 'vol_label_date'].min().date()}."
    )
    return t_star


def gate_a4(lines: list[str], frozen: dict) -> pd.DataFrame:
    firsts = _month_firsts()
    dates = firsts[(firsts >= CONTINUITY_START) & (firsts <= CONTINUITY_END)]
    assert len(dates) == 24, f"continuity grid has {len(dates)} dates, expected 24"

    # The single permitted read of predictions_test.parquet (prohibition 0.2).
    preds = pd.read_parquet(TASK8_DIR / "predictions_test.parquet")
    preds = preds.set_index(["ticker", "window_end"])

    f = _fit_frame()
    y_map = f[f["split"] == "test"].set_index(["ticker", "window_end"])["y_vol"]

    rows = []
    for t in dates:
        wf_coeffs = fit_coeffs(t)
        trail = vol_trail_20(t)
        wf = forecast_vol(t)
        for tk in UNIVERSE:
            fa = frozen["per_ticker"][tk]["a"]
            fb = frozen["per_ticker"][tk]["b"]
            frozen_pred = fa + fb * trail[tk]
            ref = float(preds.at[(tk, t), "pred_ols_ticker"])
            assert abs(frozen_pred - ref) < GATE_A4_XCHECK_TOL, (
                f"frozen forecast at ({tk}, {t.date()}) = {frozen_pred!r} vs "
                f"predictions_test.parquet pred_ols_ticker = {ref!r}, "
                f"|diff| {abs(frozen_pred - ref):.3e} >= {GATE_A4_XCHECK_TOL} "
                "-- deployment module does not tie back to the Task 8 artifact, STOP"
            )
            rows.append(
                {
                    "date": t,
                    "ticker": tk,
                    "vol_trail_20": trail[tk],
                    "wf_a": wf_coeffs[tk][0],
                    "wf_b": wf_coeffs[tk][1],
                    "wf_forecast": wf[tk],
                    "frozen_forecast": frozen_pred,
                    "pred_ols_ticker_ref": ref,
                    "abs_wf_minus_frozen": abs(wf[tk] - frozen_pred),
                    "y_vol": float(y_map.at[(tk, t)]),
                }
            )
    df = pd.DataFrame(rows)
    assert len(df) == 192, f"{len(df)} continuity pairs, expected 192"
    lines.append(
        f"**GATE A4 (stop-and-report, no acceptance criterion)** — continuity "
        f"check on the 24 monthly rebalance dates {dates[0].date()} .. "
        f"{dates[-1].date()} (192 pairs). Frozen forecasts tie back to "
        f"predictions_test.parquet pred_ols_ticker at all 192 points "
        f"(max |diff| {max(abs(df['frozen_forecast'] - df['pred_ols_ticker_ref'])):.3e} "
        f"< {GATE_A4_XCHECK_TOL:g}). Walk-forward vs frozen divergence and RMSE "
        f"tables follow; numbers are documented, no action taken on them."
    )
    return df


# --------------------------------------------------------------------------- #
# Continuity summary tables (report content, not gates)
# --------------------------------------------------------------------------- #
def _rmse(pred: pd.Series, actual: pd.Series) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def continuity_tables(df: pd.DataFrame, frozen: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = []
    for tk in UNIVERSE:
        g = df[df["ticker"] == tk]
        per.append(
            {
                "ticker": tk,
                "mean_abs_wf_minus_frozen": g["abs_wf_minus_frozen"].mean(),
                "max_abs_wf_minus_frozen": g["abs_wf_minus_frozen"].max(),
                "rmse_wf": _rmse(g["wf_forecast"], g["y_vol"]),
                "rmse_frozen": _rmse(g["frozen_forecast"], g["y_vol"]),
            }
        )
    per.append(
        {
            "ticker": "POOLED",
            "mean_abs_wf_minus_frozen": df["abs_wf_minus_frozen"].mean(),
            "max_abs_wf_minus_frozen": df["abs_wf_minus_frozen"].max(),
            "rmse_wf": _rmse(df["wf_forecast"], df["y_vol"]),
            "rmse_frozen": _rmse(df["frozen_forecast"], df["y_vol"]),
        }
    )

    first_t, last_t = df["date"].min(), df["date"].max()
    c_first, c_last = fit_coeffs(first_t), fit_coeffs(last_t)
    drift = pd.DataFrame(
        [
            {
                "ticker": tk,
                f"a@{first_t.date()}": c_first[tk][0],
                f"b@{first_t.date()}": c_first[tk][1],
                f"a@{last_t.date()}": c_last[tk][0],
                f"b@{last_t.date()}": c_last[tk][1],
                "a_frozen": frozen["per_ticker"][tk]["a"],
                "b_frozen": frozen["per_ticker"][tk]["b"],
            }
            for tk in UNIVERSE
        ]
    )
    return pd.DataFrame(per), drift


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.6g}") -> str:
    body = df.copy()
    for c in body.columns:
        if body[c].dtype.kind == "f":
            body[c] = body[c].map(lambda v: floatfmt.format(v))
    header = "| " + " | ".join(body.columns) + " |"
    sep = "|" + "|".join(["---"] * len(body.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in body.to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task 9 Stage A report — forecaster deployment",
        "",
        "Production forecaster: OLS-per-ticker `y_vol = a_i + b_i * vol_trail_20`, "
        "walk-forward refit, raw daily vol units.",
        "",
        "## Gates",
        "",
    ]
    try:
        frozen = gate_a1(lines)
        gate_a2(lines)
        gate_a3(lines, frozen)
        cont = gate_a4(lines, frozen)
    except AssertionError as e:
        lines += ["", f"**GATE FAILURE — STOP**: {e}", "",
                  "Diagnosis recorded."]
        report = "\n".join(lines)
        (OUT_DIR / "stage_a_report.md").write_text(report, encoding="utf-8")
        print(report)
        sys.exit(1)

    # Deliverable: frozen train-fit JSON (written only after gates A1-A3 pass).
    payload = {
        "fit_basis": (
            "train split rows, window_end 2010-01-04..2020-12-02, "
            "Task 8 Stage C construction"
        ),
        "pooled": frozen["pooled"],
        "per_ticker": frozen["per_ticker"],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(OUT_DIR / "forecaster_trainfit.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    cont_out = cont.copy()
    cont_out["date"] = cont_out["date"].dt.date
    cont_out.to_csv(OUT_DIR / "continuity_check.csv", index=False)

    # Frozen per-ticker coefficients at full precision (Task 11 anchors).
    lines += [
        "",
        "## Frozen per-ticker coefficients (full precision — Task 11 reproducibility anchors)",
        "",
        _md_table(
            pd.DataFrame(
                [
                    {"ticker": tk, "a": repr(frozen["per_ticker"][tk]["a"]),
                     "b": repr(frozen["per_ticker"][tk]["b"]),
                     "n": frozen["per_ticker"][tk]["n"]}
                    for tk in UNIVERSE
                ]
            )
        ),
        "",
        f"Pooled (full precision): a={frozen['pooled']['a']!r}, "
        f"b={frozen['pooled']['b']!r}.",
    ]

    per, drift = continuity_tables(cont, frozen)
    lines += [
        "",
        "## Continuity check (A.4): walk-forward vs frozen, 2023-01-03 .. 2024-12-02",
        "",
        "Per ticker and pooled — |walk-forward − frozen| and RMSE vs realised "
        "y_vol (192 pairs, all labels realised):",
        "",
        _md_table(per),
        "",
        "Coefficient drift (walk-forward fits at the first and last continuity "
        "dates vs the frozen train fit):",
        "",
        _md_table(drift),
        "",
        "Narrative: the walk-forward coefficients at 2023-01-03 already include "
        "the 2021-22 val-period samples (labels realised well before 2023), so "
        "they differ from the frozen train fit by construction; through 2024 "
        "they continue to absorb realised test-period samples. The divergence "
        "documented above is the small, deliberate difference between the "
        "deployment mode (expanding walk-forward refit) and Task 8's fixed "
        "train fit. Per Gate A4 these numbers are recorded, not acted on.",
        "",
        "## Interpretive notes (recorded decisions)",
        "",
        "- Fit universe = split rows (train/val/test) of the Task 8 "
        "construction; the 896 split='drop' rows (pre-2010 warm-up, boundary "
        "straddles) carry no y_vol in any frozen artifact and never enter a "
        "fit. 576 of them have vol_label_date <= t*; including them would "
        "break the Gate A3 tie-out identity, confirming the exclusion matches "
        "the intent.",
        "- Gate A2 sampling: 30 row indices drawn without replacement from the "
        "29712 split rows of baseline.parquet via numpy default_rng(42).",
        "- Walk-forward fits and the frozen fit share one closed-form OLS code "
        "path and one row construction, so Gate A3 is an exact-identity check. "
        "",
        "## Deliverables",
        "",
        "- `src/task9/forecaster_deploy.py`",
        "- `outputs/task9/forecaster_trainfit.json`",
        "- `outputs/task9/continuity_check.csv` (192 rows)",
        "- `outputs/task9/stage_a_report.md`",
        "",
        "**STOP — Stage A boundary.** Gates A1–A4 reported above."
    ]

    report = "\n".join(lines)
    (OUT_DIR / "stage_a_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
