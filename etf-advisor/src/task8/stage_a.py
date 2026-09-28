"""Task 8 Stage A: verify Task 7's frozen artifacts, build the persistence
baseline, and score it on train and val.

Pre-registered protocol:
  1. npz shapes / meta split counts / window_end ranges match the Task 7
     contract exactly, and meta's stored row order is (window_end, ticker)
     within each split (the npz row order).
  2. y_vol reconstruction: for 50 random samples per split,
     recompute std(ret[t+1 .. t+20], ddof=1) from returns.parquet via meta
     and compare to the stored y_vol at the same row position. This checks
     the label definition AND the meta<->npz row alignment at once.
  3a. Audit-basis reproduction (independent cross-check): corr(trailing 20d
     vol, forward 20d vol) on the EDA notebook's exact basis — SPY only,
     full range, off the returns panel — must equal 0.5310 to |diff| < 1e-3.
  3b. Pooled-train checksum (reproducibility, not expectation): the pooled
     corr(vol_trail_20, y_vol) on train rows must match the value pinned on
     the first Stage A run to |diff| < 1e-6. On frozen artifacts with
     deterministic construction, drift beyond numerical noise means the
     construction changed, and the gate fails.

Gate 3 history, before any model training:
the original gate expected the pooled-train corr to be ~0.531 +/- 0.05, but
that anchor was mis-derived — 0.531 is the SPY-only, full-range figure from
notebooks/01_data_eda.ipynb cell 18, not a pooled figure. Pooling across the
8 tickers' persistent vol levels (train mean y_vol spans AGG 0.0021 to EEM
0.0128) inflates the pooled corr to 0.6596, above every within-ticker corr
(AGG 0.2600 ... VNQ 0.5698). The amendment splits the gate into 3a + 3b
above; the +/-0.05 band is removed.

"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..data.config import PROCESSED_DIR, ROOT, SEED
from .baseline import SPLITS, build_vol_trail_20
from .metrics import basic_metrics, decile_table, per_ticker_table

OUT_DIR = ROOT / "outputs" / "task8"

EXPECTED_SHAPES = {"train": 21992, "val": 3864, "test": 3856}
EXPECTED_META_TOTAL = 30608
EXPECTED_WINDOW_END = {  # first / last window_end per split
    "train": ("2010-01-04", "2020-12-02"),
    "val": ("2021-01-04", "2022-12-01"),
    "test": ("2023-01-03", "2024-12-02"),
}
# Gate 3a: the notebook audit figure (SPY only, full range) and its band.
AUDIT_SPY_CORR = 0.5310
AUDIT_SPY_TOL = 1e-3
# Gate 3b: pooled-train corr pinned at full precision on the first Stage A
#A checksum, not an expectation
POOLED_TRAIN_CORR_PIN = 0.6596426015679578
POOLED_TRAIN_CORR_TOL = 1e-6
N_SPOT_CHECKS = 50
VOL_HORIZON = 20
TRAIL_WINDOW = 20


def gate_1_contract(npz: dict, meta: pd.DataFrame, lines: list[str]) -> None:
    assert len(meta) == EXPECTED_META_TOTAL, f"meta rows {len(meta)}"
    counts = meta["split"].value_counts()
    for split, n in EXPECTED_SHAPES.items():
        d = npz[split]
        assert d["X"].shape == (n, 30, 6), f"{split} X {d['X'].shape}"
        assert d["y_ret"].shape == (n,), f"{split} y_ret {d['y_ret'].shape}"
        assert d["y_vol"].shape == (n,), f"{split} y_vol {d['y_vol'].shape}"
        assert counts[split] == n, f"meta {split} count {counts[split]} != {n}"

        m = meta[meta["split"] == split]
        first, last = m["window_end"].iloc[0], m["window_end"].iloc[-1]
        exp_first, exp_last = EXPECTED_WINDOW_END[split]
        assert first == pd.Timestamp(exp_first), f"{split} first window_end {first}"
        assert last == pd.Timestamp(exp_last), f"{split} last window_end {last}"

        # stored order must be the npz row order: window_end, then ticker
        resorted = m.sort_values(["window_end", "ticker"], kind="mergesort")
        assert (
            resorted.index == m.index
        ).all(), f"{split} meta rows not in (window_end, ticker) order"

    assert counts["drop"] == EXPECTED_META_TOTAL - sum(EXPECTED_SHAPES.values())
    lines.append(
        "GATE 1 PASS — shapes, split counts, window_end ranges, and row order "
        "all match the Task 7 contract"
    )


def gate_2_yvol_reconstruction(
    npz: dict, meta: pd.DataFrame, returns: pd.DataFrame, lines: list[str]
) -> None:
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for split in SPLITS:
        m = meta[meta["split"] == split].reset_index(drop=True)
        y_vol = npz[split]["y_vol"]
        idx = rng.choice(len(m), size=N_SPOT_CHECKS, replace=False)
        for i in idx:
            ticker = m.at[i, "ticker"]
            t = m.at[i, "window_end"]
            ret = returns[ticker]
            pos = ret.index.get_loc(t)
            recon = ret.iloc[pos + 1 : pos + 1 + VOL_HORIZON].std(ddof=1)
            diff = abs(float(recon) - float(y_vol[i]))
            worst = max(worst, diff)
            assert diff < 1e-7, (
                f"y_vol mismatch {split} row {i} ({ticker} @ {t.date()}): "
                f"stored {y_vol[i]:.8g} vs recomputed {recon:.8g}"
            )
    lines.append(
        f"GATE 2 PASS — y_vol reconstruction, {N_SPOT_CHECKS} random samples "
        f"per split (seed {SEED}); max |diff| = {worst:.3g}"
    )


def gate_3a_audit_basis(returns: pd.DataFrame, lines: list[str]) -> None:
    """Reproduce the EDA notebook's figure on its exact basis.

    SPY only, full range, straight off the returns panel — the basis on
    which 0.531 was actually computed (01_data_eda.ipynb cell 18). This is
    an independent cross-check that our trailing-vol construction matches
    the Task 7 audit's.
    """
    trail = returns["SPY"].rolling(TRAIL_WINDOW).std(ddof=1)
    fwd = trail.shift(-VOL_HORIZON)  # std(ret[t+1 .. t+20]) indexed at t
    corr = float(fwd.corr(trail))
    diff = abs(corr - AUDIT_SPY_CORR)
    assert diff < AUDIT_SPY_TOL, (
        f"SPY audit-basis corr = {corr:.6f}, |diff| {diff:.2g} >= "
        f"{AUDIT_SPY_TOL} from the notebook figure {AUDIT_SPY_CORR}"
    )
    lines.append(
        f"GATE 3a PASS — SPY-only full-range corr(trail20, fwd20) = "
        f"{corr:.4f} reproduces the notebook audit figure {AUDIT_SPY_CORR} "
        f"(|diff| < {AUDIT_SPY_TOL:g})"
    )


def gate_3b_pooled_checksum(
    baseline: pd.DataFrame, npz: dict, lines: list[str]
) -> None:
    """Pooled-train corr must match the pinned first-run value exactly.

    Reproducibility checksum, not an expectation: on frozen artifacts any
    drift beyond numerical noise means the construction changed.
    """
    train = baseline[baseline["split"] == "train"]
    corr = float(
        np.corrcoef(train["vol_trail_20"], npz["train"]["y_vol"])[0, 1]
    )
    diff = abs(corr - POOLED_TRAIN_CORR_PIN)
    assert diff < POOLED_TRAIN_CORR_TOL, (
        f"pooled-train corr(vol_trail_20, y_vol) = {corr!r} drifted "
        f"{diff:.2g} >= {POOLED_TRAIN_CORR_TOL} from the pinned "
        f"{POOLED_TRAIN_CORR_PIN!r} — construction changed, investigate"
    )
    lines.append(
        f"GATE 3b PASS — pooled-train corr(vol_trail_20, y_vol) = {corr:.4f} "
        f"matches the pinned checksum {POOLED_TRAIN_CORR_PIN!r} "
        f"(|diff| < {POOLED_TRAIN_CORR_TOL:g})"
    )


def fmt_df(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.6f}")


def main() -> None:
    npz = {s: np.load(PROCESSED_DIR / f"{s}.npz") for s in SPLITS}
    meta = pd.read_parquet(PROCESSED_DIR / "meta.parquet")
    returns = pd.read_parquet(PROCESSED_DIR / "returns.parquet")

    lines: list[str] = ["Task 8 Stage A — artifact gates + persistence baseline", ""]

    gate_1_contract(npz, meta, lines)
    gate_2_yvol_reconstruction(npz, meta, returns, lines)
    gate_3a_audit_basis(returns, lines)

    baseline = build_vol_trail_20(returns, meta)
    gate_3b_pooled_checksum(baseline, npz, lines)

    lines += [
        "",
        "Persistence-strength characterisation (per Gate 3 amendment):",
        "0.531 is the SPY-only full-range audit figure, NOT a",
        "pooled one. The honest pooled-train figure is 0.6596; pooling",
        "across the tickers' persistent vol levels inflates it above every",
        "within-ticker corr — per-ticker train corr(vol_trail_20, y_vol):",
    ]
    tr = baseline[baseline["split"] == "train"].copy()
    tr["y_vol"] = npz["train"]["y_vol"].astype(np.float64)
    per_corr = tr.groupby("ticker").apply(
        lambda g: g["vol_trail_20"].corr(g["y_vol"]), include_groups=False
    )
    lines.append(
        "  " + " | ".join(f"{t} {c:.4f}" for t, c in per_corr.items())
    )

    # persistence metrics on train and val only (test untouched until Stage C)
    for split in ("train", "val"):
        b = baseline[baseline["split"] == split]
        pred = b["vol_trail_20"].to_numpy()
        actual = npz[split]["y_vol"].astype(np.float64)

        m = basic_metrics(pred, actual)
        lines += [
            "",
            f"— persistence on {split} (raw daily vol units) —",
            f"  RMSE {m['rmse']:.6f} | MAE {m['mae']:.6f} | "
            f"Pearson r {m['pearson_r']:.4f} | std ratio {m['std_ratio']:.4f}",
            "",
            fmt_df(per_ticker_table(pred, actual, b["ticker"])),
            "",
            "  decile calibration (by predicted decile):",
            fmt_df(decile_table(pred, actual)),
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # freeze the baseline join: Stage B reuses this exact series as the
    # persistence forecast and as T3's denominator
    baseline.to_parquet(OUT_DIR / "baseline.parquet", index=False)
    lines += ["", f"baseline frame frozen to {OUT_DIR / 'baseline.parquet'}"]

    report = "\n".join(lines)
    (OUT_DIR / "stage_a_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"STAGE A GATE FAILURE — STOP: {e}", file=sys.stderr)
        sys.exit(1)
