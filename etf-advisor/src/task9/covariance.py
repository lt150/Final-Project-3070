"""Task 9 Stage B: shared covariance module for both allocator variants.

Sigma = D * R * D in raw daily variance units, where R is one shared
correlation matrix and D is a diagonal of vols:

  * ``corr_matrix(t)``        -- Pearson correlation of daily simple returns
                                 over the 250 trading days ending at and
                                 including t (rows [t-249 .. t], full
                                 8-ticker panel, no pairwise exclusions).
  * ``sigma(t, "advisor")``   -- D = walk-forward forecast vols (Stage A
                                 ``forecast_vol``).
  * ``sigma(t, "classical")`` -- D = trailing vols (Stage A ``vol_trail_20``).

R is identical for both variants, so any advisor edge enters only through D
-- the clean ablation the design pre-registers. The 250-day window is the
pre-registered headline constant: 8 assets carry 28 free
correlation parameters, short windows are noise-dominated, and crisis
responsiveness lives in D (20-day reactive), not R. It is deliberately NOT a
parameter of any public function; the {60, 120} sensitivity values belong to
the Task 11 robustness appendix only.

Gate B1 (four smoke dates): Sigma symmetric to < 1e-14 and min eigenvalue
> 1e-10 for both variants -- on violation STOP, with no silent ridge,
shrinkage, or clipping.

Gate B2 records full-precision anchors at 2024-12-02 for Task 11 reproducibility.
On any gatefailure: STOP, diagnosis to the stage report.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd

from ..data.config import ROOT, UNIVERSE
from .forecaster_deploy import _returns, forecast_vol, vol_trail_20

OUT_DIR = ROOT / "outputs" / "task9"

CORR_WINDOW = 250  # pre-registered headline constant; not parameterised
SMOKE_DATES = ("2012-02-01", "2020-03-02", "2022-06-01", "2024-12-02")
GATE_B2_DATE = "2024-12-02"

GATE_B1_SYM_TOL = 1e-14
GATE_B1_EIG_MIN = 1e-10

VARIANTS = ("advisor", "classical")


# --------------------------------------------------------------------------- #
# B.1 correlation matrix
# --------------------------------------------------------------------------- #
def corr_matrix(t: pd.Timestamp) -> pd.DataFrame:
    """Pearson correlation over the 250 trading days ending at and including t.

    Return rows [t-249 .. t] of the full 8-ticker panel; uses only data
    realised by t.
    """
    t = pd.Timestamp(t)
    ret = _returns()
    if t not in ret.index:
        raise ValueError(f"{t.date()} is not a trading day on the panel index")
    window = ret.loc[:t].tail(CORR_WINDOW)
    if len(window) != CORR_WINDOW or window.isna().any().any():
        raise ValueError(f"insufficient return history for R at {t.date()}")
    return window.corr()


# --------------------------------------------------------------------------- #
# B.2 the two Sigma variants
# --------------------------------------------------------------------------- #
def sigma(t: pd.Timestamp, variant: Literal["advisor", "classical"]) -> pd.DataFrame:
    """Sigma = D * R * D, raw daily variance; only D differs between variants."""
    t = pd.Timestamp(t)
    if variant == "advisor":
        d = forecast_vol(t)
    elif variant == "classical":
        d = vol_trail_20(t)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    R = corr_matrix(t)
    d = d.reindex(R.index)
    assert not d.isna().any(), "vol vector does not cover the panel tickers"
    # Elementwise R_ij * d_i * d_j == D R D, and preserves R's exact symmetry.
    return R * np.outer(d.to_numpy(), d.to_numpy())


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def gate_b1(lines: list[str]) -> None:
    rows = []
    for date in SMOKE_DATES:
        for variant in VARIANTS:
            S = sigma(date, variant).to_numpy()
            asym = float(np.abs(S - S.T).max())
            min_eig = float(np.linalg.eigvalsh(S).min())
            assert asym < GATE_B1_SYM_TOL, (
                f"Sigma_{variant}({date}) max asymmetry {asym:.3e} >= "
                f"{GATE_B1_SYM_TOL} -- STOP"
            )
            assert min_eig > GATE_B1_EIG_MIN, (
                f"Sigma_{variant}({date}) min eigenvalue {min_eig:.3e} <= "
                f"{GATE_B1_EIG_MIN} -- STOP (no silent ridge/shrinkage/clipping)"
            )
            rows.append(
                {"date": date, "variant": variant,
                 "max_asymmetry": f"{asym:.3e}", "min_eigenvalue": f"{min_eig:.6e}"}
            )
    table = pd.DataFrame(rows)
    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join(["---"] * len(table.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in table.to_numpy()]
    lines += [
        f"**GATE B1 PASS** — at all four smoke dates, both variants: max "
        f"asymmetry < {GATE_B1_SYM_TOL:g}, min eigenvalue > {GATE_B1_EIG_MIN:g}. "
        "No regularisation applied anywhere.",
        "",
        header, sep, *body,
    ]


def gate_b2(lines: list[str]) -> None:
    t = pd.Timestamp(GATE_B2_DATE)
    S_cl = sigma(t, "classical")
    R = corr_matrix(t)
    trace = float(np.trace(S_cl.to_numpy()))
    spy_agg = float(S_cl.at["SPY", "AGG"])
    r_spy_qqq = float(R.at["SPY", "QQQ"])
    lines += [
        "",
        f"**GATE B2 (checksum record)** — anchors at t = {GATE_B2_DATE}, full "
        "precision, recorded (no prior reference exists); fixed reproducibility "
        "anchors for Task 11:",
        "",
        f"- trace(Sigma_classical) = `{trace!r}`",
        f"- Sigma_classical[SPY, AGG] = `{spy_agg!r}`",
        f"- R[SPY, QQQ] = `{r_spy_qqq!r}`",
    ]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task 9 Stage B report — shared covariance module",
        "",
        f"Sigma = D·R·D, raw daily variance. R: Pearson correlation over the "
        f"{CORR_WINDOW} trading days ending at and including t (pre-registered "
        "headline constant, not parameterised; {60, 120} sensitivity is Task 11 "
        "appendix only). D_advisor = walk-forward forecast vols (Stage A); "
        "D_classical = trailing vol_trail_20. R identical for both variants — "
        "any advisor edge enters only through D.",
        "",
        "## Gates",
        "",
    ]
    try:
        gate_b1(lines)
        gate_b2(lines)
    except AssertionError as e:
        lines += ["", f"**GATE FAILURE — STOP**: {e}", "",
                  "Diagnosis recorded."]
        report = "\n".join(lines)
        (OUT_DIR / "stage_b_report.md").write_text(report, encoding="utf-8")
        print(report)
        sys.exit(1)

    lines += [
        "",
        "## Notes",
        "",
        "- Sigma is assembled as R_ij * d_i * d_j (elementwise outer-product "
        "scaling), which preserves R's exact symmetry; eigenvalues via "
        "`numpy.linalg.eigvalsh`.",
        "- Both variants share one `corr_matrix(t)` code path and the Stage A "
        "vol functions; no covariance-specific data loading exists.",
        "- Units: raw daily variance throughout; nothing is annualised.",
        "",
        "## Deliverables",
        "",
        "- `src/task9/covariance.py`",
        "- `outputs/task9/stage_b_report.md`",
        "",
        "**STOP — Stage B boundary.** Gates B1–B2 reported above."
    ]
    report = "\n".join(lines)
    (OUT_DIR / "stage_b_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
