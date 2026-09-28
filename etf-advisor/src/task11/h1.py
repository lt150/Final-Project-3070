"""Hypothesis H1 — registered quantities and verdict.

H1, registered before any backtest existed: because the deployed
regression shrinks trailing-vol extremes (per-ticker b ∈ [0.26, 0.57]),
``rb_advisor`` is hypothesised to be the SMOOTHER allocator than
``rb_classical`` — smaller allocation shifts, lower turnover, slower crash
response. Either outcome is reported.

Every quantity compares rb_advisor against rb_classical *within profile*, on
target-vs-drift weights at grid dates, and is therefore cost-invariant.

  H1-Q1  turnover        — mean traded fraction T over the 155 post-inception
                           grid dates. Prediction: advisor < classical in all
                           three profiles.
  H1-Q2  allocation shift— mean over grid dates of max_i |w_target,i −
                           w_drift,i| (largest single-asset shift), plus the
                           full-period max of the same quantity. Prediction:
                           both smaller for the advisor. Not redundant with Q1:
                           low total turnover can coexist with occasional
                           violent single-asset moves.
  H1-Q3  crash response  — equity-sleeve target weight E at the Feb, Mar, Apr
                           2020 grid dates. Registered prediction:
                           |E(Mar) - E(Feb)| smaller for the advisor. The
                           Feb to Apr leg is DESCRIPTIVE only (recovery dynamics
                           confound its direction) and is not a prediction.

Verdict: evaluated on the **balanced** profile as headline, with conservative
and growth as robustness rows; ∈ {supported, partially supported (naming which
components), not supported} by strict inequality direction. No statistical test
— these are deterministic functionals of a single realised path. H1 and
portfolio performance are separate claims (smoothness != Sharpe) and are never
traded off against each other.

Grid-date set for Q1 and Q2: the 155 post-inception dates. Inception is
excluded for Q1; the same exclusion is applied to Q2 because at
inception the drifted vector is the zero vector (funding from cash), so
max_i |w_target,i − 0| would report the largest target weight — a funding
artifact, not an allocation shift. Disclosed in the Stage C report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EQUITY_SLEEVE = ("SPY", "QQQ", "EFA", "EEM")

Q3_LABELS = ("Feb", "Mar", "Apr")


def equity_sleeve(target_weights: pd.DataFrame) -> pd.Series:
    """E(t) = w_SPY + w_QQQ + w_EFA + w_EEM at target weights (§1.7)."""
    return target_weights[list(EQUITY_SLEEVE)].sum(axis=1)


def max_single_asset_shift(result) -> pd.Series:
    """max_i |w_target,i − w_drift,i| at each grid date."""
    return (result.target_weights - result.drift_weights).abs().max(axis=1)


def h1_for_profile(
    res_advisor,
    res_classical,
    post_inception: pd.DatetimeIndex,
    q3_dates: dict[str, pd.Timestamp],
) -> dict[str, float]:
    """All registered H1 quantities for one profile."""
    out: dict[str, float] = {}

    # --- Q1 -----------------------------------------------------------------
    for name, res in (("advisor", res_advisor), ("classical", res_classical)):
        out[f"q1_mean_turnover_{name}"] = float(res.turnover.loc[post_inception].mean())
    out["q1_delta_advisor_minus_classical"] = (
        out["q1_mean_turnover_advisor"] - out["q1_mean_turnover_classical"]
    )
    out["q1_advisor_smoother"] = float(
        out["q1_mean_turnover_advisor"] < out["q1_mean_turnover_classical"]
    )

    # --- Q2 -----------------------------------------------------------------
    for name, res in (("advisor", res_advisor), ("classical", res_classical)):
        shift = max_single_asset_shift(res).loc[post_inception]
        out[f"q2_mean_max_shift_{name}"] = float(shift.mean())
        out[f"q2_max_max_shift_{name}"] = float(shift.max())
    out["q2_mean_delta_advisor_minus_classical"] = (
        out["q2_mean_max_shift_advisor"] - out["q2_mean_max_shift_classical"]
    )
    out["q2_max_delta_advisor_minus_classical"] = (
        out["q2_max_max_shift_advisor"] - out["q2_max_max_shift_classical"]
    )
    out["q2_mean_advisor_smoother"] = float(
        out["q2_mean_max_shift_advisor"] < out["q2_mean_max_shift_classical"]
    )
    out["q2_max_advisor_smoother"] = float(
        out["q2_max_max_shift_advisor"] < out["q2_max_max_shift_classical"]
    )

    # --- Q3 -----------------------------------------------------------------
    for name, res in (("advisor", res_advisor), ("classical", res_classical)):
        e = equity_sleeve(res.target_weights)
        for label in Q3_LABELS:
            out[f"q3_E_{label}_{name}"] = float(e.loc[q3_dates[label]])
        out[f"q3_abs_dE_feb_to_mar_{name}"] = abs(
            out[f"q3_E_Mar_{name}"] - out[f"q3_E_Feb_{name}"]
        )
        out[f"q3_dE_feb_to_apr_{name}"] = (
            out[f"q3_E_Apr_{name}"] - out[f"q3_E_Feb_{name}"]
        )
    out["q3_delta_advisor_minus_classical"] = (
        out["q3_abs_dE_feb_to_mar_advisor"] - out["q3_abs_dE_feb_to_mar_classical"]
    )
    out["q3_advisor_smoother"] = float(
        out["q3_abs_dE_feb_to_mar_advisor"] < out["q3_abs_dE_feb_to_mar_classical"]
    )
    return out


# Registered components of the verdict.
COMPONENTS = (
    ("H1-Q1 turnover", "q1_advisor_smoother"),
    ("H1-Q2 mean allocation shift", "q2_mean_advisor_smoother"),
    ("H1-Q2 max allocation shift", "q2_max_advisor_smoother"),
    ("H1-Q3 crash response", "q3_advisor_smoother"),
)


def verdict(headline: dict[str, float]) -> tuple[str, list[str], list[str]]:
    """Verdict on the balanced profile by strict inequality direction."""
    held = [label for label, key in COMPONENTS if headline[key] == 1.0]
    failed = [label for label, key in COMPONENTS if headline[key] != 1.0]
    if not failed:
        return "supported", held, failed
    if not held:
        return "not supported", held, failed
    return "partially supported", held, failed


def verdict_sentence(
    v: str, held: list[str], failed: list[str], profile: str = "balanced"
) -> str:
    """The close-out sentence minted for downstream tasks."""
    if v == "supported":
        return (
            f"H1 is **supported** on the {profile} profile: rb_advisor is the "
            f"smoother allocator on every registered component "
            f"({', '.join(held)})."
        )
    if v == "not supported":
        return (
            f"H1 is **not supported** on the {profile} profile: rb_advisor is "
            f"not the smoother allocator on any registered component "
            f"({', '.join(failed)} all run the other way)."
        )
    return (
        f"H1 is **partially supported** on the {profile} profile: rb_advisor is "
        f"the smoother allocator on {', '.join(held)}, but not on "
        f"{', '.join(failed)}."
    )


def q3_dates_from_grid(grid: pd.DatetimeIndex) -> dict[str, pd.Timestamp]:
    """The Feb/Mar/Apr 2020 grid dates named in Stage A (Gate A4)."""
    out = {}
    for label, month in zip(Q3_LABELS, (2, 3, 4)):
        sel = grid[(grid.year == 2020) & (grid.month == month)]
        if len(sel) != 1:
            raise ValueError(f"{label} 2020 has {len(sel)} grid dates, expected 1")
        out[label] = sel[0]
    return out


def h1_frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """profile -> quantity dict, flattened for CSV (full precision)."""
    return pd.DataFrame(rows).T.rename_axis("profile").reset_index()


def h1_report_table(rows: dict[str, dict[str, float]], dp: int = 3) -> pd.DataFrame:
    """T3 — Q1/Q2/Q3 for all three profiles, advisor vs classical with deltas.

    Rounded to the registered 3 dp for H1 weight deltas.
    """
    fmt = lambda x: f"{x:.{dp}f}"
    recs = []
    for profile, r in rows.items():
        recs.extend([
            {"profile": profile, "quantity": "Q1 mean turnover T̄",
             "rb_advisor": fmt(r["q1_mean_turnover_advisor"]),
             "rb_classical": fmt(r["q1_mean_turnover_classical"]),
             "advisor − classical": fmt(r["q1_delta_advisor_minus_classical"]),
             "advisor smoother": "yes" if r["q1_advisor_smoother"] else "no"},
            {"profile": profile, "quantity": "Q2 mean max single-asset shift",
             "rb_advisor": fmt(r["q2_mean_max_shift_advisor"]),
             "rb_classical": fmt(r["q2_mean_max_shift_classical"]),
             "advisor − classical": fmt(r["q2_mean_delta_advisor_minus_classical"]),
             "advisor smoother": "yes" if r["q2_mean_advisor_smoother"] else "no"},
            {"profile": profile, "quantity": "Q2 full-period max shift",
             "rb_advisor": fmt(r["q2_max_max_shift_advisor"]),
             "rb_classical": fmt(r["q2_max_max_shift_classical"]),
             "advisor − classical": fmt(r["q2_max_delta_advisor_minus_classical"]),
             "advisor smoother": "yes" if r["q2_max_advisor_smoother"] else "no"},
            {"profile": profile, "quantity": "Q3 abs ΔE(Feb→Mar) 2020",
             "rb_advisor": fmt(r["q3_abs_dE_feb_to_mar_advisor"]),
             "rb_classical": fmt(r["q3_abs_dE_feb_to_mar_classical"]),
             "advisor − classical": fmt(r["q3_delta_advisor_minus_classical"]),
             "advisor smoother": "yes" if r["q3_advisor_smoother"] else "no"},
        ])
    return pd.DataFrame(recs)


def q3_detail_table(rows: dict[str, dict[str, float]], dp: int = 3) -> pd.DataFrame:
    """Equity-sleeve levels behind Q3, plus the descriptive Feb to Apr leg."""
    fmt = lambda x: f"{x:.{dp}f}"  # noqa: E731
    recs = []
    for profile, r in rows.items():
        for name in ("advisor", "classical"):
            recs.append({
                "profile": profile, "allocator": f"rb_{name}",
                "E(Feb)": fmt(r[f"q3_E_Feb_{name}"]),
                "E(Mar)": fmt(r[f"q3_E_Mar_{name}"]),
                "E(Apr)": fmt(r[f"q3_E_Apr_{name}"]),
                "abs ΔE Feb→Mar (registered)": fmt(r[f"q3_abs_dE_feb_to_mar_{name}"]),
                "E(Apr) − E(Feb) (descriptive)": fmt(r[f"q3_dE_feb_to_apr_{name}"]),
            })
    return pd.DataFrame(recs)


__all__ = [
    "COMPONENTS", "EQUITY_SLEEVE", "Q3_LABELS", "equity_sleeve", "h1_frame",
    "h1_for_profile", "h1_report_table", "max_single_asset_shift",
    "q3_dates_from_grid", "q3_detail_table", "verdict", "verdict_sentence",
]
