"""Task 11 Stage C: the one-shot headline run.

The run is one-shot executed once and reported regardless of outcome. 
Only mechanical completion gates are permitted here, because they are
accounting rather than performance:

  GATE C-M1  all 10 paths complete: defined every trading day in span, no NaN.
  GATE C-M2  full-run re-assertion of B3, B4, B5 across all 156 grid dates x 10
             portfolios. Re-implemented here rather than imported from
             ``stage_b``, so "re-assertion" means a second statement of the
             identity, not a second call to the same code.

``run_stage_c`` is parameterised on the output directory and on the weight-
function factory purely so the machinery could be smoke-tested end to end with
stub allocators, over the full span, writing to a scratch directory, before the
registered run was executed. That smoke test computes no headline quantity: the
stubs are not the Task 9 allocators. ``main()`` runs the registered
configuration — real allocators, ``outputs/task11/`` — exactly once.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..data.config import ROOT, UNIVERSE
from ..task9.allocators import rebalance_grid
from . import h1 as H1
from .engine import COST_RATE, run_backtest
from .exhibits import (
    HINDSIGHT,
    crash_sleeve_figure,
    drawdown_figure,
    equity_curve_figure,
    weights_stack_figure,
)
from .fixtures import allocator_weight_fn, load_prices, portfolio_key
from .metrics import compute_metrics, counted_grid_dates

OUT_DIR = ROOT / "outputs" / "task11"

# --------------------------------------------------------------------------- #
# Registered constants
# --------------------------------------------------------------------------- #
SPAN_START, SPAN_END = "2012-01-03", "2024-12-31"
PROFILES = ("conservative", "balanced", "growth")
HEADLINE_PROFILE = "balanced"

PORTFOLIOS: tuple[tuple[str, str | None], ...] = (
    ("one_over_n", None),
    *[(aid, p) for aid in ("static_bucket", "rb_classical", "rb_advisor") for p in PROFILES],
)

SUBPERIODS = (
    ("COVID crash", "2020-02-01", "2020-03-31", True),
    ("2022 drawdown", "2022-01-01", "2022-12-31", False),
    ("2023–24", "2023-01-01", "2024-12-31", False),
)

F4_WINDOW = ("2019-12-01", "2020-06-30")

REGISTERED_CSVS = (
    "value_paths_daily.csv",
    "grid_weights_target.csv",
    "grid_weights_drift.csv",
    "rebalance_turnover_costs.csv",
    "metrics_full_period.csv",
    "metrics_subperiods.csv",
    "h1_quantities.csv",
)
REGISTERED_FIGURES = (
    "f1_equity_balanced.png",
    "f2_drawdown_balanced.png",
    "f3_weights_rb_balanced.png",
    "f4_crash_equity_sleeve.png",
)
APPENDIX_FIGURES = (
    "a1_equity_conservative.png",
    "a2_equity_growth.png",
)

TOL_M2_COST = 1e-15
TOL_M2_WEIGHT = 1e-12

# --------------------------------------------------------------------------- #
# Registered disclosure sentences
# --------------------------------------------------------------------------- #
DISCLOSURE_EXECUTION = (
    "Rebalancing trades are assumed executed at the closing prices that "
    "terminate the estimation window — a market-on-close idealisation; no "
    "return earned by any weight vector precedes the data used to compute it."
)
DISCLOSURE_RF = (
    "Sharpe uses rf = 0 and Sortino MAR = 0; no risk-free series exists in the "
    "pre-registered universe and adding one post hoc would be a universe change."
)
DISCLOSURE_INCEPTION = (
    "At the first grid date (2012-01-03) every portfolio is funded from cash: "
    "the traded fraction is T = 1.0 exactly, the cost is one-way 10 bps, and "
    "V_post(2012-01-03) = 0.999 identically for all ten portfolios. The "
    "inception date is excluded from all per-rebalance turnover statistics — it "
    "measures funding, not rebalancing behaviour — leaving 155 turnover "
    "observations per portfolio."
)
DISCLOSURE_COST_SYMMETRY = (
    "Every portfolio, including 1/N and static_bucket, trades to its target on "
    "the same monthly grid with the same traded fraction and the same 10 bps: "
    "1/N must trade to remain 1/N."
)


def _md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def _pct(x: float, dp: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{dp}f}%"


def _num(x: float, dp: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}"


# --------------------------------------------------------------------------- #
# Pre-flight (mechanical)
# --------------------------------------------------------------------------- #
def preflight(lines: list[str]) -> None:
    for stage, markers in (
        ("a", [f"GATE A{i} PASS" for i in (1, 2, 3, 4)]),
        ("b", [f"GATE B{i} PASS" for i in (1, 2, 3, 4, 5, 6)]),
    ):
        p = OUT_DIR / f"stage_{stage}_report.md"
        assert p.exists(), f"pre-flight: {p} missing -- Stage C may not run, STOP"
        text = p.read_text(encoding="utf-8")
        assert "GATE FAILURE" not in text, (
            f"pre-flight: {p.name} records a gate failure -- STOP"
        )
        missing = [m for m in markers if m not in text]
        assert not missing, f"pre-flight: {p.name} missing {missing} -- STOP"

    assert len(PORTFOLIOS) == 10, f"pre-flight: {len(PORTFOLIOS)} portfolios, registered 10"
    assert len(REGISTERED_CSVS) == 7, "pre-flight: registered CSV set is not the exact set"
    assert len(REGISTERED_FIGURES) == 4, "pre-flight: registered body figure set is not F1-F4"
    lines.extend([
        "**Pre-flight PASS (mechanical)** — `stage_a_report.md` records A1-A4 "
        "PASS, `stage_b_report.md` records B1-B6 PASS, neither contains a gate "
        "failure; the exhibit registration in this module is unchanged "
        "(7 CSVs, 4 body figures, 2 appendix figures, 10 portfolios).",
        "",
    ])


# --------------------------------------------------------------------------- #
# Mechanical completion gates
# --------------------------------------------------------------------------- #
def gate_cm1(lines: list[str], results: dict, days: pd.DatetimeIndex) -> None:
    bad = []
    for key, r in results.items():
        if not r.values.index.equals(days) or r.values.isna().any():
            bad.append(key)
    assert not bad, f"GATE C-M1: incomplete or NaN value paths for {bad} -- STOP"
    lines.extend([
        f"**GATE C-M1 PASS** — all {len(results)} value paths defined on every "
        f"one of the {len(days)} trading days in {SPAN_START} .. {SPAN_END}, no NaN.",
        "",
    ])


def gate_cm2(lines: list[str], results: dict, prices: pd.DataFrame,
             grid: pd.DatetimeIndex) -> None:
    worst_cost = worst_w = 0.0
    t_min = np.inf
    nan_total = 0
    for r in results.values():
        # B3 re-asserted: charged cost is the rate times the traded fraction.
        worst_cost = max(worst_cost, float(np.abs(r.cost_rate * r.turnover - r.costs).max()))
        # B4 re-asserted from the recorded holdings.
        px = prices.loc[grid, r.tickers]
        implied = (r.units * px).div(r.v_post, axis=0)
        worst_w = max(worst_w, float(np.abs(implied - r.target_weights).to_numpy().max()))
        # B5 re-asserted.
        t_min = min(t_min, float(r.turnover.min()))
        nan_total += int(
            r.target_weights.isna().to_numpy().sum()
            + r.drift_weights.isna().to_numpy().sum()
            + r.turnover.isna().sum() + r.costs.isna().sum()
            + r.v_pre.isna().sum() + r.v_post.isna().sum()
        )

    assert worst_cost <= TOL_M2_COST, f"GATE C-M2 (B3): {worst_cost:.3e} -- STOP"
    assert worst_w < TOL_M2_WEIGHT, f"GATE C-M2 (B4): {worst_w:.3e} -- STOP"
    assert t_min >= 0.0, f"GATE C-M2 (B5): negative turnover {t_min:.3e} -- STOP"
    assert nan_total == 0, f"GATE C-M2 (B5): {nan_total} NaN in grid records -- STOP"
    lines.extend([
        f"**GATE C-M2 PASS** — B3/B4/B5 re-asserted over all {len(grid)} grid "
        f"dates × {len(results)} portfolios ({len(grid) * len(results)} "
        f"rebalances). Cost identity worst |rate·T − charged| = "
        f"{worst_cost:.3e} ≤ {TOL_M2_COST:g}; post-cost weights vs allocator "
        f"target worst |diff| = {worst_w:.3e} < {TOL_M2_WEIGHT:g}; min T = "
        f"{t_min:.6f} ≥ 0; NaN in grid records = {nan_total}. Re-implemented in "
        "this module rather than imported from `stage_b`.",
        "",
    ])


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run_stage_c(
    out_dir,
    weight_fn_factory: Callable[[str, str | None], Callable] = allocator_weight_fn,
    registered: bool = True,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    stamp = datetime.now(timezone.utc)

    lines.extend([
        "# Task 11 Stage C report — headline backtest (one-shot)",
        "",
    ])
    if not registered:
        lines.extend([
            "> **SMOKE RUN — NOT THE REGISTERED RESULT.** Stub allocators; no "
            "headline quantity in this file is meaningful.",
            "",
        ])

    lines.extend([
        "Executed **once**, reported regardless of outcome. No "
        "performance gate exists in this stage; the only gates are the three "
        "mechanical completion checks, which are accounting rather "
        "than performance. H1 and portfolio performance are reported as "
        "separate claims and are never traded off against each other.",
        "",
        "## Method disclosures (registered, stated once)",
        "",
        f"1. **Execution timing.** {DISCLOSURE_EXECUTION}",
        f"2. **Inception.** {DISCLOSURE_INCEPTION}",
        f"3. **Risk-free and MAR.** {DISCLOSURE_RF}",
        f"4. **Benchmark cost symmetry.** {DISCLOSURE_COST_SYMMETRY}",
        "",
        "## Pre-flight and mechanical gates",
        "",
    ])
    if registered:
        preflight(lines)

    prices = load_prices()
    grid = rebalance_grid()
    grid = grid[(grid >= pd.Timestamp(SPAN_START)) & (grid <= pd.Timestamp(SPAN_END))]
    days = prices.index[(prices.index >= pd.Timestamp(SPAN_START))
                        & (prices.index <= pd.Timestamp(SPAN_END))]

    results = {}
    for aid, prof in PORTFOLIOS:
        results[portfolio_key(aid, prof)] = run_backtest(
            weight_fn_factory(aid, prof), prices, grid, SPAN_START, SPAN_END,
            cost_rate=COST_RATE,
        )

    gate_cm1(lines, results, days)
    gate_cm2(lines, results, prices, grid)

    # ----------------------------------------------------------------- CSVs --
    keys = list(results)
    value_paths = pd.DataFrame({k: results[k].values for k in keys})
    value_paths.index.name = "date"
    value_paths.to_csv(out_dir / "value_paths_daily.csv")

    def _weight_frame(attr: str) -> pd.DataFrame:
        recs = []
        for (aid, prof) in PORTFOLIOS:
            k = portfolio_key(aid, prof)
            w = getattr(results[k], attr).copy()
            w.insert(0, "portfolio", k)
            w.insert(0, "profile", prof or "")
            w.insert(0, "allocator_id", aid)
            recs.append(w.reset_index())
        return pd.concat(recs, ignore_index=True)

    _weight_frame("target_weights").to_csv(out_dir / "grid_weights_target.csv", index=False)
    _weight_frame("drift_weights").to_csv(out_dir / "grid_weights_drift.csv", index=False)

    tc_recs = []
    for (aid, prof) in PORTFOLIOS:
        k = portfolio_key(aid, prof)
        r = results[k]
        tc = pd.DataFrame({
            "turnover": r.turnover, "cost_rate": r.cost_rate, "cost": r.costs,
            "cost_paid": r.cost_paid, "v_pre": r.v_pre, "v_post": r.v_post,
        })
        tc.insert(0, "portfolio", k)
        tc.insert(0, "profile", prof or "")
        tc.insert(0, "allocator_id", aid)
        tc["is_inception"] = tc.index == grid[0]
        tc_recs.append(tc.reset_index())
    pd.concat(tc_recs, ignore_index=True).to_csv(
        out_dir / "rebalance_turnover_costs.csv", index=False)

    # -------------------------------------------------------------- metrics --
    post_inception = counted_grid_dates(grid, days)
    full_recs = []
    for (aid, prof) in PORTFOLIOS:
        k = portfolio_key(aid, prof)
        r = results[k]
        m = compute_metrics(r.values, r.turnover.loc[post_inception],
                            r.costs.loc[post_inception])
        full_recs.append({"portfolio": k, "allocator_id": aid, "profile": prof or "", **m})
    full_metrics = pd.DataFrame(full_recs)
    full_metrics.to_csv(out_dir / "metrics_full_period.csv", index=False)

    sub_recs = []
    for label, s, e, cum_only in SUBPERIODS:
        wdays = days[(days >= pd.Timestamp(s)) & (days <= pd.Timestamp(e))]
        counted = counted_grid_dates(grid, wdays)
        for (aid, prof) in PORTFOLIOS:
            k = portfolio_key(aid, prof)
            r = results[k]
            m = compute_metrics(
                r.values.loc[wdays],
                None if cum_only else r.turnover.loc[counted],
                None if cum_only else r.costs.loc[counted],
                cumulative_only=cum_only,
            )
            sub_recs.append({
                "window": label, "window_start": wdays[0].date(),
                "window_end": wdays[-1].date(), "portfolio": k,
                "allocator_id": aid, "profile": prof or "", **m,
            })
    sub_metrics = pd.DataFrame(sub_recs)
    sub_metrics.to_csv(out_dir / "metrics_subperiods.csv", index=False)

    # ------------------------------------------------------------------- H1 --
    q3_dates = H1.q3_dates_from_grid(grid)
    h1_rows = {
        p: H1.h1_for_profile(results[f"rb_advisor:{p}"], results[f"rb_classical:{p}"],
                             post_inception, q3_dates)
        for p in PROFILES
    }
    H1.h1_frame(h1_rows).to_csv(out_dir / "h1_quantities.csv", index=False)
    v, held, failed = H1.verdict(h1_rows[HEADLINE_PROFILE])
    sentence = H1.verdict_sentence(v, held, failed)

    # -------------------------------------------------------------- figures --
    bal = {"one_over_n": "one_over_n", "static_bucket": "static_bucket:balanced",
           "rb_classical": "rb_classical:balanced", "rb_advisor": "rb_advisor:balanced"}
    equity_curve_figure(
        {kk: results[vv].values for kk, vv in bal.items()}, "balanced",
        out_dir / "f1_equity_balanced.png", with_hindsight=True, figure_label="F1")
    drawdown_figure({kk: results[vv].values for kk, vv in bal.items()}, "balanced",
                    out_dir / "f2_drawdown_balanced.png")
    weights_stack_figure(
        {"rb_advisor": results["rb_advisor:balanced"].target_weights,
         "rb_classical": results["rb_classical:balanced"].target_weights},
        "balanced", out_dir / "f3_weights_rb_balanced.png")
    f4_window = grid[(grid >= pd.Timestamp(F4_WINDOW[0])) & (grid <= pd.Timestamp(F4_WINDOW[1]))]
    crash_sleeve_figure(
        {"rb_advisor": results["rb_advisor:balanced"].target_weights,
         "rb_classical": results["rb_classical:balanced"].target_weights},
        f4_window, q3_dates, "balanced", out_dir / "f4_crash_equity_sleeve.png")
    for prof, fname, lab in (("conservative", APPENDIX_FIGURES[0], "A1"),
                             ("growth", APPENDIX_FIGURES[1], "A2")):
        sel = {"one_over_n": "one_over_n", "static_bucket": f"static_bucket:{prof}",
               "rb_classical": f"rb_classical:{prof}", "rb_advisor": f"rb_advisor:{prof}"}
        equity_curve_figure({kk: results[vv].values for kk, vv in sel.items()}, prof,
                            out_dir / fname, with_hindsight=False, figure_label=lab)


    # ------------------------------------------------------------- report ----
    lines.extend(_results_sections(
        full_metrics, sub_metrics, h1_rows, held, failed, sentence,
        q3_dates, f4_window))

    lines.extend([
        "## Freeze",
        "",
        f"Headline results computed once on "
        f"{stamp.date().isoformat()} ({stamp.isoformat(timespec='seconds')}), "
        "frozen; no re-run without logged defect and planning ruling.",
        "",
        "## Deliverables",
        "",
        *[f"- `outputs/task11/{n}`" for n in
          (*REGISTERED_CSVS, *REGISTERED_FIGURES, *APPENDIX_FIGURES)],
        "- `outputs/task11/stage_c_report.md`",
        "- `src/task11/metrics.py`, `src/task11/h1.py`, `src/task11/exhibits.py`, "
        "`src/task11/stage_c.py`",
        "",
        "**STOP — Stage C boundary.** Mechanical gates C-M1 and C-M2 all "
        "PASS; results are reported above regardless of direction. ",
    ])
    return lines


# --------------------------------------------------------------------------- #
# Report body
# --------------------------------------------------------------------------- #
def _results_sections(full_metrics, sub_metrics, h1_rows, held, failed,
                      sentence, q3_dates, f4_window) -> list[str]:
    out: list[str] = []

    # ---- T1 ----
    t1 = pd.DataFrame([{
        "portfolio": r["portfolio"],
        "ann. return": _pct(r["ann_return"]),
        "ann. vol": _pct(r["ann_vol"]),
        "Sharpe": _num(r["sharpe"]),
        "Sortino": _num(r["sortino"]),
        "max DD": _pct(r["max_drawdown"]),
        "T̄": _pct(r["mean_turnover"]),
        "cost drag (bps/yr)": _num(r["cost_drag_bps_per_year"], 1),
    } for _, r in full_metrics.iterrows()])

    out.extend([
        "## Results",
        "",
        f"### T1 — full-period metrics, all 10 portfolios ({SPAN_START} → {SPAN_END})",
        "",
        f"Net of costs on the daily value path; {int(full_metrics['n_days'].iloc[0])} "
        f"return days over {full_metrics['years'].iloc[0]:.2f} years. T is the "
        "mean traded fraction over the 155 post-inception grid dates.",
        "",
        _md(t1),
        "",
        f"*Hindsight disclosure: {HINDSIGHT}*",
        "",
    ])

    # ---- T2 ----
    bal_keys = ["one_over_n", "static_bucket:balanced", "rb_classical:balanced",
                "rb_advisor:balanced"]
    out.extend([
        "### T2 — sub-period metrics, balanced profile",
        "",
        "The COVID row reports cumulative return and max "
        "drawdown only — an 8-week window is not annualised, and the "
        "annualised fields are left empty in `metrics_subperiods.csv` rather "
        "than filled, so no unregistered number can be quoted from the artifacts.",
        "",
        _md(_subperiod_table(sub_metrics, bal_keys)),
        "",
    ])

    # ---- T3 ----
    out.extend([
        "### T3 — H1 quantities, all three profiles",
        "",
        "rb_advisor vs rb_classical within profile, on target-vs-drift weights "
        "at grid dates; cost-invariant. Weight quantities at the "
        "registered 3 dp. 'advisor smoother' records the registered direction "
        "(advisor < classical).",
        "",
        _md(H1.h1_report_table(h1_rows)),
        "",
        "Equity-sleeve levels behind H1-Q3 (grid dates "
        f"{q3_dates['Feb'].date()}, {q3_dates['Mar'].date()}, "
        f"{q3_dates['Apr'].date()}). The Feb to Apr leg is descriptive only — "
        "recovery dynamics confound its direction and it is not a registered "
        "prediction:",
        "",
        _md(H1.q3_detail_table(h1_rows)),
        "",
        "#### H1 verdict",
        "",
        f"{sentence}",
        "",
        f"- Components holding in the registered direction: "
        f"{', '.join(held) if held else 'none'}.",
        f"- Components running the other way: "
        f"{', '.join(failed) if failed else 'none'}.",
        f"- Robustness rows (conservative, growth) are in T3 above; the verdict "
        f"itself is evaluated on the {HEADLINE_PROFILE} profile.",
        "",
        "H1 concerns smoothness, not risk-adjusted performance. The two claims "
        "are reported independently and neither is traded off against the other.",
        "",
    ])

    # ---- exhibits ----
    out.extend([
        "## Registered exhibits",
        "",
        "| exhibit | file | content |",
        "|---|---|---|",
        "| F1 | `f1_equity_balanced.png` | net-of-cost equity curves, balanced, "
        "log y-axis, full span (1/N is profile-less and appears in every "
        "profile's figure) |",
        "| F2 | `f2_drawdown_balanced.png` | drawdown curves, balanced, full span |",
        "| F3 | `f3_weights_rb_balanced.png` | stacked-area target-weight "
        "trajectories at grid dates, rb_advisor and rb_classical, two panels |",
        "| F4 | `f4_crash_equity_sleeve.png` | equity-sleeve target weight at "
        f"grid dates {f4_window[0].date()} → {f4_window[-1].date()}, "
        "rb_advisor vs rb_classical |",
        "| A1 | `a1_equity_conservative.png` | appendix — conservative equity curves |",
        "| A2 | `a2_equity_growth.png` | appendix — growth equity curves |",
        "",
        "## Appendix — per-profile sub-period tables",
        "",
    ])
    for prof in ("conservative", "growth"):
        keys = ["one_over_n", f"static_bucket:{prof}", f"rb_classical:{prof}",
                f"rb_advisor:{prof}"]
        out.extend([f"### {prof}", "", _md(_subperiod_table(sub_metrics, keys)), ""])

    return out


def _subperiod_table(sub_metrics: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    recs = []
    for label, _s, _e, _cum_only in SUBPERIODS:
        for k in keys:
            r = sub_metrics[(sub_metrics["window"] == label)
                            & (sub_metrics["portfolio"] == k)].iloc[0]
            recs.append({
                "window": label, "portfolio": k,
                "cumulative return": _pct(r["cumulative_return"]),
                "ann. return": _pct(r["ann_return"]),
                "ann. vol": _pct(r["ann_vol"]),
                "Sharpe": _num(r["sharpe"]),
                "Sortino": _num(r["sortino"]),
                "max DD": _pct(r["max_drawdown"]),
                "T̄": _pct(r["mean_turnover"]),
                "cost drag (bps/yr)": _num(r["cost_drag_bps_per_year"], 1),
            })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Main — the registered, one-shot configuration
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    try:
        lines = run_stage_c(OUT_DIR)
    except AssertionError as e:
        report = "\n".join([
            "# Task 11 Stage C report — GATE FAILURE",
            "",
            f"**STOP**: {e}",
            "",
            "Diagnosis recorded.",
        ])
        (OUT_DIR / "stage_c_report.md").write_text(report, encoding="utf-8")
        print(report)
        sys.exit(1)
    report = "\n".join(lines)
    (OUT_DIR / "stage_c_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()


__all__ = ["PORTFOLIOS", "SPAN_END", "SPAN_START", "main", "run_stage_c"]
