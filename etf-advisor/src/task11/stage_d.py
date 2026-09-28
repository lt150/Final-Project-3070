"""Task 11 Stage D: sensitivity appendix.

Executed only after the Stage C freeze. Five one 
factor at a time deviations from the headline configuration:

  V1  correlation window R = 60      (parallel path, §7)
  V2  correlation window R = 120     (parallel path, §7)
  V3  quarterly rebalancing          (subset of the headline grid)
  V4  transaction costs 5 bps        (frozen weights, value path only)
  V5  transaction costs 25 bps       (frozen weights, value path only)

Scope: balanced profile; `rb_advisor`, `rb_classical`, `one_over_n`. All
artifacts are quarantined to `outputs/task11/appendix_sensitivity/`.

**Stage D feeds nothing back.** No headline number, sentence, exhibit or
verdict is edited in response to anything below; the frozen Stage C body 
are untouched by this stage. Nothing here participates in any selection.

  GATE D1  before any 60/120 run: at window = 250 the parallel path reproduces
           R[SPY, QQQ] and trace(Σ_classical) at 2024-12-02 to <= 1e-12 (pure
           arithmetic) AND the rb_advisor / rb_classical balanced 2024-12-02
           weight vectors to within 1e-6 elementwise (solver terminal-accuracy
           band, per the Gate C2 lesson). D1 failure -> STOP.
  GATE D2  mechanical accounting re-assertion per variant, B3/B4/B5-style.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..data.config import ROOT, UNIVERSE
from ..task9.allocators import rebalance_grid
from ..task9.covariance import corr_matrix, sigma
from .engine import COST_RATE, V_CASH, run_backtest
from .fixtures import allocator_weight_fn, load_prices, portfolio_key
from .metrics import compute_metrics, counted_grid_dates
from .sensitivity_cov import (
    HEADLINE_WINDOW,
    corr_matrix_w,
    one_over_n_weight_fn,
    rb_weight_fn,
    sigma_w,
)
from .stage_c import SPAN_END, SPAN_START

HEADLINE_DIR = ROOT / "outputs" / "task11"
OUT_DIR = HEADLINE_DIR / "appendix_sensitivity"

PROFILE = "balanced"
SCOPE = ("one_over_n", "rb_classical", "rb_advisor")
SCOPE_KEYS = ("one_over_n", "rb_classical:balanced", "rb_advisor:balanced")

D1_DATE = "2024-12-02"
TOL_D1_ARITH = 1e-12
TOL_D1_SOLVER = 1e-6

TOL_D2_COST = 1e-15
TOL_D2_WEIGHT = 1e-12
TOL_REUSE = 1e-12
TOL_INVARIANCE = 1e-12

QUARTERLY_MONTHS = (1, 4, 7, 10)
V3_GRID_COUNT = 52

VARIANTS = (
    ("V1", "R = 60 d", "v1_R60_metrics.csv"),
    ("V2", "R = 120 d", "v2_R120_metrics.csv"),
    ("V3", "quarterly rebalance", "v3_quarterly_metrics.csv"),
    ("V4", "costs 5 bps", "v4_cost5bps_metrics.csv"),
    ("V5", "costs 25 bps", "v5_cost25bps_metrics.csv"),
)

lines: list[str] = []


def _md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def _pct(x, dp: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{dp}f}%"


def _num(x, dp: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}"


# --------------------------------------------------------------------------- #
# GATE D1
# --------------------------------------------------------------------------- #
def gate_d1() -> None:
    t = pd.Timestamp(D1_DATE)

    r_ref = float(corr_matrix(t).at["SPY", "QQQ"])
    r_par = float(corr_matrix_w(t, HEADLINE_WINDOW).at["SPY", "QQQ"])
    tr_ref = float(np.trace(sigma(t, "classical").to_numpy()))
    tr_par = float(np.trace(sigma_w(t, "classical", HEADLINE_WINDOW).to_numpy()))

    arith = [
        ("R[SPY, QQQ]", r_ref, r_par, abs(r_ref - r_par)),
        ("trace(Σ_classical)", tr_ref, tr_par, abs(tr_ref - tr_par)),
    ]
    worst_arith = max(a[3] for a in arith)

    wrows, worst_solver = [], 0.0
    for variant, aid in (("advisor", "rb_advisor"), ("classical", "rb_classical")):
        w_ref = allocator_weight_fn(aid, PROFILE)(t)
        w_par = rb_weight_fn(variant, PROFILE, HEADLINE_WINDOW)(t)
        d = float(np.abs(w_ref - w_par).max())
        worst_solver = max(worst_solver, d)
        wrows.append({"allocator": aid, "max abs elementwise diff": f"{d:.3e}",
                      "within 1e-6": "yes" if d < TOL_D1_SOLVER else "NO"})

    lines.extend([
        "### GATE D1 — parallel path validated at the headline window",
        "",
        f"At window = {HEADLINE_WINDOW} the parallel implementation must "
        "reproduce the registered path. Arithmetic quantities are gated at "
        f"{TOL_D1_ARITH:g}; the weight vectors pass through the registered "
        f"L-BFGS-B solve and are gated at {TOL_D1_SOLVER:g}, the solver's "
        "terminal-accuracy band — no gate tolerance sits inside it.",
        "",
        _md(pd.DataFrame([
            {"quantity": n, "registered": repr(a), "parallel": repr(b),
             "abs diff": f"{d:.3e}"} for n, a, b, d in arith
        ])),
        "",
        f"Balanced weight vectors at {D1_DATE}, all eight tickers:",
        "",
        _md(pd.DataFrame(wrows)),
        "",
    ])
    assert worst_arith <= TOL_D1_ARITH, (
        f"GATE D1: arithmetic reproduction worst |diff| {worst_arith:.3e} > "
        f"{TOL_D1_ARITH:g} -- STOP, no 60/120 run"
    )
    assert worst_solver < TOL_D1_SOLVER, (
        f"GATE D1: weight reproduction worst |diff| {worst_solver:.3e} >= "
        f"{TOL_D1_SOLVER:g} -- STOP, no 60/120 run"
    )
    lines.extend([
        f"**GATE D1 PASS** — worst arithmetic |diff| {worst_arith:.3e} ≤ "
        f"{TOL_D1_ARITH:g}; worst weight |diff| {worst_solver:.3e} < "
        f"{TOL_D1_SOLVER:g}. The parallel path differs from the registered one "
        "in the correlation window and in nothing else; V1 and V2 may run.",
        "",
    ])


# --------------------------------------------------------------------------- #
# Frozen-weight reuse (V4 / V5)
# --------------------------------------------------------------------------- #
def _frozen(name: str) -> pd.DataFrame:
    return pd.read_csv(HEADLINE_DIR / name, parse_dates=["date"])


def path_from_frozen(w_tgt: pd.DataFrame, turnover: pd.Series,
                     prices: pd.DataFrame, days: pd.DatetimeIndex,
                     cost_rate: float) -> dict:
    """Value path from frozen target weights and turnover at a variant rate.

    The cost haircut is proportional and uniform, so target weights, 
    drifted weights and turnover are cost-invariant. V4/V5 therefore
    need no allocator call and no solve: only the value path is recomputed.
    """
    grid = w_tgt.index
    tickers = list(w_tgt.columns)
    values = pd.Series(np.nan, index=days, dtype=float)
    units_rec, vpost_rec, cost_rec = {}, {}, {}

    v = float(V_CASH)
    units = pd.Series(0.0, index=tickers)
    for k, t in enumerate(grid):
        if k > 0:
            v = float(units @ prices.loc[t, tickers])
        cost = cost_rate * float(turnover.loc[t])
        v = v * (1.0 - cost)
        units = v * w_tgt.loc[t] / prices.loc[t, tickers]
        values.loc[t] = v
        t_next = grid[k + 1] if k + 1 < len(grid) else None
        seg_end = t_next if t_next is not None else days[-1]
        seg = days[(days > t) & (days <= seg_end)]
        if len(seg) > 0:
            values.loc[seg] = prices.loc[seg, tickers].to_numpy() @ units.to_numpy()
        units_rec[t], vpost_rec[t], cost_rec[t] = units.copy(), v, cost

    return {
        "values": values,
        "units": pd.DataFrame(units_rec).T[tickers],
        "v_post": pd.Series(vpost_rec),
        "costs": pd.Series(cost_rec),
        "turnover": turnover,
        "target_weights": w_tgt,
        "cost_rate": cost_rate,
    }


def reuse_validation(frozen_w: dict, frozen_t: dict, prices, days) -> None:
    """At the headline rate the reuse path must rebuild the frozen value paths."""
    ref = pd.read_csv(HEADLINE_DIR / "value_paths_daily.csv",
                      index_col="date", parse_dates=["date"])
    worst = 0.0
    for key in SCOPE_KEYS:
        got = path_from_frozen(frozen_w[key], frozen_t[key], prices, days, COST_RATE)
        rel = float((np.abs(got["values"] - ref[key]) / np.abs(ref[key])).max())
        worst = max(worst, rel)
    assert worst < TOL_REUSE, (
        f"frozen-weight reuse does not rebuild the frozen value path at the "
        f"headline rate: worst relative |diff| {worst:.3e} >= {TOL_REUSE:g} -- STOP"
    )
    lines.extend([
        "### Reuse validation — frozen weights rebuild the frozen paths",
        "",
        "Before V4/V5 deviate the cost rate, the reuse machinery is run *at* "
        f"the headline rate ({COST_RATE:g}) from the frozen "
        "`grid_weights_target.csv` and `rebalance_turnover_costs.csv`, and "
        "checked against the frozen `value_paths_daily.csv`. Worst relative "
        f"|diff| = {worst:.3e} < {TOL_REUSE:g} across the three in-scope "
        "portfolios. **PASS** — V4/V5 differ from the headline in the cost rate "
        "and in nothing else.",
        "",
    ])


def invariance_check(prices, grid, frozen_w) -> None:
    """Direct test of the §1.2 cost-invariance claim that V4/V5 rely on."""
    probe_rate = 0.0025
    worst = 0.0
    for aid in ("rb_advisor", "rb_classical"):
        key = portfolio_key(aid, PROFILE)
        r = run_backtest(allocator_weight_fn(aid, PROFILE), prices, grid,
                         SPAN_START, SPAN_END, cost_rate=probe_rate)
        worst = max(worst, float(np.abs(r.target_weights - frozen_w[key]).to_numpy().max()))
    assert worst < TOL_INVARIANCE, (
        f"Cost-invariance violated: target weights move {worst:.3e} when "
        f"the cost rate changes -- STOP"
    )
    lines.extend([
        "### Recorded check — cost-invariance of weights",
        "",
        "Because the haircut is proportional and uniform, "
        "target weights are cost-invariant; V4/V5 rest on it. Re-running the "
        f"engine end to end at {probe_rate:g} (25 bps) and comparing the "
        "target-weight frames against the frozen ones gives worst elementwise "
        f"|diff| = {worst:.3e} < {TOL_INVARIANCE:g} over both risk-budget "
        "allocators × 156 grid dates × 8 tickers. The claim holds exactly. "
        "Recorded, not a gate.",
        "",
    ])


# --------------------------------------------------------------------------- #
# GATE D2
# --------------------------------------------------------------------------- #
def gate_d2(tag: str, runs: dict, prices: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    worst_cost = worst_w = 0.0
    t_min, nan_total = np.inf, 0
    for r in runs.values():
        rate = r["cost_rate"] if isinstance(r, dict) else r.cost_rate
        turn = r["turnover"] if isinstance(r, dict) else r.turnover
        costs = r["costs"] if isinstance(r, dict) else r.costs
        units = r["units"] if isinstance(r, dict) else r.units
        vpost = r["v_post"] if isinstance(r, dict) else r.v_post
        tgt = r["target_weights"] if isinstance(r, dict) else r.target_weights
        vals = r["values"] if isinstance(r, dict) else r.values

        worst_cost = max(worst_cost, float(np.abs(rate * turn - costs).max()))
        px = prices.loc[tgt.index, list(tgt.columns)]
        implied = (units * px).div(vpost, axis=0)
        worst_w = max(worst_w, float(np.abs(implied - tgt).to_numpy().max()))
        t_min = min(t_min, float(turn.min()))
        nan_total += int(vals.isna().sum() + tgt.isna().to_numpy().sum()
                         + turn.isna().sum() + costs.isna().sum())
        assert vals.index.equals(days), f"GATE D2 [{tag}]: value path does not cover the span"

    assert worst_cost <= TOL_D2_COST, f"GATE D2 [{tag}] (B3): {worst_cost:.3e} -- STOP"
    assert worst_w < TOL_D2_WEIGHT, f"GATE D2 [{tag}] (B4): {worst_w:.3e} -- STOP"
    assert t_min >= 0.0, f"GATE D2 [{tag}] (B5): negative turnover {t_min:.3e} -- STOP"
    assert nan_total == 0, f"GATE D2 [{tag}] (B5): {nan_total} NaN -- STOP"
    return {"variant": tag, "worst abs(rate·T − cost)": f"{worst_cost:.3e}",
            "worst abs(units·P/V_post − w_target)": f"{worst_w:.3e}",
            "min T": f"{t_min:.6f}", "NaN": nan_total, "result": "PASS"}


# --------------------------------------------------------------------------- #
# Metric assembly
# --------------------------------------------------------------------------- #
def variant_metrics(runs: dict, grid: pd.DatetimeIndex,
                    days: pd.DatetimeIndex) -> pd.DataFrame:
    counted = counted_grid_dates(grid, days)
    recs = []
    for key in SCOPE_KEYS:
        r = runs[key]
        vals = r["values"] if isinstance(r, dict) else r.values
        turn = r["turnover"] if isinstance(r, dict) else r.turnover
        costs = r["costs"] if isinstance(r, dict) else r.costs
        recs.append({"portfolio": key,
                     **compute_metrics(vals, turn.loc[counted], costs.loc[counted])})
    return pd.DataFrame(recs)


def report_table(m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "portfolio": r["portfolio"],
        "ann. return": _pct(r["ann_return"]),
        "ann. vol": _pct(r["ann_vol"]),
        "Sharpe": _num(r["sharpe"]),
        "Sortino": _num(r["sortino"]),
        "max DD": _pct(r["max_drawdown"]),
        "T̄": _pct(r["mean_turnover"]),
        "cost drag (bps/yr)": _num(r["cost_drag_bps_per_year"], 1),
    } for _, r in m.iterrows()])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)

    lines.extend([
        "# Task 11 Stage D report — sensitivity appendix",
        "",
        "Executed after the Stage C freeze. "
        "Five one-factor-at-a-time deviations from the headline "
        "configuration; scope is the **balanced** profile and the three "
        "portfolios `one_over_n`, `rb_classical`, `rb_advisor`. All variant "
        "artifacts are quarantined to `outputs/task11/appendix_sensitivity/`.",
        "",
        "**Stage D feeds nothing back.** No headline number, sentence, exhibit "
        "or verdict has been edited in response to anything in this report; the "
        "frozen Stage C body and its seven CSVs are untouched. Nothing "
        "here participates in any selection. The headline rows reproduced below "
        "are read from the frozen `metrics_full_period.csv`, for reference only.",
        "",
        "## Gates and validation",
        "",
    ])

    prices = load_prices()
    grid = rebalance_grid()
    grid = grid[(grid >= pd.Timestamp(SPAN_START)) & (grid <= pd.Timestamp(SPAN_END))]
    days = prices.index[(prices.index >= pd.Timestamp(SPAN_START))
                        & (prices.index <= pd.Timestamp(SPAN_END))]

    try:
        gate_d1()

        # Frozen weights / turnover for the in-scope portfolios.
        tgt_all = _frozen("grid_weights_target.csv")
        tc_all = _frozen("rebalance_turnover_costs.csv")
        frozen_w = {k: tgt_all[tgt_all["portfolio"] == k].set_index("date")[list(UNIVERSE)]
                    for k in SCOPE_KEYS}
        frozen_t = {k: tc_all[tc_all["portfolio"] == k].set_index("date")["turnover"]
                    for k in SCOPE_KEYS}

        reuse_validation(frozen_w, frozen_t, prices, days)
        invariance_check(prices, grid, frozen_w)

        results, grids, d2_rows = {}, {}, []

        # ---- V1 / V2: window override ------------------------------------
        for tag, window in (("V1", 60), ("V2", 120)):
            runs = {}
            for aid in SCOPE:
                key = portfolio_key(aid, None if aid == "one_over_n" else PROFILE)
                wf = (one_over_n_weight_fn() if aid == "one_over_n"
                      else rb_weight_fn("advisor" if aid == "rb_advisor" else "classical",
                                        PROFILE, window))
                runs[key] = run_backtest(wf, prices, grid, SPAN_START, SPAN_END,
                                         cost_rate=COST_RATE)
            results[tag], grids[tag] = runs, grid
            d2_rows.append(gate_d2(tag, runs, prices, days))

        # ---- V3: quarterly subset of the headline grid --------------------
        q_grid = grid[np.isin(grid.month, QUARTERLY_MONTHS)]
        assert len(q_grid) == V3_GRID_COUNT, (
            f"V3 grid has {len(q_grid)} dates, §7 registers {V3_GRID_COUNT} -- STOP"
        )
        assert q_grid[0] == pd.Timestamp(SPAN_START), (
            f"V3 grid starts {q_grid[0].date()}, registered {SPAN_START} -- STOP"
        )
        runs = {}
        for aid in SCOPE:
            prof = None if aid == "one_over_n" else PROFILE
            runs[portfolio_key(aid, prof)] = run_backtest(
                allocator_weight_fn(aid, prof), prices, q_grid,
                SPAN_START, SPAN_END, cost_rate=COST_RATE)
        results["V3"], grids["V3"] = runs, q_grid
        d2_rows.append(gate_d2("V3", runs, prices, days))

        # ---- V4 / V5: frozen weights, variant cost rate --------------------
        for tag, rate in (("V4", 0.0005), ("V5", 0.0025)):
            runs = {k: path_from_frozen(frozen_w[k], frozen_t[k], prices, days, rate)
                    for k in SCOPE_KEYS}
            results[tag], grids[tag] = runs, grid
            d2_rows.append(gate_d2(tag, runs, prices, days))

    except AssertionError as e:
        lines.extend(["", f"**GATE FAILURE — STOP**: {e}", "",
                      "Diagnosis recorded; the headline "
                      "remains frozen and unedited."])
        report = "\n".join(lines)
        (HEADLINE_DIR / "stage_d_report.md").write_text(report, encoding="utf-8")
        print(report)
        sys.exit(1)

    lines.extend([
        "### GATE D2 — mechanical accounting, per variant",
        "",
        _md(pd.DataFrame(d2_rows)),
        "",
        f"**GATE D2 PASS** — B3/B4/B5-style accounting re-asserted for all five "
        f"variants (cost identity ≤ {TOL_D2_COST:g}, post-cost weights vs "
        f"target < {TOL_D2_WEIGHT:g}, T ≥ 0, no NaN, path complete over all "
        f"{len(days)} trading days).",
        "",
        "## Results",
        "",
        "### Headline reference (frozen, reproduced — not recomputed)",
        "",
        "R = 250 · monthly · 10 bps, read from `metrics_full_period.csv`:",
        "",
    ])

    head = pd.read_csv(HEADLINE_DIR / "metrics_full_period.csv")
    head = head[head["portfolio"].isin(SCOPE_KEYS)].set_index("portfolio").loc[list(SCOPE_KEYS)]
    head = head.reset_index()
    lines.extend([_md(report_table(head)), ""])

    summary = {"headline": head.set_index("portfolio")}
    for (tag, label, fname) in VARIANTS:
        m = variant_metrics(results[tag], grids[tag], days)
        m.insert(0, "variant", tag)
        m.insert(1, "variant_detail", label)
        m.to_csv(OUT_DIR / fname, index=False)
        summary[tag] = m.set_index("portfolio")
        lines.extend([
            f"### {tag} — {label}",
            "",
            _md(report_table(m)),
            "",
            f"CSV: `outputs/task11/appendix_sensitivity/{fname}` (full precision).",
            "",
        ])

    # Cross-variant summary on the two quantities sensitivity bears on most.
    rows = []
    for metric, col, fmt in (("Sharpe", "sharpe", _num),
                             ("T̄", "mean_turnover", _pct)):
        for key in SCOPE_KEYS:
            rec = {"metric": metric, "portfolio": key,
                   "headline": fmt(summary["headline"].at[key, col])}
            for tag, _label, _f in VARIANTS:
                rec[tag] = fmt(summary[tag].at[key, col])
            rows.append(rec)

    lines.extend([
        "### Cross-variant summary",
        "",
        "The two quantities the sensitivity grid bears on most. Headline = "
        "R 250 · monthly · 10 bps.",
        "",
        _md(pd.DataFrame(rows)),
        "",
        "T̄ under V4 and V5 is identical to the headline to the printed "
        "precision, which is the cost-invariance of weights showing up "
        "directly in the table rather than only in the check above.",
        "",
        "## Deliverables",
        "",
        *[f"- `outputs/task11/appendix_sensitivity/{f}`" for _t, _l, f in VARIANTS],
        "- `outputs/task11/stage_d_report.md`",
        "",
        "**STOP — Stage D boundary.** Gates D1 and D2 PASS; five variants "
        "reported. Headline results remain frozen and unedited.",
    ])

    report = "\n".join(lines)
    (HEADLINE_DIR / "stage_d_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
