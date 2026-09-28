"""Task 11 Stage B: engine + accounting gates on fixtures.

This stage proves the accounting on two registered fixtures, 
against an independently implemented reconciliation path,
before Stage C is allowed to execute once.

  GATE B1 — engine output matches the Fixture 1 registered exact expressions
            elementwise <= 1e-12.
  GATE B2 — on Fixture 2, the reconciliation path matches engine daily values
            <= 1e-10 relative and drift weights <= 1e-12 elementwise, all
            portfolios.
  GATE B3 — cost identity: independently recomputed rate * T equals the charged
            cost <= 1e-15 at every rebalance.
  GATE B4 — post-cost weights at every rebalance equal the allocator's target
            output <= 1e-12.
  GATE B5 — no NaN; the value path is defined for every trading day in the
            window; T >= 0 at every grid date.
  GATE B6 — rb_classical and rb_advisor balanced weights at 2012-02-01 match
            the Task 9 Stage C smoke table at its printed precision (6 dp).
            Solver-band gate, not a repr gate: this quantity
            passes through the registered L-BFGS-B solve, so no tolerance may
            sit inside the solver's terminal-accuracy band.

On any gate failure: STOP1.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..data.config import ROOT, UNIVERSE
from ..task9.allocators import rebalance_grid
from .engine import COST_RATE, run_backtest
from .fixtures import (
    D0,
    D1,
    D2,
    FIXTURE1_DATES,
    FIXTURE1_GRID,
    FIXTURE1_PRICES,
    FIXTURE1_TICKERS,
    FIXTURE2_END,
    FIXTURE2_GRID_COUNT,
    FIXTURE2_PORTFOLIOS,
    FIXTURE2_START,
    allocator_weight_fn,
    fixture1_expected,
    fixture1_weight_fn,
    load_prices,
    load_returns,
    portfolio_key,
)
from .reconcile import reconcile_backtest

TASK9_DIR = ROOT / "outputs" / "task9"
OUT_DIR = ROOT / "outputs" / "task11"

TOL_B1 = 1e-12
TOL_B2_VALUE_REL = 1e-10
TOL_B2_WEIGHT = 1e-12
TOL_B3 = 1e-15
TOL_B4 = 1e-12
TOL_ONE_OVER_N_DRIFT = 1e-12   # closed-form check, same engine-arithmetic band

B6_DATE = "2012-02-01"
B6_DP = 6

_PASS: list[str] = []
lines: list[str] = []


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.to_numpy()]
    return "\n".join([header, sep, *rows])


# --------------------------------------------------------------------------- #
# GATE B1 — Fixture 1
# --------------------------------------------------------------------------- #
def gate_b1() -> None:
    exp = fixture1_expected()
    res = run_backtest(
        fixture1_weight_fn,
        FIXTURE1_PRICES,
        FIXTURE1_GRID,
        FIXTURE1_DATES[0],
        FIXTURE1_DATES[-1],
        cost_rate=COST_RATE,
    )

    checks: list[tuple[str, float, float]] = [
        ("T(D0)", float(res.turnover.loc[D0]), float(exp["T_D0"])),
        ("cost(D0)", float(res.costs.loc[D0]), float(exp["cost_rate"] * exp["T_D0"])),
        ("V_post(D0)", float(res.v_post.loc[D0]), float(exp["V_post_D0"])),
        ("V(D0) on path", float(res.values.loc[D0]), float(exp["V_post_D0"])),
    ]
    for i, tk in enumerate(FIXTURE1_TICKERS):
        checks.append(
            (f"w_drift(D1)[{tk}]", float(res.drift_weights.loc[D1, tk]),
             float(exp["w_drift_D1"][i]))
        )
    checks += [
        ("T(D1)", float(res.turnover.loc[D1]), float(exp["T_D1"])),
        ("cost(D1)", float(res.costs.loc[D1]), float(exp["cost_D1"])),
        ("V_pre(D1)", float(res.v_pre.loc[D1]), float(exp["V_pre_D1"])),
        ("V_post(D1)", float(res.v_post.loc[D1]), float(exp["V_post_D1"])),
        ("V(D1) on path", float(res.values.loc[D1]), float(exp["V_post_D1"])),
        ("V(D2)", float(res.values.loc[D2]), float(exp["V_D2"])),
    ]

    rows, worst, worst_at = [], 0.0, None
    for label, got, want in checks:
        d = abs(got - want)
        if d > worst:
            worst, worst_at = d, label
        rows.append({"quantity": label, "engine": repr(got),
                     "registered (exact -> float)": repr(want),
                     "abs diff": f"{d:.3e}"})

    lines.extend([
        "### GATE B1 — Fixture 1 (synthetic, exact expressions)",
        "",
        "Prices A (100, 110, 121) · B (100, 100, 100) · C (100, 90, 81); targets "
        "w(D0) = (0.5, 0.3, 0.2), w(D1) = (0.4, 0.4, 0.2). Expected values are "
        "held as `fractions.Fraction` and converted to float only at the "
        "comparison. The registered expressions were additionally "
        "re-derived from the fixture inputs through the formulae and found "
        "exactly equal as rationals (asserted inside `fixture1_expected`), so "
        "the fixture is self-consistent before the gate is evaluated.",
        "",
        "Grid reading: {D0, D1}, with D2 the terminal valuation day —  "
        "no w(D2) and registers V(D2) as the un-haircut mark of the "
        "units bought at D1. See `fixtures.py`.",
        "",
        _md_table(pd.DataFrame(rows)),
        "",
    ])

    assert worst < TOL_B1, (
        f"GATE B1: worst elementwise |engine − registered| = {worst:.3e} at "
        f"{worst_at}, >= {TOL_B1:g} -- STOP"
    )
    lines.extend([
        f"**GATE B1 PASS** — all {len(checks)} registered quantities reproduced; "
        f"worst |diff| = {worst:.3e} (at {worst_at}) < {TOL_B1:g}.",
        "",
    ])
    _PASS.append("B1")


# --------------------------------------------------------------------------- #
# Fixture 2 runs
# --------------------------------------------------------------------------- #
def run_fixture2() -> tuple[dict, dict, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    prices = load_prices()
    returns = load_returns()
    grid = rebalance_grid()
    grid = grid[(grid >= pd.Timestamp(FIXTURE2_START)) & (grid <= pd.Timestamp(FIXTURE2_END))]
    assert len(grid) == FIXTURE2_GRID_COUNT, (
        f"Fixture 2 window holds {len(grid)} grid dates"
        f"{FIXTURE2_GRID_COUNT} -- STOP"
    )
    days = prices.index[
        (prices.index >= pd.Timestamp(FIXTURE2_START))
        & (prices.index <= pd.Timestamp(FIXTURE2_END))
    ]

    eng, rec = {}, {}
    for aid, prof in FIXTURE2_PORTFOLIOS:
        key = portfolio_key(aid, prof)
        wf = allocator_weight_fn(aid, prof)
        eng[key] = run_backtest(wf, prices, grid, FIXTURE2_START, FIXTURE2_END)
        rec[key] = reconcile_backtest(wf, returns, grid, days)
    return eng, rec, prices, grid, days


# --------------------------------------------------------------------------- #
# GATE B2
# --------------------------------------------------------------------------- #
def gate_b2(eng: dict, rec: dict) -> None:
    rows, worst_v, worst_w = [], 0.0, 0.0
    for key in eng:
        e, r = eng[key], rec[key]
        assert e.values.index.equals(r.values.index), f"{key}: day indices differ"
        rel = float((np.abs(e.values - r.values) / np.abs(e.values)).max())
        dw = float(np.abs(e.drift_weights - r.drift_weights).to_numpy().max())
        worst_v, worst_w = max(worst_v, rel), max(worst_w, dw)
        rows.append({"portfolio": key, "max rel ΔV (daily path)": f"{rel:.3e}",
                     "max abs Δw_drift": f"{dw:.3e}"})

    lines.extend([
        "### GATE B2 — independent reconciliation, Fixture 2",
        "",
        "Engine: fixed unit counts marked at adjusted closes, drift from one "
        "two-point price ratio. Reconciliation: no units and no price level "
        "anywhere — daily compounding of the portfolio return with a one-day "
        "weight-drift recursion, so a grid-date drift vector is a product of "
        "~21 daily steps. Same accounting, disjoint arithmetic.",
        "",
        _md_table(pd.DataFrame(rows)),
        "",
    ])
    assert worst_v < TOL_B2_VALUE_REL, (
        f"GATE B2: worst relative daily-value disagreement {worst_v:.3e} >= "
        f"{TOL_B2_VALUE_REL:g} -- STOP"
    )
    assert worst_w < TOL_B2_WEIGHT, (
        f"GATE B2: worst drift-weight disagreement {worst_w:.3e} >= "
        f"{TOL_B2_WEIGHT:g} -- STOP"
    )
    lines.extend([
        f"**GATE B2 PASS** — worst relative |ΔV| = {worst_v:.3e} < "
        f"{TOL_B2_VALUE_REL:g}; worst |Δw_drift| = {worst_w:.3e} < "
        f"{TOL_B2_WEIGHT:g}, across all four portfolios.",
        "",
    ])
    _PASS.append("B2")


# --------------------------------------------------------------------------- #
# GATES B3, B4, B5 — mechanical accounting (re-asserted in Stage C as C-M2)
# --------------------------------------------------------------------------- #
def gates_b3_b4_b5(eng: dict, prices: pd.DataFrame, days: pd.DatetimeIndex) -> None:
    b3_worst = b4_worst = 0.0
    b3_at = b4_at = None
    t_min = np.inf
    nan_report = []

    for key, e in eng.items():
        recomputed = e.cost_rate * e.turnover
        d3 = float(np.abs(recomputed - e.costs).max())
        if d3 > b3_worst:
            b3_worst, b3_at = d3, key

        # B4 recomputed from the holdings, not from the stored target frame.
        px = prices.loc[e.turnover.index, e.tickers]
        implied = (e.units * px).div(e.v_post, axis=0)
        d4 = float(np.abs(implied - e.target_weights).to_numpy().max())
        if d4 > b4_worst:
            b4_worst, b4_at = d4, key

        t_min = min(t_min, float(e.turnover.min()))
        nan_report.append({
            "portfolio": key,
            "value days": int(e.values.notna().sum()),
            "span days": len(days),
            "path complete": "yes" if e.values.index.equals(days) and e.values.notna().all() else "NO",
            "NaN in weights/turnover/costs": int(
                e.target_weights.isna().to_numpy().sum()
                + e.drift_weights.isna().to_numpy().sum()
                + e.turnover.isna().sum() + e.costs.isna().sum()
            ),
            "min T": f"{float(e.turnover.min()):.6f}",
            "max T": f"{float(e.turnover.max()):.6f}",
        })

    assert b3_worst <= TOL_B3, (
        f"GATE B3: cost identity worst |rate*T − charged| = {b3_worst:.3e} > "
        f"{TOL_B3:g} at {b3_at} -- STOP"
    )
    lines.extend([
        "### GATE B3 — cost identity",
        "",
        f"**GATE B3 PASS** — worst |rate·T − charged cost| = {b3_worst:.3e} ≤ "
        f"{TOL_B3:g} over all rebalances and portfolios ("
        + (f"worst at {b3_at}" if b3_at is not None
           else "bit-exact identity at every rebalance, so no worst case exists")
        + ").",
        "",
    ])
    _PASS.append("B3")

    assert b4_worst < TOL_B4, (
        f"GATE B4: post-cost weights vs allocator target worst |diff| = "
        f"{b4_worst:.3e} >= {TOL_B4:g} at {b4_at} -- STOP"
    )
    lines.extend([
        "### GATE B4 — post-cost weights equal the allocator target",
        "",
        "Recomputed from the holdings actually recorded (units · P / V_post), "
        "not from the stored target frame, so the check has content.",
        "",
        f"**GATE B4 PASS** — worst |units·P/V_post − w_target| = {b4_worst:.3e} "
        f"< {TOL_B4:g} (worst at {b4_at or 'n/a'}).",
        "",
    ])
    _PASS.append("B4")

    bad = [r for r in nan_report if r["path complete"] != "yes" or r["NaN in weights/turnover/costs"]]
    assert not bad, f"GATE B5: incomplete or NaN-bearing output: {bad} -- STOP"
    assert t_min >= 0.0, f"GATE B5: negative turnover {t_min:.3e} -- STOP"
    lines.extend([
        "### GATE B5 — completeness, no NaN, T ≥ 0",
        "",
        _md_table(pd.DataFrame(nan_report)),
        "",
        f"**GATE B5 PASS** — every portfolio's value path covers all "
        f"{len(days)} trading days in the window with no NaN anywhere in "
        f"values, weights, turnover or costs; min T over all grid dates and "
        f"portfolios = {t_min:.6f} ≥ 0.",
        "",
    ])
    _PASS.append("B5")


# --------------------------------------------------------------------------- #
# closed-form check for 1/N
# --------------------------------------------------------------------------- #
def check_one_over_n(eng: dict, prices: pd.DataFrame) -> None:
    e = eng["one_over_n"]
    grid = e.turnover.index
    worst = 0.0
    for k in range(1, len(grid)):
        t, t_prev = grid[k], grid[k - 1]
        g = prices.loc[t, e.tickers] / prices.loc[t_prev, e.tickers]
        closed = g / g.sum()          # equal-weight targets -> w_drift ∝ gross returns
        worst = max(worst, float(np.abs(closed - e.drift_weights.loc[t]).max()))
    assert worst < TOL_ONE_OVER_N_DRIFT, (
        f"1/N closed form: worst |diff| {worst:.3e} >= "
        f"{TOL_ONE_OVER_N_DRIFT:g} -- STOP"
    )
    lines.extend([
        "### Registered closed-form check — 1/N drift",
        "",
        "With equal-weight targets, w_drift at each grid date is exactly the "
        "per-ticker gross return since the previous grid date, normalised. "
        f"Checked at all {len(grid) - 1} post-inception grid dates: worst "
        f"|diff| = {worst:.3e} < {TOL_ONE_OVER_N_DRIFT:g}. PASS.",
        "",
    ])


# --------------------------------------------------------------------------- #
# GATE B6 — Task 9 smoke-table tie-out at printed precision
# --------------------------------------------------------------------------- #
def gate_b6(eng: dict) -> None:
    smoke = pd.read_csv(TASK9_DIR / "smoke_weights.csv")
    smoke = smoke[smoke["date"].astype(str) == B6_DATE]
    t = pd.Timestamp(B6_DATE)
    rows, mismatches = [], []
    for aid in ("rb_classical", "rb_advisor"):
        ref = smoke[(smoke["allocator_id"] == aid) & (smoke["profile"] == "balanced")]
        assert len(ref) == 1, f"GATE B6: {aid} balanced @ {B6_DATE} not unique in smoke_weights.csv"
        ref = ref.iloc[0]
        got = eng[portfolio_key(aid, "balanced")].target_weights.loc[t]
        for tk in UNIVERSE:
            g, r = f"{float(got[tk]):.{B6_DP}f}", f"{float(ref[tk]):.{B6_DP}f}"
            ok = g == r
            if not ok:
                mismatches.append((aid, tk, g, r))
            rows.append({"allocator": aid, "ticker": tk, "engine (6 dp)": g,
                         "Task 9 smoke (6 dp)": r, "match": "yes" if ok else "NO",
                         "abs diff": f"{abs(float(got[tk]) - float(ref[tk])):.3e}"})

    lines.extend([
        f"### GATE B6 — Task 9 Stage C smoke tie-out at {B6_DATE} ({B6_DP} dp)",
        "",
        "Solver-band gate: this quantity passes through the "
        "registered L-BFGS-B solve, so it is matched at the Task 9 report's "
        "printed precision rather than at repr level.",
        "",
        _md_table(pd.DataFrame(rows)),
        "",
    ])
    assert not mismatches, f"GATE B6: printed-precision mismatches {mismatches} -- STOP"
    lines.extend([
        f"**GATE B6 PASS** — all 16 weights (rb_classical and rb_advisor, "
        f"balanced) agree with the Task 9 Stage C smoke table to {B6_DP} dp; the "
        "engine consumes the allocator interface without perturbing it.",
        "",
    ])
    _PASS.append("B6")


# --------------------------------------------------------------------------- #
# Fixture 2 reporting tables
# --------------------------------------------------------------------------- #
def fixture2_tables(eng: dict, days: pd.DatetimeIndex) -> None:
    lines.extend([
        "## Fixture 2 tables (real data, "
        f"{FIXTURE2_START} .. {FIXTURE2_END}, {FIXTURE2_GRID_COUNT} grid dates, "
        f"{len(days)} trading days)",
        "",
        "Per-rebalance turnover T (traded fraction; inception = 1.0, "
        "excluded from turnover statistics in Stage C):",
        "",
    ])
    turn = pd.DataFrame({k: e.turnover for k, e in eng.items()})
    turn.index = [d.date() for d in turn.index]
    lines.append(_md_table(turn.reset_index(names="date").round(6).astype(str)))

    lines.extend([
        "",
        "Terminal net value at "
        f"{days[-1].date()} (all paths funded at 1.0 and haircut to 0.999 at "
        "inception):",
        "",
    ])
    term = pd.DataFrame(
        [{"portfolio": k, "V(inception)": repr(float(e.values.iloc[0])),
          "V(end)": repr(float(e.values.iloc[-1]))} for k, e in eng.items()]
    )
    lines.extend([_md_table(term), ""])

    v0 = {k: float(e.values.iloc[0]) for k, e in eng.items()}
    identical = len(set(v0.values())) == 1
    lines.extend([
        f"Inception values are bit-identical across portfolios: **{identical}** "
        f"; common value "
        f"`{next(iter(v0.values()))!r}`.",
        "",
    ])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines.extend([
        "# Task 11 Stage B report — engine and accounting on fixtures",
        "",
        "This stage proves the accounting on two registered fixtures,  "
        "against an independently implemented reconciliation path;"
        "`src/task11/reconcile.py` implements the same accounting by daily "
        "compounding with a one-day weight-drift recursion. **No full-period "
        "run happens in this stage.** Engine-side tolerances are registered "
        "values (1e-12 elementwise / 1e-10 relative); the one quantity "
        "that passes through the registered L-BFGS-B solve (Gate B6) is matched "
        "at printed precision instead, never inside the solver band.",
        "",
        f"Headline cost rate in force: {COST_RATE:g} (10 bps one-way, §0.1).",
        "",
        "## Gates",
        "",
    ])

    try:
        gate_b1()
        eng, rec, prices, grid, days = run_fixture2()
        gate_b2(eng, rec)
        gates_b3_b4_b5(eng, prices, days)
        check_one_over_n(eng, prices)
        gate_b6(eng)
        fixture2_tables(eng, days)
    except AssertionError as e:
        lines.extend([
            "",
            f"**GATE FAILURE — STOP**: {e}",
            "",
            f"Gates passed before the failure: {_PASS or 'none'}. Diagnosis "
            "recorded; nothing patched; Stage C is not executed on a failed "
            "Stage B.",
        ])
        report = "\n".join(lines)
        (OUT_DIR / "stage_b_report.md").write_text(report, encoding="utf-8")
        print(report)
        sys.exit(1)

    lines.extend([
        "## Deliverables",
        "",
        "- `src/task11/stage_b.py`",
        "- `outputs/task11/stage_b_report.md`",
        "",
        f"**STOP — Stage B boundary.** Gates {', '.join(_PASS)} all PASS. "
    ])
    report = "\n".join(lines)
    (OUT_DIR / "stage_b_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
