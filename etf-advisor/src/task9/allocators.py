"""Task 9 Stage C: allocator interface, four allocators, rebalance grid, smoke test.

Interface contract (consumed by the Task 11 backtester):

    get_allocator(allocator_id, profile) -> Allocator
    Allocator.weights(t) -> pd.Series      # 8 tickers, sum 1, all >= 0

``allocator_id`` in {"one_over_n", "static_bucket", "rb_classical",
"rb_advisor"}; ``profile`` in {"conservative", "balanced", "growth"} (exactly
these lowercase strings -- the Task 14 elicitor will emit one of them;
``one_over_n`` accepts and ignores it). Point-in-time contract: ``weights(t)``
may use only information realised by t (prices/returns through t, OLS fits on
vol_label_date <= t). The backtester supplies nothing but t.

Pre-registered:

  * ``STATIC_BUCKET`` -- capital weights on a 30/55/80 equity ladder, split
    equally within each asset class (no per-ticker discretion to defend).
  * ``RISK_BUDGETS``  -- risk-contribution budgets on a 35/60/75 equity-risk
    ladder, split equally within each class; all b_i > 0 guarantees an
    interior long-only solution.
  * Risk-budget solver -- minimise F(w) = 0.5 w'Sw - sum_i b_i ln(w_i) over
    w > 0 with S~ = S / mean(diag(S)) (conditioning rescale; the normalised
    solution is scale-invariant), via scipy L-BFGS-B, x0 = 1/8, bounds
    [1e-8, inf), gtol = 1e-12, maxiter = 500, then w <- w / sum(w).
    Implementation notes on unpinned internals: the gradient is supplied
    analytically (exact, not finite-differenced), and ftol is set to 1e-18 so
    that scipy's default f-based stop (ftol ~ 2.2e-9) cannot preempt the
    registered gtol criterion.

Gates: 
C1 -- max_i |RC_i - b_i| < 1e-6 at every evaluation (asserted inside
the solve, on the original Sigma; RC is identical under the rescale).
C2 -- at the four smoke dates, solving with S and with S~ yields the same
normalised w to < 1e-6. 
C3 -- the monthly rebalance grid starts 2012-01-03,
counts 156 dates to 2024-12, and contains the four smoke dates. On any gate
failure: STOP, diagnosis to the stage report.

The smoke test evaluates allocators at four spot dates only -- it is not a
backtest (backtesting is Task 11).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data.config import ROOT, UNIVERSE
from .covariance import sigma
from .forecaster_deploy import _prices_index

OUT_DIR = ROOT / "outputs" / "task9"

ALLOCATOR_IDS = ("one_over_n", "static_bucket", "rb_classical", "rb_advisor")
PROFILES = ("conservative", "balanced", "growth")
SMOKE_DATES = ("2012-02-01", "2020-03-02", "2022-06-01", "2024-12-02")

GATE_C1_TOL = 1e-6
GATE_C2_TOL = 1e-6
SUM_TOL = 1e-9
GRID_START = "2012-01-03"
GRID_COUNT_2012_2024 = 156

# C.3 -- capital weights, 30/55/80 equity ladder, equal within class.
STATIC_BUCKET = pd.DataFrame(
    {
        "conservative": [0.100, 0.100, 0.050, 0.050, 0.300, 0.300, 0.050, 0.050],
        "balanced":     [0.175, 0.175, 0.100, 0.100, 0.175, 0.175, 0.050, 0.050],
        "growth":       [0.275, 0.275, 0.125, 0.125, 0.050, 0.050, 0.050, 0.050],
    },
    index=list(UNIVERSE),
)

# C.4 -- risk-contribution budgets, 35/60/75 equity-risk ladder, equal within class.
RISK_BUDGETS = pd.DataFrame(
    {
        "conservative": [0.125, 0.125, 0.050, 0.050, 0.250, 0.250, 0.075, 0.075],
        "balanced":     [0.200, 0.200, 0.100, 0.100, 0.125, 0.125, 0.075, 0.075],
        "growth":       [0.250, 0.250, 0.125, 0.125, 0.050, 0.050, 0.075, 0.075],
    },
    index=list(UNIVERSE),
)

for _table in (STATIC_BUCKET, RISK_BUDGETS):
    assert np.allclose(_table.sum(axis=0), 1.0), "pre-registered table does not sum to 1"


# --------------------------------------------------------------------------- #
# C.4 risk-budget solver (pre-registered, deterministic)
# --------------------------------------------------------------------------- #
def solve_risk_budget(
    S: pd.DataFrame, budgets: pd.Series, rescale: bool = True
) -> pd.Series:
    """One registered solve; ``rescale=False`` exists only for the Gate C2 check."""
    tickers = list(UNIVERSE)
    S_arr = S.loc[tickers, tickers].to_numpy()
    b = budgets.reindex(tickers).to_numpy()
    M = S_arr / np.mean(np.diag(S_arr)) if rescale else S_arr

    def fun(w: np.ndarray) -> float:
        return 0.5 * (w @ M @ w) - b @ np.log(w)

    def jac(w: np.ndarray) -> np.ndarray:
        return M @ w - b / w

    res = minimize(
        fun,
        np.full(len(b), 1.0 / len(b)),
        jac=jac,
        method="L-BFGS-B",
        bounds=[(1e-8, None)] * len(b),
        options={"gtol": 1e-12, "maxiter": 500, "ftol": 1e-18},
    )
    w = res.x / res.x.sum()

    # Gate C1, at every evaluation, on the original Sigma.
    Sw = S_arr @ w
    rc = w * Sw / (w @ Sw)
    err = float(np.abs(rc - b).max())
    assert err < GATE_C1_TOL, (
        f"GATE C1: max|RC_i - b_i| = {err:.3e} >= {GATE_C1_TOL} "
        f"(rescale={rescale}, solver status {res.status}: {res.message}) -- STOP"
    )
    return pd.Series(w, index=tickers)


# --------------------------------------------------------------------------- #
# C.1 interface and the four allocators
# --------------------------------------------------------------------------- #
class Allocator:
    """Point-in-time contract: weights(t) uses only information realised by t."""

    def weights(self, t: pd.Timestamp) -> pd.Series:
        raise NotImplementedError


class _OneOverN(Allocator):
    def weights(self, t: pd.Timestamp) -> pd.Series:
        return pd.Series(1.0 / len(UNIVERSE), index=list(UNIVERSE))


class _StaticBucket(Allocator):
    def __init__(self, profile: str) -> None:
        self._w = STATIC_BUCKET[profile]

    def weights(self, t: pd.Timestamp) -> pd.Series:
        # Fixed table at every t; drift-and-rebalance mechanics are Task 11.
        return self._w.copy()


class _RiskBudget(Allocator):
    def __init__(self, variant: str, profile: str) -> None:
        self._variant = variant
        self._b = RISK_BUDGETS[profile]

    def weights(self, t: pd.Timestamp) -> pd.Series:
        return solve_risk_budget(sigma(pd.Timestamp(t), self._variant), self._b)


def get_allocator(allocator_id: str, profile: str | None = None) -> Allocator:
    if allocator_id not in ALLOCATOR_IDS:
        raise ValueError(f"unknown allocator_id {allocator_id!r}")
    if allocator_id == "one_over_n":
        return _OneOverN()  # accepts and ignores profile
    if profile not in PROFILES:
        raise ValueError(
            f"{allocator_id} requires profile in {PROFILES}, got {profile!r}"
        )
    if allocator_id == "static_bucket":
        return _StaticBucket(profile)
    return _RiskBudget(
        "classical" if allocator_id == "rb_classical" else "advisor", profile
    )


# --------------------------------------------------------------------------- #
# C.5 rebalance grid
# --------------------------------------------------------------------------- #
def rebalance_grid(start: str = GRID_START, end: str | None = None) -> pd.DatetimeIndex:
    """First trading day of each calendar month, from the prices.parquet index."""
    idx = _prices_index()
    firsts = idx.to_series().groupby(idx.to_period("M")).min()
    grid = pd.DatetimeIndex(firsts.to_numpy())
    grid = grid[grid >= pd.Timestamp(start)]
    if end is not None:
        grid = grid[grid <= pd.Timestamp(end)]
    return grid


# --------------------------------------------------------------------------- #
# Gates and smoke test
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.6f}") -> str:
    body = df.copy()
    for c in body.columns:
        if body[c].dtype.kind == "f":
            body[c] = body[c].map(lambda v: floatfmt.format(v))
    header = "| " + " | ".join(body.columns) + " |"
    sep = "|" + "|".join(["---"] * len(body.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in body.to_numpy()]
    return "\n".join([header, sep, *rows])


def gate_c3(lines: list[str]) -> None:
    grid = rebalance_grid()
    assert grid[0] == pd.Timestamp(GRID_START), (
        f"grid starts {grid[0].date()}, expected {GRID_START} -- STOP"
    )
    missing = [s for s in SMOKE_DATES if pd.Timestamp(s) not in grid]
    assert not missing, f"smoke dates missing from grid: {missing} -- STOP"
    n = len(grid[grid <= pd.Timestamp("2024-12-31")])
    assert n == GRID_COUNT_2012_2024, (
        f"grid 2012-01..2024-12 has {n} dates, expected {GRID_COUNT_2012_2024} -- STOP"
    )
    lines.append(
        f"**GATE C3 PASS** — rebalance grid derived from the prices.parquet "
        f"index: first element {grid[0].date()}, {n} dates 2012-01..2024-12, "
        f"contains all four smoke dates."
    )


def smoke_portfolios() -> pd.DataFrame:
    """4 dates x 10 portfolios; mechanical gates asserted per portfolio."""
    rows = []
    for date in SMOKE_DATES:
        t = pd.Timestamp(date)
        specs = [("one_over_n", None)] + [
            (aid, p)
            for aid in ("static_bucket", "rb_classical", "rb_advisor")
            for p in PROFILES
        ]
        for aid, prof in specs:
            w = get_allocator(aid, prof).weights(t)
            gap = abs(float(w.sum()) - 1.0)
            assert gap < SUM_TOL, f"{aid}/{prof}@{date}: |sum(w)-1| = {gap:.3e} -- STOP"
            assert (w >= 0).all(), f"{aid}/{prof}@{date}: negative weight -- STOP"
            rows.append(
                {"date": date, "allocator_id": aid, "profile": prof or "",
                 **{tk: float(w[tk]) for tk in UNIVERSE}}
            )
    return pd.DataFrame(rows)


def gate_variants_differ(lines: list[str], smoke: pd.DataFrame) -> None:
    tick_cols = list(UNIVERSE)
    diffs = []
    for date in SMOKE_DATES:
        for prof in PROFILES:
            w_cl = smoke.query(
                "date == @date and allocator_id == 'rb_classical' and profile == @prof"
            )[tick_cols].to_numpy()[0]
            w_ad = smoke.query(
                "date == @date and allocator_id == 'rb_advisor' and profile == @prof"
            )[tick_cols].to_numpy()[0]
            d = float(np.abs(w_cl - w_ad).max())
            assert d > 1e-12, (
                f"rb_classical == rb_advisor at {date}/{prof} -- D wiring broken, STOP"
            )
            diffs.append(d)
    lines.append(
        f"**GATE (variant difference) PASS** — rb_classical and rb_advisor "
        f"weights differ at every date/profile; max|dw| ranges "
        f"{min(diffs):.4f} .. {max(diffs):.4f}."
    )


def gate_c2(lines: list[str]) -> None:
    rows, worst, worst_at = [], 0.0, None
    for date in SMOKE_DATES:
        for variant in ("advisor", "classical"):
            S = sigma(date, variant)
            for prof in PROFILES:
                w1 = solve_risk_budget(S, RISK_BUDGETS[prof], rescale=True)
                w0 = solve_risk_budget(S, RISK_BUDGETS[prof], rescale=False)
                d = float(np.abs(w1 - w0).max())
                if d > worst:
                    worst, worst_at = d, (date, variant, prof)
                rows.append(
                    {"date": date, "variant": variant, "profile": prof,
                     "max_abs_w_diff": f"{d:.3e}",
                     f"pass_{GATE_C2_TOL:.0e}": "yes" if d < GATE_C2_TOL else "NO"}
                )
    lines += [
        "",
        "Gate C2 evidence (scale invariance, all 24 rb solves at the smoke dates):",
        "",
        _md_table(pd.DataFrame(rows)),
        "",
    ]
    assert worst < GATE_C2_TOL, (
        f"GATE C2: max normalised-w disagreement between the Sigma and "
        f"Sigma-tilde solves is {worst:.3e} >= {GATE_C2_TOL} -- STOP"
    )
    lines.append(
        f"**GATE C2 PASS** — worst disagreement {worst!r} (= {worst:.3e}) at "
        f"{worst_at}, under the amended threshold {GATE_C2_TOL:g}. Full "
        f"precision printed for the determinism cross-check against the "
        f"pre-amendment failure record."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task 9 Stage C report — allocators and interface",
        "",
        "Four allocators behind one point-in-time interface; risk-budget solver "
        "per the pre-registered spec (L-BFGS-B, x0=1/8, bounds [1e-8, inf), "
        "gtol=1e-12, maxiter=500; analytic gradient; ftol=1e-18 so the "
        "registered gtol governs termination). Smoke test = 4 spot dates x 10 "
        "portfolios; not a backtest.",
        "",
        "## Gates",
        "",
    ]
    try:
        gate_c3(lines)
        smoke = smoke_portfolios()
        lines.append(
            "**GATE C1 PASS** — max|RC_i − b_i| < 1e-6 asserted inside "
            "every risk-budget solve of the smoke evaluation (24 production "
            "solves)."
        )
        gate_variants_differ(lines, smoke)
        lines.append(
            "**Mechanical gates PASS** — all 40 portfolios: |sum(w)−1| < 1e-9, "
            "all w ≥ 0."
        )

        lines += ["", "## Smoke weight tables (10 portfolios per date)", ""]
        for date in SMOKE_DATES:
            g = smoke[smoke["date"] == date].drop(columns="date")
            lines += [f"### {date}", "", _md_table(g), ""]

        bond = ["AGG", "IEF"]
        eq = ["SPY", "QQQ", "EFA", "EEM"]
        obs = []
        for date in SMOKE_DATES:
            for var in ("rb_classical", "rb_advisor"):
                row = smoke.query(
                    "date == @date and allocator_id == @var and profile == 'conservative'"
                ).iloc[0]
                obs.append(
                    f"- {date} {var} conservative: bond sleeve "
                    f"{row[bond].sum():.3f}, equity sleeve {row[eq].sum():.3f}."
                )
        eq_2012, eq_2020 = {}, {}
        for var in ("rb_classical", "rb_advisor"):
            for date, store in (("2012-02-01", eq_2012), ("2020-03-02", eq_2020)):
                r = smoke.query(
                    "date == @date and allocator_id == @var and profile == 'balanced'"
                ).iloc[0]
                store[var] = float(r[eq].sum())
        obs.append(
            f"- Balanced equity sleeve, 2012-02-01 vs 2020-03-02: rb_classical "
            f"{eq_2012['rb_classical']:.3f} -> {eq_2020['rb_classical']:.3f}, "
            f"rb_advisor {eq_2012['rb_advisor']:.3f} -> {eq_2020['rb_advisor']:.3f}."
        )

        gate_c2(lines)

        # Reached only if every gate passed.
        smoke.to_csv(OUT_DIR / "smoke_weights.csv", index=False)
        anchor = smoke.query(
            "date == '2024-12-02' and allocator_id == 'rb_advisor' "
            "and profile == 'balanced'"
        ).iloc[0]
        lines += [
            "",
            "## Checksum record — rb_advisor balanced @ 2024-12-02 (Task 11 anchor, full precision)",
            "",
            *[f"- {tk} = `{float(anchor[tk])!r}`" for tk in UNIVERSE],
            "",
            "## Deliverables",
            "",
            "- `src/task9/allocators.py`",
            "- `outputs/task9/smoke_weights.csv` (40 rows)",
            "- `outputs/task9/stage_c_report.md`",
            "",
            "**STOP — Stage C boundary.** Gates C1–C3 and smoke tables reported above."
        ]
    except AssertionError as e:
        lines += ["", f"**GATE FAILURE — STOP**: {e}", "",
                  "Diagnosis recorded. "
                  "smoke_weights.csv and the checksum record are NOT written on "
                  "a failed stage."]
        report = "\n".join(lines)
        (OUT_DIR / "stage_c_report.md").write_text(report, encoding="utf-8")
        print(report)
        sys.exit(1)

    report = "\n".join(lines)
    (OUT_DIR / "stage_c_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
