"""Loaders bound to the frozen Task 9 / 10 / 12 artifacts.

Everything here is READ-ONLY.  Column names, the canonical ticker order, the
cell dates and the attribution matrix orientation are all *read from the CSV
headers*.

Two representations of every number are kept side by side:

- ``raw`` -- the exact character token as written in the frozen CSV.  Task 10
  and Task 12 wrote these with ``repr()``, so the token is the full-precision
  value.  Tie-out (a) checks the round trip.
- ``value`` -- the parsed ``float``, used for arithmetic (efficiency check,
  differences, rankings) and for ``decimal.Decimal`` adjudication downstream.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from . import (
    ANCHOR_CELL,
    ATTRIBUTIONS_MAIN,
    PROFILE_ORDER,
    SMOKE_RECOMMENDATIONS,
    SMOKE_WEIGHTS_CANDIDATES,
)

ADVISOR_ID = "rb_advisor"
CLASSICAL_ID = "rb_classical"


class GroundingError(RuntimeError):
    pass


def resolve_smoke_weights() -> Path:
    """Registered resolution rule: ``outputs/task10/`` first, then ``outputs/task9/``.

    If neither resolves, halt at Gate A2 and report -- no further searching and
    no substitution.
    """
    for candidate in SMOKE_WEIGHTS_CANDIDATES:
        if candidate.exists():
            return candidate
    raise GroundingError(
        "GATE A2: smoke_weights.csv resolved under neither "
        + " nor ".join(str(c) for c in SMOKE_WEIGHTS_CANDIDATES)
        + " -- the registered resolution rule is exhausted; HALT and report "
        "(no further search, no substitute) -- STOP"
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cell:
    date: str
    profile: str

    @property
    def key(self) -> str:
        return f"{self.date}_{self.profile}"

    @property
    def is_anchor(self) -> bool:
        return (self.date, self.profile) == ANCHOR_CELL


@dataclass
class Grounding:
    """Every frozen surface Task 13 is allowed to narrate from."""

    tickers: tuple[str, ...]
    cells: tuple[Cell, ...]
    smoke_weights_path: Path
    # (date, profile) -> ticker -> value / raw token
    advisor_w: dict[tuple[str, str], dict[str, float]]
    advisor_w_raw: dict[tuple[str, str], dict[str, str]]
    forecast_vols: dict[tuple[str, str], dict[str, float]]
    forecast_vols_raw: dict[tuple[str, str], dict[str, str]]
    classical_w: dict[tuple[str, str], dict[str, float]]
    classical_w_raw: dict[tuple[str, str], dict[str, str]]
    # (date, profile) -> (feature, output) -> value / raw token
    phi: dict[tuple[str, str], dict[tuple[str, str], float]]
    phi_raw: dict[tuple[str, str], dict[tuple[str, str], str]]
    # provenance of each grounding surfacw
    provenance: dict[str, str]

    def diff(self, cell: Cell) -> dict[str, float]:
        """w_advisor - w_classical, per ticker.  Permitted derivation."""
        a, c = self.advisor_w[(cell.date, cell.profile)], self.classical_w[(cell.date, cell.profile)]
        return {t: a[t] - c[t] for t in self.tickers}

    def phi_colsum(self, cell: Cell) -> dict[str, float]:
        """Column sums of the attribution matrix, per OUTPUT ticker."""
        table = self.phi[(cell.date, cell.profile)]
        return {
            o: sum(table[(f, o)] for f in self.tickers) for o in self.tickers
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [dict(zip(header, row)) for row in reader if row]
    return header, rows


def load() -> Grounding:
    if not SMOKE_RECOMMENDATIONS.exists():
        raise GroundingError(
            f"GATE A2: {SMOKE_RECOMMENDATIONS} not found -- STOP"
        )
    if not ATTRIBUTIONS_MAIN.exists():
        raise GroundingError(f"GATE A2: {ATTRIBUTIONS_MAIN} not found -- STOP")
    sw_path = resolve_smoke_weights()

    # -- smoke_recommendations.csv: advisor weights + forecast_vols ---------- #
    rec_header, rec_rows = _read_csv(SMOKE_RECOMMENDATIONS)
    w_cols = [c for c in rec_header if c.startswith("w_")]
    d_cols = [c for c in rec_header if c.startswith("d_")]
    tickers = tuple(c[2:] for c in w_cols)               # canonical order, READ
    if tuple(c[2:] for c in d_cols) != tickers:
        raise GroundingError(
            f"GATE A2: smoke_recommendations.csv d_* order "
            f"{tuple(c[2:] for c in d_cols)} != w_* order {tickers} -- STOP"
        )

    advisor_w: dict[tuple[str, str], dict[str, float]] = {}
    advisor_w_raw: dict[tuple[str, str], dict[str, str]] = {}
    fvols: dict[tuple[str, str], dict[str, float]] = {}
    fvols_raw: dict[tuple[str, str], dict[str, str]] = {}
    for r in rec_rows:
        key = (r["date"], r["profile"])
        advisor_w_raw[key] = {t: r[f"w_{t}"] for t in tickers}
        advisor_w[key] = {t: float(r[f"w_{t}"]) for t in tickers}
        fvols_raw[key] = {t: r[f"d_{t}"] for t in tickers}
        fvols[key] = {t: float(r[f"d_{t}"]) for t in tickers}

    dates = tuple(sorted({r["date"] for r in rec_rows}))   # READ, never retyped
    cells = tuple(Cell(d, p) for d in dates for p in PROFILE_ORDER)

    # -- smoke_weights.csv: classical benchmark ----------------------------- #
    sw_header, sw_rows = _read_csv(sw_path)
    sw_tickers = tuple(c for c in sw_header if c in set(tickers))
    if tuple(sw_header[3:]) != tickers:
        raise GroundingError(
            f"GATE A2: {sw_path.name} ticker columns {tuple(sw_header[3:])} != "
            f"canonical {tickers} read from smoke_recommendations.csv -- STOP"
        )
    classical_w: dict[tuple[str, str], dict[str, float]] = {}
    classical_w_raw: dict[tuple[str, str], dict[str, str]] = {}
    stored_advisor_raw: dict[tuple[str, str], dict[str, str]] = {}
    for r in sw_rows:
        key = (r["date"], r["profile"])
        if r["allocator_id"] == CLASSICAL_ID:
            classical_w_raw[key] = {t: r[t] for t in sw_tickers}
            classical_w[key] = {t: float(r[t]) for t in sw_tickers}
        elif r["allocator_id"] == ADVISOR_ID:
            stored_advisor_raw[key] = {t: r[t] for t in sw_tickers}

    # -- attributions_main.csv --------------------------------------------- #
    at_header, at_rows = _read_csv(ATTRIBUTIONS_MAIN)
    for required in ("date", "profile", "feature", "output", "phi"):
        if required not in at_header:
            raise GroundingError(
                f"GATE A2: attributions_main.csv header {at_header} lacks "
                f"{required!r} -- STOP"
            )
    phi: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    phi_raw: dict[tuple[str, str], dict[tuple[str, str], str]] = {}
    for r in at_rows:
        key = (r["date"], r["profile"])
        phi.setdefault(key, {})[(r["feature"], r["output"])] = float(r["phi"])
        phi_raw.setdefault(key, {})[(r["feature"], r["output"])] = r["phi"]

    g = Grounding(
        tickers=tickers,
        cells=cells,
        smoke_weights_path=sw_path,
        advisor_w=advisor_w,
        advisor_w_raw=advisor_w_raw,
        forecast_vols=fvols,
        forecast_vols_raw=fvols_raw,
        classical_w=classical_w,
        classical_w_raw=classical_w_raw,
        phi=phi,
        phi_raw=phi_raw,
        provenance={
            "advisor weights": f"{SMOKE_RECOMMENDATIONS.as_posix()} (w_* columns)",
            "forecast_vols": f"{SMOKE_RECOMMENDATIONS.as_posix()} (d_* columns)",
            "classical weights": f"{sw_path.as_posix()} (allocator_id={CLASSICAL_ID})",
            "attributions": f"{ATTRIBUTIONS_MAIN.as_posix()} (long: feature x output -> phi)",
            "cell dates": f"{SMOKE_RECOMMENDATIONS.as_posix()} (date column, sorted)",
            "canonical ticker order": f"{SMOKE_RECOMMENDATIONS.as_posix()} (w_* header order)",
        },
    )
    g._stored_advisor_raw = stored_advisor_raw          # type: ignore[attr-defined]
    g._attribution_row_count = len(at_rows)             # type: ignore[attr-defined]
    return g


# --------------------------------------------------------------------------- #
# Stage A tie-outs (a), (b), (c)
# --------------------------------------------------------------------------- #
def tieout_repr_exact(g: Grounding) -> list[tuple[str, str, int, int, str]]:
    """(a) Loader output reproduces the raw frozen CSV token repr-exactly.

    The frozen CSVs were written with ``repr()``, so ``repr(float(token)) ==
    token`` for every numeric cell iff nothing was lost on the way in.
    """
    rows: list[tuple[str, str, int, int, str]] = []
    surfaces = [
        ("advisor weights", g.advisor_w_raw, g.advisor_w),
        ("forecast_vols", g.forecast_vols_raw, g.forecast_vols),
        ("classical weights", g.classical_w_raw, g.classical_w),
    ]
    for name, raw_map, val_map in surfaces:
        checked = mismatched = 0
        first_bad = ""
        for key, raws in raw_map.items():
            for t, token in raws.items():
                checked += 1
                if repr(val_map[key][t]) != token:
                    mismatched += 1
                    first_bad = first_bad or f"{key}/{t}: {token} -> {val_map[key][t]!r}"
        rows.append((name, "repr(float(token)) == token", checked, mismatched,
                     first_bad or "-"))

    checked = mismatched = 0
    first_bad = ""
    for key, table in g.phi_raw.items():
        for fo, token in table.items():
            checked += 1
            if repr(g.phi[key][fo]) != token:
                mismatched += 1
                first_bad = first_bad or f"{key}/{fo}: {token} -> {g.phi[key][fo]!r}"
    rows.append(("attributions", "repr(float(token)) == token", checked,
                 mismatched, first_bad or "-"))
    return rows


def tieout_completeness(g: Grounding) -> list[tuple[str, str, str, str]]:
    """(b) 12 cells x 8 tickers present on every surface; 768 attribution rows."""
    n_cells, n_t = len(g.cells), len(g.tickers)
    rows: list[tuple[str, str, str, str]] = []

    def check(label: str, expected: object, got: object) -> None:
        rows.append((label, str(expected), str(got),
                     "PASS" if expected == got else "**FAIL**"))

    check("cells (dates x profiles)", 12, n_cells)
    check("tickers", 8, n_t)
    check("advisor-weight cells", n_cells,
          sum(1 for c in g.cells if (c.date, c.profile) in g.advisor_w))
    check("forecast_vol cells", n_cells,
          sum(1 for c in g.cells if (c.date, c.profile) in g.forecast_vols))
    check("classical-weight cells", n_cells,
          sum(1 for c in g.cells if (c.date, c.profile) in g.classical_w))
    check("attribution cells", n_cells,
          sum(1 for c in g.cells if (c.date, c.profile) in g.phi))
    check("advisor entries (cells x tickers)", n_cells * n_t,
          sum(len(v) for v in g.advisor_w.values()))
    check("forecast_vol entries", n_cells * n_t,
          sum(len(v) for v in g.forecast_vols.values()))
    check("classical entries", n_cells * n_t,
          sum(len(v) for v in g.classical_w.values()))
    check("attribution rows", n_cells * n_t * n_t,
          getattr(g, "_attribution_row_count"))
    check("attribution entries (cells x feature x output)", n_cells * n_t * n_t,
          sum(len(v) for v in g.phi.values()))
    check("anchor cell present", True,
          any(c.is_anchor for c in g.cells))
    # every (feature, output) pair populated on every cell
    holes = sum(
        1
        for c in g.cells
        for f in g.tickers
        for o in g.tickers
        if (f, o) not in g.phi.get((c.date, c.profile), {})
    )
    check("attribution (feature,output) holes", 0, holes)
    # profile-invariance of the d-vector at fixed date
    bad_d = 0
    for d in sorted({c.date for c in g.cells}):
        ref = g.forecast_vols_raw[(d, PROFILE_ORDER[0])]
        for p in PROFILE_ORDER[1:]:
            if g.forecast_vols_raw[(d, p)] != ref:
                bad_d += 1
    check("d-vector profile-invariance breaches", 0, bad_d)
    # the two frozen artifacts agree on the advisor weights
    stored = getattr(g, "_stored_advisor_raw")
    bad_adv = sum(
        1
        for c in g.cells
        for t in g.tickers
        if repr(float(stored[(c.date, c.profile)][t]))
        != repr(g.advisor_w[(c.date, c.profile)][t])
    )
    check("advisor weights: smoke_recommendations vs smoke_weights rb_advisor "
          "(repr-exact)", 0, bad_adv)
    return rows


def tieout_efficiency(g: Grounding, tol: float) -> tuple[list[list[object]], float]:
    """(c) Per (cell, output): sum_features phi == w_advisor - w_classical.

    Engine-arithmetic tolerance.  This reproduces the identity as 
    a LOADER SANITY CHECK -- it proves the loaders read the
    artifacts correctly and re-litigates nothing upstream.
    """
    rows: list[list[object]] = []
    worst = 0.0
    for c in g.cells:
        diff = g.diff(c)
        cols = g.phi_colsum(c)
        cell_worst = max(abs(cols[t] - diff[t]) for t in g.tickers)
        worst = max(worst, cell_worst)
        rows.append([
            c.date, c.profile, f"{cell_worst:.3e}",
            "PASS" if cell_worst <= tol else "**FAIL**",
        ])
    return rows, worst
