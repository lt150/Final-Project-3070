"""Regenerate X2 from the frozen ``stability_probes.csv``.

Make X2 a two-panel exhibit:
the registered full-range log view UNCHANGED on top, and a linear-scale
companion over the observed band below, showing the SAME complete 96 points with
the threshold annotated as off-scale above. Legibility only -- no datum is
added, removed, or re-emphasised.

The ruling requires the regeneration to read the FROZEN artifact: no enumeration
is re-run and no L-BFGS-B solve is repeated. The rho values plotted here are the
ones already written by Stage C, parsed back at full repr precision. In-repo
precedent: the Task 11 defect DC-1 log, where F1 was rebuilt from the frozen
CSVs rather than by re-running the backtest.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone

from .exhibits import x2_stability
from .fixtures import (
    CONTROL_PAIR,
    MONITORED_PAIR,
    OUT_DIR,
    RHO_THRESHOLD,
    cells,
)
from .probes import CONTROL_PROBES, MONITORED_PROBES

CSV_PATH = OUT_DIR / "stability_probes.csv"
PAIR_LABEL = {
    MONITORED_PAIR: "-".join(MONITORED_PAIR),
    CONTROL_PAIR: "-".join(CONTROL_PAIR),
}


def load_frozen() -> tuple[dict, dict, int]:
    """(monitored, control, n_rows) rebuilt from the frozen Stage C artifact."""
    rows = list(csv.DictReader(CSV_PATH.read_text(encoding="utf-8").splitlines()))
    monitored: dict[tuple[str, str], list[tuple[str, float]]] = {c: [] for c in cells()}
    control: dict[tuple[str, str], list[tuple[str, float]]] = {c: [] for c in cells()}
    for r in rows:
        cell = (r["date"], r["profile"])
        assert cell in monitored, f"unregistered cell {cell} in {CSV_PATH.name} -- STOP"
        sink = monitored if r["pair"] == PAIR_LABEL[MONITORED_PAIR] else control
        assert r["pair"] in PAIR_LABEL.values(), (
            f"unregistered pair {r['pair']!r} in {CSV_PATH.name} -- STOP"
        )
        sink[cell].append((r["probe"], float(r["rho"])))

    # Completeness is re-asserted against the registered probe lists: the
    # regeneration must plot every point Stage C measured, not a subset.
    for cell in cells():
        got_m = sorted(n for n, _ in monitored[cell])
        got_c = sorted(n for n, _ in control[cell])
        assert got_m == sorted(p.name for p in MONITORED_PROBES), (
            f"{cell}: monitored probes {got_m} != registered -- STOP"
        )
        assert got_c == sorted(p.name for p in CONTROL_PROBES), (
            f"{cell}: control probes {got_c} != registered -- STOP"
        )
    return monitored, control, len(rows)


def main() -> None:
    monitored, control, n_rows = load_frozen()
    n_points = sum(len(v) for v in monitored.values()) + sum(
        len(v) for v in control.values()
    )
    path = x2_stability(
        list(cells()),
        monitored,
        control,
        RHO_THRESHOLD,
        f"monitored pair {PAIR_LABEL[MONITORED_PAIR]} (gated)",
        f"control pair {PAIR_LABEL[CONTROL_PAIR]} (descriptive)",
    )
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"regenerated {path}")
    print(f"  source        : {CSV_PATH} (frozen Stage C artifact, {n_rows} rows)")
    print(f"  points plotted: {n_points} (48 monitored + 48 control)")
    print(f"  amendment     : T12-A2, two-panel; no enumeration re-run")
    print(f"  interpreter   : {sys.executable}")
    print(f"  run           : {stamp}")


if __name__ == "__main__":
    main()
